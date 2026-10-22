import argparse
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchaudio.transforms as AT
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from configs.config import (
    CHECKPOINT_DIR,
    HOP_LENGTH,
    MODEL_NAME,
    MODEL_VERSION,
    N_FFT,
    N_MELS,
    NUM_CLASSES,
    SAMPLE_RATE,
    SEED,
    TARGET_SAMPLES,
    checkpoint_dir,
)
from src import calibration
from src.dataset import SpeechCommandsDataset
from src.models import CommandSense
from src.transforms import INT16_NORMALIZER, AudioToThreeChannelMel
from train import set_seed

# torch.onnx prints non-ASCII glyphs; on a pipe (cp1252 on Windows) those prints
# would raise UnicodeEncodeError mid-export, so switch the streams to UTF-8 first.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic stream types
            pass

OPSET_VERSION = 17
INPUT_NAME = "raw_audio"
OUTPUT_NAME = "logits"
# The graph input is a plain [-1, 1] signal; the front-end rounds it onto the
# int16 grid the training cache uses and divides by 32768, exactly as
# src/transforms.py does. The two constants must stay in this relation:
# round(clip(x) * INT16_SCALE) / INT16_NORMALIZER == the training sample.
INT16_SCALE = 32767.0
LOG_EPS = 1e-6
DELTA_WINDOW = 5                       # AudioToThreeChannelMel's compute_deltas win_length
CLASSIFIER_CHANNELS = 3                # the first Conv that consumes the 3 feature channels
CALIBRATION_SPLIT = "validation"       # never the training split, never the testing one
CALIBRATION_CLIPS = 512


def artifact_paths(version: str):
    """``(float32_reference, int8_artifact)`` inside the version's checkpoint folder."""
    directory = checkpoint_dir(version)
    return (
        directory / f"{MODEL_NAME}_{version}_fp32.onnx",
        directory / f"{MODEL_NAME}_{version}.onnx",
    )


def resolve_checkpoint(version=MODEL_VERSION):
    """Return the path of the weights to export, canonical layout first."""
    candidates = [
        checkpoint_dir(version) / f"{MODEL_NAME}_{version}.pth",
        CHECKPOINT_DIR / f"{MODEL_NAME}_{version}.pth",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return str(candidate)
    return None


def resolve_calibration(version=MODEL_VERSION):
    """Return the calibration metadata of the deployed release, or ``{}``."""
    return calibration.load_calibration(version=version)


class RealValuedSpectrogram(torch.nn.Module):
    """The magnitude stage of ``AT.Spectrogram``, kept real-valued for ONNX.

    ``torch.stft`` returns a complex tensor and ONNX only defines a real-valued
    STFT (``onesided``, trailing axis 2), which is also the single ``STFT`` node
    ONNX Runtime Web can execute. So this module mirrors what
    ``AT.Spectrogram.forward`` does to the STFT output - ``|X|^2`` for
    ``power == 2``, ``|X|`` for ``power == 1`` - on ``view_as_real`` output,
    while every constant that defines the front-end (window, ``center``,
    ``pad_mode``, mel filterbank, log epsilon) keeps coming from the same
    torchaudio objects ``src.transforms`` builds. Only this op is re-expressed;
    nothing else in the front-end is.
    """

    def __init__(self, source: AT.Spectrogram):
        super().__init__()
        # Copied rather than re-derived: whatever the training front-end was
        # constructed with is what the graph must reproduce.
        self.n_fft = source.n_fft
        self.hop_length = source.hop_length
        self.win_length = source.win_length
        self.center = source.center
        self.pad_mode = source.pad_mode
        self.normalized = source.normalized
        self.onesided = source.onesided
        self.power = source.power
        self.register_buffer("window", source.window)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.dtype),
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=False,
        )
        power = stft.pow(2.0).sum(dim=-1)
        return power if self.power == 2.0 else power.pow(0.5)


class CommandSenseFrontend(torch.nn.Module):
    """``raw_audio`` ``[B, TARGET_SAMPLES]`` in [-1, 1] -> ``[B, 3, n_mels, frames]``.

    The serving convention baked into the graph, in order: clip the incoming
    signal to [-1, 1], put it on the int16 grid the training cache stores
    (``round(x * 32767)``), normalize exactly as ``AudioToThreeChannelMel``
    does (``/ 32768``), then mel -> ``log(x + 1e-6)`` -> delta -> delta-delta.

    Baking the gridding in is what lets a browser client send the samples it
    already has: no client-side int16 conversion, and no chance of a client
    picking a different rounding rule than training did.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
    ):
        super().__init__()
        # Same construction - and therefore the same defaults (power=2.0,
        # center=True, pad_mode='reflect', power-spectrum mel filterbank) as
        # AudioToThreeChannelMel.__init__.
        self.mel_spectrogram = AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.mel_spectrogram.spectrogram = RealValuedSpectrogram(
            self.mel_spectrogram.spectrogram
        )

        # torchaudio's compute_deltas with win_length=5, mode='replicate':
        # denom = sum(i^2 for i in -2..2) = 10. The kernel is a registered
        # buffer so it travels with the module, and it is applied as a
        # depthwise conv1d over the time axis of every mel band.
        half = DELTA_WINDOW // 2
        kernel = torch.arange(-half, half + 1, dtype=torch.float32).reshape(1, 1, -1)
        self.register_buffer("delta_kernel", kernel.repeat(n_mels, 1, 1))
        self.delta_denom = float(half * (half + 1) * (2 * half + 1) / 3.0)

    def delta(self, features: torch.Tensor) -> torch.Tensor:
        """``torchaudio.functional.compute_deltas`` for ``win_length=5``.

        Re-expressed as ``Pad`` (mode ``edge``) + grouped ``Conv`` because
        ``compute_deltas`` flattens the batch/channel axes into one dimension,
        which the exporter would have to undo with reshapes it cannot always
        fold. Both ops are standard ONNX Runtime Web kernels.
        """
        padded = F.pad(features, (DELTA_WINDOW // 2,) * 2, mode="replicate")
        delta = F.conv1d(padded, self.delta_kernel, groups=self.delta_kernel.shape[0])
        return delta / self.delta_denom

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        if raw_audio.ndim == 3:  # tolerate a [B, 1, T] caller
            raw_audio = raw_audio.squeeze(1)
        pcm = torch.round(torch.clamp(raw_audio, -1.0, 1.0) * INT16_SCALE)
        log_mel = torch.log(self.mel_spectrogram(pcm / INT16_NORMALIZER) + LOG_EPS)
        delta = self.delta(log_mel)
        return torch.stack((log_mel, delta, self.delta(delta)), dim=1)


class CommandSenseONNX(torch.nn.Module):
    """The deployed graph in one module: raw audio in, raw logits out.

    Wrapping the classifier instead of editing it keeps ``src/models.py`` the
    training-time definition - the checkpoint loads unchanged, and the
    quantization pass can still see the classifier's own submodules.
    """

    def __init__(self, classifier: torch.nn.Module, frontend=None):
        super().__init__()
        self.frontend = frontend if frontend is not None else CommandSenseFrontend()
        self.classifier = classifier

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.frontend(raw_audio))


def load_classifier(checkpoint_path):
    """Load the release weights on CPU, in eval mode, ready to trace."""
    device = torch.device("cpu")
    model = CommandSense(num_classes=NUM_CLASSES, in_channels=CLASSIFIER_CHANNELS)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()


def export_fp32_graph(wrapper, onnx_path):
    """Trace ``wrapper`` into a float32 ONNX graph with a dynamic batch axis.

    ``dynamo=False`` pins the TorchScript exporter: the dynamo path needs the
    STFT and the ``Pad``/``Conv`` delta re-expression to be lowered through
    onnxscript, and the traced graph is the one the quantizer's node-name
    exclusions are written against.
    """
    torch.onnx.export(
        wrapper,
        torch.zeros(1, TARGET_SAMPLES),
        str(onnx_path),
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAME: {0: "batch_size"},
            OUTPUT_NAME: {0: "batch_size"},
        },
    )
    print(f"[+] float32 reference exported -> {onnx_path}")


def check_frontend_parity(frontend, clips=8):
    """The baked front-end must reproduce ``AudioToThreeChannelMel`` exactly.

    Runs the torch front-end and the training-time transform on the same
    waveforms and compares all three channels: a mismatch here would mean the
    graph scores features the checkpoint never saw, which no quantization
    tolerance could excuse - so it is asserted before an artifact is written.
    The material is a seeded mix of noise and a tone, covering both a flat and
    a structured spectrum without touching the dataset.
    """
    reference = AudioToThreeChannelMel().eval()
    generator = torch.Generator().manual_seed(SEED)
    time = torch.arange(TARGET_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
    waveforms = []
    for index in range(clips):
        noise = torch.randn(1, TARGET_SAMPLES, generator=generator) * 0.1
        tone = torch.sin(2 * torch.pi * 110.0 * (index + 1) * time)
        waveforms.append(torch.clamp(noise + 0.3 * tone, -1.0, 1.0))
    batch = torch.cat(waveforms, dim=0)

    with torch.inference_mode():
        baked = frontend(batch)
        training = reference(torch.round(batch * INT16_SCALE))  # the cache's grid

    assert baked.shape == training.shape, (
        f"[!] Export Error: baked front-end yields {tuple(baked.shape)}, the "
        f"training front-end {tuple(training.shape)}"
    )
    drift = (baked - training).abs().max().item()
    assert drift < 1e-4, (
        f"[!] Export Error: baked front-end drifts {drift:.3e} from "
        f"AudioToThreeChannelMel - the graph would score features the "
        f"checkpoint never saw"
    )
    print(
        f"[+] Baked front-end matches AudioToThreeChannelMel: "
        f"{tuple(baked.shape)}, max |d| {drift:.2e}"
    )


def frontend_node_names(graph):
    """Names of the nodes the quantizer must leave in float32.

    The baked front-end ends where the classifier starts: at the first ``Conv``
    whose weight consumes ``CLASSIFIER_CHANNELS`` input channels. Everything
    before it holds preprocessing constants (int16 scale, mel filterbank, log
    epsilon, delta kernel) that quantizing would round away, and the INT8 win
    the release claims comes from the classifier's weights anyway.

    Exclusions are passed to ONNX Runtime by node *name*, so unnamed nodes
    (``Constant``s, which the quantizer never touches) are counted and skipped
    rather than silently dropped - if a quantizable front-end node ever came
    out anonymous, the count printed here is the trace of why.
    """
    channels = {}
    for initializer in graph.initializer:
        channels[initializer.name] = list(initializer.dims)

    excluded = []
    anonymous = 0
    for node in graph.node:
        if node.op_type == "Conv" and len(node.input) > 1:
            weight_dims = channels.get(node.input[1])
            if weight_dims and len(weight_dims) == 4 and weight_dims[1] == CLASSIFIER_CHANNELS:
                break  # first classifier layer: the front-end is over
        if node.name:
            excluded.append(node.name)
        else:
            anonymous += 1
    if anonymous:
        print(f"[i] {anonymous} anonymous front-end nodes carry no name to exclude")
    return excluded


def waveform_batch(audio):
    """``[1, TARGET_SAMPLES]`` float32 in [-1, 1] from one cached clip.

    The dataset returns int16 PCM (``return_audio=True``); dividing by the
    rounding scale inverts the gridding the front-end applies, so the clip
    enters the graph exactly as it did in training.
    """
    raw = torch.as_tensor(audio)
    integer_grid = raw.dtype in (torch.int16, torch.int32, torch.int64)
    samples = raw.to(torch.float32).reshape(-1)[:TARGET_SAMPLES]
    if samples.numel() < TARGET_SAMPLES:
        samples = F.pad(samples, (0, TARGET_SAMPLES - samples.numel()))
    if integer_grid:
        samples = samples / INT16_SCALE
    return samples.reshape(1, TARGET_SAMPLES).numpy().astype(np.float32)


class WaveformCalibrationReader(CalibrationDataReader):
    """Feeds the quantizer raw 1.0 s windows, the way a client will.

    Calibration has to observe the graph *input*: the deployed graph takes
    microphone samples, so the ranges the MinMax pass records must come from
    waveforms - feeding features would describe tensors the graph never sees
    and skew every activation range downstream of the front-end.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.cursor = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def get_next(self):
        if self.cursor >= len(self.indices):
            return None
        window = waveform_batch(self.dataset[self.indices[self.cursor]][0])
        self.cursor += 1
        self.minimum = min(self.minimum, float(window.min()))
        self.maximum = max(self.maximum, float(window.max()))
        return {INPUT_NAME: window}

    def rewind(self):
        self.cursor = 0


def load_calibration_reader(limit=CALIBRATION_CLIPS, split=CALIBRATION_SPLIT):
    """Build a reader over a seeded draw of the calibration split.

    ``CALIBRATION_SPLIT`` is the validation split: it trained neither the
    weights nor the T/tau pair, and the testing split is left untouched so the
    numbers ``tests/test_v3_1_quantization.py`` prints stay honest.
    """
    set_seed(SEED)
    dataset = SpeechCommandsDataset(subset=split, cache_in_ram=False, return_audio=True)
    size = len(dataset)
    if not limit or limit >= size:
        indices = list(range(size))
    else:
        chosen = np.random.default_rng(SEED).choice(size, size=limit, replace=False)
        indices = [int(index) for index in chosen]
    print(f"[+] Calibration: {len(indices)} of {size} '{split}' clips -> '{INPUT_NAME}'")
    return WaveformCalibrationReader(dataset, indices)


def embed_calibration_metadata(onnx_path: str, version: str = MODEL_VERSION):
    """Attach the fitted calibration to the graph's ``metadata_props``."""
    metadata = resolve_calibration(version)
    if not metadata:
        print(
            "[!] No calibration file found in configs/calibration/, so the artifact carries "
            "no T/tau metadata. Run evaluate.py first; the runtime then falls back on the "
            "defaults."
        )
        return

    props = calibration.metadata_to_onnx_props(metadata)
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)

    # Idempotent: refresh the calibration namespace, keep foreign entries (producer).
    prefix = calibration.ONNX_METADATA_PREFIX
    kept = [prop for prop in onnx_model.metadata_props if not prop.key.startswith(prefix)]
    del onnx_model.metadata_props[:]
    for key, value in [(prop.key, prop.value) for prop in kept] + list(props.items()):
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, str(value)

    onnx.save(onnx_model, str(onnx_path))
    print(
        f"[+] Embedded calibration metadata (T={metadata.get('temperature')}, "
        f"tau={metadata.get('confidence_threshold')}) under '{prefix}*'"
    )


def read_calibration_metadata(onnx_path):
    """Read the ``commandsense.calibration.*`` block back out of a graph.

    The release claims the T/tau of the shipped artifact can be recovered from
    the graph alone, so the claim is verifiable: the values are JSON-decoded
    out of a namespace no other producer writes to, exactly as a single-file
    runtime (an ONNX Runtime Web client) reads them. An empty dict means the
    graph documents nothing.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    prefix = calibration.ONNX_METADATA_PREFIX
    payload = {}
    for prop in model.metadata_props:
        if not prop.key.startswith(prefix):
            continue
        key = prop.key[len(prefix):]
        try:
            payload[key] = json.loads(prop.value)
        except json.JSONDecodeError:  # pragma: no cover - free text stays text
            payload[key] = prop.value
    return payload


def quantize_graph(fp32_path, int8_path, reader, nodes_to_exclude=None):
    """Static QDQ INT8: per-channel int8 weights, int8 activations, MinMax ranges.

    QDQ (not the older QOperator format) is what ONNX Runtime Web's WASM SIMD
    and WebNN kernels consume, and per-channel weights keep the classifier's
    convolutions from sharing one scale across kernels.
    """
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=list(nodes_to_exclude or []),
    )
    print(f"[+] INT8 QDQ graph written -> {int8_path}")


def verify_graph(onnx_path):
    """Structural contract of a deployment artifact: raw audio in, logits out.

    Both artifacts must declare exactly one input, ``raw_audio`` of rank 2 on a
    dynamic batch axis with ``TARGET_SAMPLES`` samples, and exactly one output
    of ``NUM_CLASSES`` logits. Anything else means a client would have to
    preprocess or postprocess - the point of baking the front-end is that it
    does not.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(model)

    external = [
        initializer.name
        for initializer in model.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
        or initializer.external_data
    ]
    assert not external, (
        f"[!] Export Error: {onnx_path} carries external tensor data: {external}"
    )

    assert len(model.graph.input) == 1, (
        f"[!] Export Error: expected one graph input, got {len(model.graph.input)}"
    )
    declared = model.graph.input[0].type.tensor_type.shape.dim
    assert len(declared) == 2, (
        f"[!] Export Error: expected a rank-2 input '[batch, {TARGET_SAMPLES}]' "
        f"(raw audio), got rank {len(declared)}"
    )
    assert declared[-1].dim_value == TARGET_SAMPLES, (
        f"[!] Export Error: expected {TARGET_SAMPLES} input samples, got "
        f"{declared[-1].dim_value}"
    )
    assert declared[0].dim_param, (
        "[!] Export Error: the graph lost its dynamic batch axis - export and "
        "quantization must stay batch-size agnostic"
    )

    assert len(model.graph.output) == 1, (
        f"[!] Export Error: expected a single output, got {len(model.graph.output)}"
    )
    logits = model.graph.output[0].type.tensor_type.shape.dim
    assert logits[-1].dim_value == NUM_CLASSES, (
        f"[!] Export Error: expected {NUM_CLASSES} logits, got {logits[-1].dim_value}"
    )
    print(
        f"[+] {os.path.basename(str(onnx_path))}: "
        f"{model.graph.input[0].name} "
        f"[{declared[0].dim_param}, {declared[-1].dim_value}] -> "
        f"{NUM_CLASSES} raw logits (dynamic batch)"
    )


def check_runtime_contract(fp32_path, int8_path, clips=16):
    """Run both artifacts on dummy waveforms, the way a client will.

    A structurally valid graph is worthless if the front-end it bakes is wired
    wrongly, so this executes the float32 reference and the INT8 copy through
    ONNX Runtime on seeded material (silence, a tone, noise) and checks shapes,
    finiteness and how far quantization moved the decision. The agreement floor
    is a collapse guard, not a fidelity claim: these waveforms are outside the
    training distribution by construction, and the fidelity numbers come from
    the calibration split in ``tests/test_v3_1_quantization.py``.
    """
    sessions = [
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for path in (fp32_path, int8_path)
    ]
    input_name = sessions[0].get_inputs()[0].name
    output_name = sessions[0].get_outputs()[0].name

    generator = np.random.default_rng(SEED)
    time = np.arange(TARGET_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    windows = [
        np.zeros(TARGET_SAMPLES, dtype=np.float32),
        (0.5 * np.sin(2 * np.pi * 440.0 * time)).astype(np.float32),
    ]
    for _ in range(clips):
        windows.append((0.1 * generator.standard_normal(TARGET_SAMPLES)).astype(np.float32))
    batch = np.stack(windows).astype(np.float32)

    logits = [
        session.run([output_name], {input_name: batch})[0] for session in sessions
    ]
    for name, values in zip(("float32", "int8"), logits):
        assert values.shape == (batch.shape[0], NUM_CLASSES), (
            f"[!] Export Error: the {name} graph answered {values.shape} for "
            f"{batch.shape[0]} waveforms, expected ({batch.shape[0]}, {NUM_CLASSES})"
        )
        assert np.isfinite(values).all(), (
            f"[!] Export Error: the {name} graph produced non-finite logits"
        )

    agreement = float((logits[0].argmax(axis=1) == logits[1].argmax(axis=1)).mean())
    drift = float(np.abs(logits[0] - logits[1]).max())
    print(
        f"[+] ONNX Runtime check on {batch.shape[0]} dummy waveforms: "
        f"{NUM_CLASSES} logits each, top-1 agreement {agreement * 100:.1f}%, "
        f"max |d logit| {drift:.2f}"
    )
    assert agreement >= 0.5, (
        f"[!] Export Error: the INT8 graph only agrees with float32 on "
        f"{agreement * 100:.1f}% of dummy waveforms - the quantized graph is "
        f"not tracking the reference"
    )


def main(argv=None):
    """Export one release: baked front-end, float32 reference, INT8 artifact."""
    parser = argparse.ArgumentParser(
        description="Export the deployment graph of one release: the waveform "
        "front-end baked in, then quantized to INT8."
    )
    parser.add_argument(
        "--version", default=MODEL_VERSION, help="release to export (default: %(default)s)"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="weights to export (default: the version's own checkpoint)",
    )
    parser.add_argument(
        "--calibration-clips",
        type=int,
        default=CALIBRATION_CLIPS,
        help="validation clips used to fit the INT8 ranges; 0 uses the whole split "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--fp32-only",
        action="store_true",
        help="write the float32 reference only and skip quantization",
    )
    args = parser.parse_args(argv)

    checkpoint_path = args.checkpoint or resolve_checkpoint(args.version)
    if checkpoint_path is None:
        print(
            f"[!] Error: no checkpoint found at "
            f"'{checkpoint_dir(args.version) / f'{MODEL_NAME}_{args.version}.pth'}'. "
            f"Download the {args.version} release bundle from the GitHub Releases "
            f"page or train that version first."
        )
        return 1
    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: checkpoint '{checkpoint_path}' does not exist.")
        return 1

    fp32_path, int8_path = artifact_paths(args.version)
    os.makedirs(str(fp32_path.parent), exist_ok=True)
    print(f"[*] Exporting {args.version} weights <- {checkpoint_path}")

    wrapper = CommandSenseONNX(load_classifier(checkpoint_path))
    check_frontend_parity(wrapper.frontend)
    export_fp32_graph(wrapper, fp32_path)

    graph = onnx.load(str(fp32_path), load_external_data=False).graph
    excluded = frontend_node_names(graph)
    print(
        f"[i] Quantizer keeps {len(excluded)} front-end nodes in float32; "
        f"the classifier is quantized"
    )
    verify_graph(fp32_path)
    embed_calibration_metadata(str(fp32_path), args.version)

    if args.fp32_only:
        print(f"[i] --fp32-only: skipped quantization, '{int8_path}' was not written")
        return 0

    reader = load_calibration_reader(args.calibration_clips)
    quantize_graph(fp32_path, int8_path, reader, nodes_to_exclude=excluded)
    print(
        f"[i] Calibration observed input range "
        f"[{reader.minimum:+.4f}, {reader.maximum:+.4f}] on "
        f"{len(reader.indices)} waveforms"
    )
    verify_graph(int8_path)
    embed_calibration_metadata(str(int8_path), args.version)
    check_runtime_contract(fp32_path, int8_path)

    external_data = f"{int8_path}.data"
    if os.path.exists(external_data):
        os.remove(external_data)

    fp32_mb = os.path.getsize(str(fp32_path)) / (1024 * 1024)
    int8_mb = os.path.getsize(str(int8_path)) / (1024 * 1024)
    print(
        f"[+] Size: float32 reference {fp32_mb:.3f} MB -> INT8 artifact "
        f"{int8_mb:.3f} MB ({100.0 * (1.0 - int8_mb / fp32_mb):.1f}% smaller)"
    )
    print(
        "[*] The graph stays raw-logit: T and tau are applied by src/inference.py, "
        "so they can change without a re-export."
    )
    print(f"[+] Deployment artifact ready -> {int8_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
