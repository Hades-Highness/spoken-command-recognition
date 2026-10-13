"""Static INT8 quantization of the v3.0 CommandSense graph (v3.1).

v3.0 ships a float32 ONNX graph; v3.1 keeps its weights frozen and produces the
INT8 sibling that the browser / edge runtime (v4.0, ONNX Runtime Web) will serve:

1. re-export the v3.0 checkpoint to ONNX (opset 17, dynamic batch axis) - this
   float32 graph is the reference the quantizer needs and the baseline every
   fidelity and latency number below is measured against,
2. calibrate the activation ranges on ``CALIBRATION_SAMPLES`` clips of the
   *validation* split, streamed one clip at a time through the v3.0 front-end so
   the same ``src.transforms.AudioToThreeChannelMel`` math trains and deploys,
3. quantize statically with ONNX Runtime post-training quantization in the
   ``QDQ`` format (per-channel int8 weights, int8 activations, MinMax ranges).
   QDQ is the format ONNX Runtime Web executes natively, which keeps the v4.0
   WASM/WebNN path open,
4. write the artifact to ``checkpoints/model_v3.1/`` and a human-readable report
   to ``reports/model_v3.1/quantization_report.txt``.

The run is reproducible: ``set_seed`` fixes the split draw - including the extra
``_silence_`` / ``_unknown_`` clips the v3.0 dataset samples from the global
generator - before calibration, so a rebuild of the same checkpoint reproduces the
same activation ranges.

Nothing here touches training or model code: the checkpoint is read-only and the
graph stays raw-logit, so the v3.0 temperature ``T`` and threshold ``tau`` from
``src/calibration.py`` still apply unchanged at runtime.

Run with:  python quantize_int8.py
"""

import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs.config import (  # noqa: E402
    CHECKPOINT_DIR,
    MODEL_NAME,
    NUM_CLASSES,
    SEED,
    checkpoint_dir,
    ensure_version_dirs,
)
from src.dataset import SpeechCommandsDataset  # noqa: E402
from src.models import CommandSense  # noqa: E402
from src.transforms import AudioToThreeChannelMel  # noqa: E402
from train import set_seed  # noqa: E402  (shared seeding helper, as in evaluate.py)

# torch.onnx prints progress messages containing non-ASCII glyphs. When stdout is
# a pipe, Python falls back on the legacy console code page (cp1252 on Windows)
# and those prints would raise UnicodeEncodeError in the middle of the export, so
# the stream is switched to UTF-8 first.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic stream types
            pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_VERSION = "v3.0"        # frozen float32 release that gets quantized
TARGET_VERSION = "v3.1"        # INT8-only release
OPSET_VERSION = 17             # same opset as export_onnx.py
INPUT_NAME = "input"
OUTPUT_NAME = "output"
CALIBRATION_SPLIT = "validation"   # T and tau are fitted here too; see evaluate.py
# 500 clips is the empirical sweet spot: the activation ranges stop moving well
# before it, while the whole split (~11k clips) takes minutes to decode on CPU.
CALIBRATION_SAMPLES = 500
# Held-out validation clips used for the fidelity/latency numbers of the report.
# They are drawn from the same split as the calibration clips but never overlap.
EVALUATION_SAMPLES = 100
BENCHMARK_ROUNDS = 5
FP32_SUFFIX = "_fp32"
INT8_SUFFIX = "_int8"
REPORT_NAME = "quantization_report.txt"
# Every session below runs on the CPU provider, so the measured latency is
# comparable across machines and independent of whichever other EPs are installed.
CPU_PROVIDERS = ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Model / graph plumbing
# ---------------------------------------------------------------------------
def resolve_checkpoint(version=SOURCE_VERSION):
    """Return the weights to quantize, canonical layout first (see export_onnx.py)."""
    candidates = [
        checkpoint_dir(version) / f"{MODEL_NAME}_{version}.pth",
        CHECKPOINT_DIR / f"{MODEL_NAME}_{version}.pth",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return str(candidate)
    return None


def load_fp32_model(checkpoint_path):
    """Rebuild ``CommandSense`` and load the frozen v3.0 weights."""
    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def export_fp32_graph(model, onnx_path):
    """Export the quantizer's float32 reference graph (opset 17, dynamic batch).

    ``dynamo=False`` selects the legacy TorchScript exporter, the one
    ``export_onnx.py`` uses, and ``external_data=False`` keeps every weight inside
    the single ``.onnx`` file the quantizer reads. The batch axis stays dynamic so
    the INT8 graph inherits it and can serve one or many clips per call.
    """
    torch.onnx.export(
        model,
        torch.randn(1, 3, 64, 63),
        onnx_path,
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
    print(f"[+] float32 reference graph exported -> {onnx_path}")


def count_parameters(onnx_path):
    """Number of weight elements carried by a graph's initializers."""
    model = onnx.load(onnx_path, load_external_data=False)
    return int(
        sum(int(np.prod(init.dims)) if init.dims else 1 for init in model.graph.initializer)
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class ClipCalibrationReader(CalibrationDataReader):
    """Streams calibration clips through the v3.0 front-end, one clip at a time.

    ``get_next`` runs the real ``AudioToThreeChannelMel`` on the int16 waveform of
    a randomly drawn clip, so the ranges ONNX Runtime records are the ones the
    deployed graph will actually see: the raw-audio cache is never bypassed by a
    second, divergent feature path.

    ONNX Runtime drives a reader by calling ``get_next`` until it returns ``None``
    and never calls ``rewind``, so the cursor is initialised here rather than
    lazily. The input range is accumulated on the fly so the report can document
    what the calibration pass observed.
    """

    def __init__(self, dataset, extractor, indices):
        self.dataset = dataset
        self.extractor = extractor
        self.indices = [int(index) for index in indices]
        self.cursor = 0
        self.input_min = float("inf")
        self.input_max = float("-inf")

    def get_next(self):
        if self.cursor >= len(self.indices):
            return None
        audio, _ = self.dataset[self.indices[self.cursor]]
        self.cursor += 1
        with torch.no_grad():
            features = self.extractor(audio.unsqueeze(0)).numpy().astype(np.float32)
        self.input_min = min(self.input_min, float(features.min()))
        self.input_max = max(self.input_max, float(features.max()))
        return {INPUT_NAME: features}

    def rewind(self):
        self.cursor = 0


def features_for(dataset, extractor, indices):
    """Pre-compute the ``[1, 3, 64, 63]`` features of ``indices`` as a list."""
    features = []
    with torch.no_grad():
        for index in indices:
            audio, _ = dataset[int(index)]
            features.append(extractor(audio.unsqueeze(0)).numpy().astype(np.float32))
    return features


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
def quantize_graph(fp32_path, int8_path, reader):
    """Post-training static quantization to QDQ int8 (weights and activations)."""
    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f"[+] INT8 graph written -> {int8_path}")


# ---------------------------------------------------------------------------
# Fidelity / latency measurement
# ---------------------------------------------------------------------------
def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def run_session(session, features):
    """Stack the predictions of ``features`` into a single ``[n, NUM_CLASSES]``."""
    predictions = [
        session.run(None, {INPUT_NAME: feature})[0] for feature in features
    ]
    return np.concatenate(predictions, axis=0)


def fidelity_metrics(fp32_logits, int8_logits):
    """Deviation between the two graphs, in logit space and in decision space."""
    logit_gap = np.abs(fp32_logits - int8_logits)
    probability_gap = np.abs(softmax(fp32_logits) - softmax(int8_logits))
    agreement = np.mean(
        np.argmax(fp32_logits, axis=1) == np.argmax(int8_logits, axis=1)
    )
    return {
        "max_logit_gap": float(logit_gap.max()),
        "mean_logit_gap": float(logit_gap.mean()),
        "max_probability_gap": float(probability_gap.max()),
        "mean_probability_gap": float(probability_gap.mean()),
        "top1_agreement": float(agreement),
    }


def benchmark(session, features, rounds=BENCHMARK_ROUNDS):
    """Median milliseconds per clip (single clip per call, after a warm-up)."""
    for _ in range(10):
        session.run(None, {INPUT_NAME: features[0]})
    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        for feature in features:
            session.run(None, {INPUT_NAME: feature})
        timings.append((time.perf_counter() - start) / len(features) * 1000.0)
    return float(np.median(timings))


def verify_int8_graph(int8_path):
    """The INT8 graph must be a valid ONNX model with one dynamic batch axis."""
    model = onnx.load(int8_path, load_external_data=False)
    onnx.checker.check_model(model)
    assert len(model.graph.output) == 1, "[!] expected a single INT8 output tensor"
    dimensions = model.graph.output[0].type.tensor_type.shape.dim
    assert dimensions[-1].dim_value == NUM_CLASSES, (
        f"[!] expected {NUM_CLASSES} logits, got {dimensions[-1].dim_value}"
    )
    assert dimensions[0].dim_param, (
        "[!] INT8 graph lost its dynamic batch axis: quantization must stay "
        "batch-size agnostic"
    )
    print(
        f"[+] INT8 graph verified: {NUM_CLASSES} raw logits, dynamic batch "
        f"'{dimensions[0].dim_param}'"
    )
    return model


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(context):
    """Render the plain-text quantization report (mirrors report_v3.0.txt)."""
    metrics = context["fidelity"]
    compression = 100.0 * (1.0 - context["int8_mb"] / context["fp32_mb"])
    ratio = context["fp32_mb"] / context["int8_mb"]
    latency_ratio = context["int8_ms"] / context["fp32_ms"]
    lines = [
        "=" * 60,
        f"     STATIC INT8 QUANTIZATION REPORT ({MODEL_NAME} {TARGET_VERSION})",
        "=" * 60,
        "",
        f"Source checkpoint : {context['checkpoint']}",
        f"Source graph      : {context['fp32_path']} (opset {OPSET_VERSION}, float32)",
        f"INT8 graph        : {context['int8_path']}",
        f"Calibration split : '{CALIBRATION_SPLIT}' - {CALIBRATION_SAMPLES} clips streamed "
        "through the v3.0 front-end",
        "Quantization      : ONNX Runtime post-training static quantization",
        "                    format=QDQ, per_channel=True, activation=QInt8,",
        "                    weight=QInt8, calibrate_method=MinMax",
        f"Input activation  : [{context['input_min']:.4f}, {context['input_max']:.4f}] "
        "(range observed while calibrating)",
        "",
        "--- ARTIFACT SIZE ---",
        f"float32 graph     : {context['fp32_mb']:.3f} MB "
        f"({context['parameters']:,} parameters)",
        f"int8 graph        : {context['int8_mb']:.3f} MB",
        f"Compression       : {compression:.1f}% smaller ({ratio:.2f}x smaller file)",
        "",
        f"--- FIDELITY ({EVALUATION_SAMPLES} validation clips held out from calibration) ---",
        f"Top-1 agreement   : {metrics['top1_agreement'] * 100:.2f}% of clips keep the "
        "float32 label",
        f"Mean |d logit|    : {metrics['mean_logit_gap']:.6f}",
        f"Max  |d logit|    : {metrics['max_logit_gap']:.6f} (informational, see notes)",
        f"Mean |d softmax|  : {metrics['mean_probability_gap']:.6f}",
        f"Max  |d softmax|  : {metrics['max_probability_gap']:.6f}",
        "",
        f"--- LATENCY (median of {BENCHMARK_ROUNDS} rounds, 1 clip/call, "
        "CPUExecutionProvider) ---",
        f"float32 graph     : {context['fp32_ms']:.3f} ms/clip",
        f"int8 graph        : {context['int8_ms']:.3f} ms/clip",
        f"int8 / float32    : {latency_ratio:.2f}x",
        "",
        "--- NOTES ---",
        "* The graph stays raw-logit: the v3.0 temperature T and threshold tau are "
        "still",
        "  applied by src/inference.py, so quantization changes neither the decision",
        "  rule nor the calibration artifact.",
        "* Absolute logit deviation is not a meaningful INT8 tolerance on this "
        "checkpoint.",
        "  Its logits span roughly +/-40, so a single int8 step is ~0.1-0.4 logit "
        "units and",
        "  the observed max |d logit| sits above 0.05 for every configuration tried",
        "  (MinMax/Entropy/Percentile, per_channel on/off, 50 vs 500 calibration "
        "clips,",
        "  float head kept in fp32, and dynamic quantization alike). The tolerance "
        "that",
        "  actually characterises a well-calibrated INT8 copy is the decision-space "
        "one:",
        "  top-1 agreement and mean |d softmax|.",
        "* The latency comparison is host dependent and INT8 is not reliably faster here:",
        "  ONNX Runtime's QDQ path pays a quantize/dequantize pair per activation and",
        "  has no vectorized int8 kernel for these small 3x64x63 activations, so the",
        "  win here is the download budget, not CPU time. The v4.0 target is ONNX",
        "  Runtime Web, whose WASM SIMD/WebNN kernels consume the QDQ graph natively,",
        "  and tests/test_v3_1_quantization.py re-measures the ratio on whatever",
        "  machine it runs on instead of asserting one - the latency block above",
        "  carries whatever this host measured.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    checkpoint_path = resolve_checkpoint(SOURCE_VERSION)
    if checkpoint_path is None:
        expected = checkpoint_dir(SOURCE_VERSION) / f"{MODEL_NAME}_{SOURCE_VERSION}.pth"
        print(f"[!] Error: no {SOURCE_VERSION} checkpoint found at '{expected}'.")
        print("[!] Train and export v3.0 (python train.py) before quantizing it.")
        return

    checkpoints, reports = ensure_version_dirs(TARGET_VERSION)
    fp32_path = str(checkpoints / f"{MODEL_NAME}_{TARGET_VERSION}{FP32_SUFFIX}.onnx")
    int8_path = str(checkpoints / f"{MODEL_NAME}_{TARGET_VERSION}{INT8_SUFFIX}.onnx")
    report_path = reports / REPORT_NAME

    print(
        f"[*] Quantizing {MODEL_NAME} {SOURCE_VERSION} -> {TARGET_VERSION} "
        "(static int8/QDQ)"
    )
    print(f"[*] Source weights <- {checkpoint_path}")

    export_fp32_graph(load_fp32_model(checkpoint_path), fp32_path)

    # Seeded before the split is built: the dataset draws its extra `_silence_` /
    # `_unknown_` clips from the global generator, so without this the calibrated
    # ranges - and therefore the shipped INT8 artifact - would drift between runs.
    set_seed(SEED)
    print(f"[*] Building '{CALIBRATION_SPLIT}' split (raw-int16 cache, no augmentation)...")
    dataset = SpeechCommandsDataset(
        subset=CALIBRATION_SPLIT, cache_in_ram=False, return_audio=True
    )
    extractor = AudioToThreeChannelMel().eval()
    print(f"[+] Split ready: {len(dataset)} clips")

    # One deterministic draw serves both roles: the leading CALIBRATION_SAMPLES
    # clips set the activation ranges, the trailing EVALUATION_SAMPLES clips are
    # held out of calibration and measure the fidelity of the finished graph.
    indices = np.random.default_rng(SEED).choice(
        len(dataset),
        size=CALIBRATION_SAMPLES + EVALUATION_SAMPLES,
        replace=False,
    )
    calibration_indices = indices[:CALIBRATION_SAMPLES]
    evaluation_indices = indices[CALIBRATION_SAMPLES:]

    print(f"[*] Calibrating on {CALIBRATION_SAMPLES} clips...")
    start = time.perf_counter()
    reader = ClipCalibrationReader(dataset, extractor, calibration_indices)
    quantize_graph(fp32_path, int8_path, reader)
    print(f"[+] Calibration + quantization: {time.perf_counter() - start:.1f} s")
    verify_int8_graph(int8_path)

    evaluation_features = features_for(dataset, extractor, evaluation_indices)
    options = ort.SessionOptions()
    fp32_session = ort.InferenceSession(fp32_path, options, providers=CPU_PROVIDERS)
    int8_session = ort.InferenceSession(int8_path, options, providers=CPU_PROVIDERS)

    metrics = fidelity_metrics(
        run_session(fp32_session, evaluation_features),
        run_session(int8_session, evaluation_features),
    )
    context = {
        "checkpoint": checkpoint_path,
        "fp32_path": fp32_path,
        "int8_path": int8_path,
        "fp32_mb": os.path.getsize(fp32_path) / (1024 * 1024),
        "int8_mb": os.path.getsize(int8_path) / (1024 * 1024),
        "parameters": count_parameters(fp32_path),
        "input_min": reader.input_min,
        "input_max": reader.input_max,
        "fidelity": metrics,
        "fp32_ms": benchmark(fp32_session, evaluation_features),
        "int8_ms": benchmark(int8_session, evaluation_features),
    }

    report = build_report(context)
    print("\n" + report)
    report_path.write_text(report, encoding="utf-8")
    print(f"[+] Report written -> {report_path}")


if __name__ == "__main__":
    main()
