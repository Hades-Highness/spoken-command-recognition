"""Static INT8 quantization of the v3.0 CommandSense graph (v3.1).

v3.0 ships a float32 ONNX graph; v3.1 keeps its weights frozen and produces the
INT8 sibling that the browser / edge runtime (v4.0, ONNX Runtime Web) will serve:

1. re-export the v3.0 checkpoint to ONNX (opset 17, dynamic batch axis): this
   float32 graph is the reference the quantizer needs and the baseline every
   fidelity, decision and latency number below is measured against,
2. calibrate the activation ranges on ``CALIBRATION_SAMPLES`` clips of the
   *validation* split, streamed one clip at a time through the v3.0 front-end so
   the same ``src.transforms.AudioToThreeChannelMel`` math trains and deploys,
3. quantize statically with ONNX Runtime post-training quantization in the ``QDQ``
   format (per-channel int8 weights, int8 activations, MinMax ranges) - the format
   ONNX Runtime Web executes natively, which keeps the v4.0 WASM/WebNN path open,
4. score *both* graphs over the whole ``testing`` split - which trained neither the
   weights, nor the activation ranges, nor ``T``/``tau`` - and report the fidelity
   (top-1 agreement, per-class flips, softmax drift) and the operating point
   (coverage / retained accuracy / FRR / FAR at v3.0's T and tau),
5. reuse v3.0's T/tau while quantizing moves the coverage by less than
   ``COVERAGE_TOLERANCE``, and re-fit T and tau on the INT8 logits of the
   validation split above it,
6. embed that payload into the graph's ``metadata_props`` and read it back - which
   exercises the reader a single-file runtime would use, not only the writer - then
   write the artifact to ``checkpoints/model_v3.1/`` and a human-readable report to
   ``reports/model_v3.1/quantization_report.txt``.

Split provenance: the activation ranges and ``T``/``tau`` both come from
'validation' ('testing' would leak the measurement split into the shipped
artifact) and the measurement uses 'testing' only, never feeding back into the
artifact. Because the validation split is then used twice, v3.0's operating point
is re-measured on 'testing' before being reused.

The run is reproducible: ``set_seed`` fixes the split draw, including the extra
``_silence_`` / ``_unknown_`` clips, before calibration. The checkpoint is read-only
and the graph stays raw-logit, so ``T`` and ``tau`` from ``src/calibration.py`` are
still applied at runtime by ``src/inference.py``.

Run with:  python quantize_int8.py
"""

import gc
import json
import os
import platform
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
    CALIBRATION_BINS,
    CHECKPOINT_DIR,
    DEFAULT_TEMPERATURE,
    MODEL_NAME,
    NUM_CLASSES,
    REJECTION_CLASSES,
    SEED,
    THRESHOLD_CRITERION,
    THRESHOLD_MIN_COVERAGE,
    checkpoint_dir,
    ensure_version_dirs,
)
from export_onnx import embed_calibration_metadata  # noqa: E402  (same writer as v3.0)
from src import calibration  # noqa: E402  (T/tau maths shared with evaluate.py)
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

SOURCE_VERSION = "v3.0"        # frozen float32 release that gets quantized
TARGET_VERSION = "v3.1"        # INT8-only release
OPSET_VERSION = 17             # same opset as export_onnx.py
INPUT_NAME = "input"
OUTPUT_NAME = "output"
# Split the activation ranges are measured on: 'testing' would leak the evaluation
# split into the shipped artifact, so it is never used; 'validation' is the split
# evaluate.py also fits T/tau on (see the module docstring and the report).
CALIBRATION_SPLIT = "validation"
# 500 clips is the empirical sweet spot: the activation ranges stop moving well
# before it, while the whole split (~11k clips) takes minutes to decode on CPU.
CALIBRATION_SAMPLES = 500
# Split the decision numbers are measured on: it fed neither the weights, nor the
# activation ranges, nor T/tau, and it is scored in full.
EVALUATION_SPLIT = "testing"
EVALUATION_SAMPLES = 0             # 0 -> every clip of EVALUATION_SPLIT
# Reusing v3.0's T/tau for the quantized graph is defensible while quantization
# moves the coverage of a graph by no more than this (0.5 point of the split);
# above it, T and tau are re-fitted on the INT8 logits - see fit_operating_point.
COVERAGE_TOLERANCE = 0.005
# Clips materialised at once while a split is scored: the features of a whole
# split never sit in RAM at the same time.
CLIP_CHUNK = 512
# Features kept for the latency comparison: enough rounds for a stable median
# without turning the benchmark into a second pass over the split.
LATENCY_SAMPLES = 200
BENCHMARK_ROUNDS = 5
FP32_SUFFIX = "_fp32"
INT8_SUFFIX = "_int8"
REPORT_NAME = "quantization_report.txt"
# Every session below runs on the CPU provider, so the measured latency stays
# comparable across machines and independent of whichever other EPs are installed.
CPU_PROVIDERS = ["CPUExecutionProvider"]


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


def clip_batch(dataset, extractor, indices):
    """``(features, targets)`` of ``indices``, one clip at a time.

    Each spectrogram comes out of the same ``AudioToThreeChannelMel`` the training
    and deployment paths use, and the target is returned with it so the decision
    metrics never have to index the split a second time.
    """
    features, targets = [], []
    with torch.no_grad():
        for index in indices:
            audio, target = dataset[int(index)]
            features.append(extractor(audio.unsqueeze(0)).numpy().astype(np.float32))
            targets.append(int(target))
    return features, targets


def score_split(dataset, extractor, sessions, indices, chunk=CLIP_CHUNK):
    """Logits of every session over ``indices``, plus the targets.

    The split is walked in ``chunk``-sized batches whose features are built, scored
    by every session and then dropped, so scoring a whole split keeps only the
    ``[n, NUM_CLASSES]`` logit matrices in memory instead of 23,000 spectrograms.
    """
    targets = []
    parts = {name: [] for name in sessions}
    for start in range(0, len(indices), chunk):
        features, batch_targets = clip_batch(
            dataset, extractor, indices[start:start + chunk]
        )
        targets.extend(batch_targets)
        for name, session in sessions.items():
            parts[name].append(run_session(session, features))

    logits = {name: np.concatenate(chunks, axis=0) for name, chunks in parts.items()}
    return logits, np.asarray(targets, dtype=np.int64)


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


def quality_block(logits, targets, temperature):
    """NLL / ECE / MCE of one graph before (T=1) and after temperature scaling.

    Same arithmetic as ``evaluate.py``'s ``split_summary``: the calibration block
    of two releases only compares if it is measured identically.
    """
    probabilities_before = calibration.logits_to_probabilities(logits, DEFAULT_TEMPERATURE)
    probabilities_after = calibration.logits_to_probabilities(logits, temperature)
    diagram_before = calibration.reliability_diagram(
        probabilities_before, targets, n_bins=CALIBRATION_BINS
    )
    diagram_after = calibration.reliability_diagram(
        probabilities_after, targets, n_bins=CALIBRATION_BINS
    )
    return {
        "n_samples": int(len(targets)),
        "accuracy": calibration.accuracy(probabilities_after, targets),
        "nll_before": calibration.negative_log_likelihood(
            logits, targets, DEFAULT_TEMPERATURE
        ),
        "nll_after": calibration.negative_log_likelihood(logits, targets, temperature),
        "ece_before": diagram_before["ece"],
        "ece_after": diagram_after["ece"],
        "mce_before": diagram_before["mce"],
        "mce_after": diagram_after["mce"],
    }


def format_quality_block(title, summary, temperature):
    """Render one quality block in the shape ``report_v3.0.txt`` uses."""
    label = f"after (T={format(temperature, '.4f')})"
    return "\n".join(
        [
            f"--- {title} ---",
            f"Samples            : {summary['n_samples']}",
            "Top-1 accuracy (unchanged by temperature scaling): "
            f"{summary['accuracy'] * 100:.4f}%",
            "",
            f"{'':14s}{'before (T=1)':>16s}{label:>22s}",
            f"{'NLL':14s}{summary['nll_before']:>16.6f}{summary['nll_after']:>22.6f}",
            f"{'ECE':14s}{summary['ece_before']:>16.6f}{summary['ece_after']:>22.6f}",
            f"{'MCE':14s}{summary['mce_before']:>16.6f}{summary['mce_after']:>22.6f}",
        ]
    )


def operating_point(logits, targets, temperature, threshold):
    """Coverage / retained accuracy / FRR / FAR of one graph at ``(T, tau)``."""
    probabilities = calibration.logits_to_probabilities(logits, temperature)
    return calibration.threshold_metrics(probabilities, targets, threshold)


def fit_operating_point(logits, targets):
    """Re-fit ``(T, tau)`` on one graph's own logits, exactly like evaluate.py.

    Used only when quantization moved the operating point beyond
    ``COVERAGE_TOLERANCE``: T minimises the NLL of *this* graph, tau is swept and
    selected with the project's criterion and minimum coverage, so the re-fitted
    pair stays comparable to the v3.0 one it replaces.
    """
    temperature = calibration.fit_temperature(logits, targets)
    sweep = calibration.sweep_thresholds(
        calibration.logits_to_probabilities(logits, temperature), targets
    )
    best = calibration.select_threshold(
        sweep, criterion=THRESHOLD_CRITERION, min_coverage=THRESHOLD_MIN_COVERAGE
    )
    return float(temperature), float(best["tau"]), best


def per_class_agreement(fp32_logits, int8_logits, targets, labels):
    """Top-1 agreement class by class, with the flips, for every class present."""
    fp32_top1 = np.argmax(fp32_logits, axis=1)
    int8_top1 = np.argmax(int8_logits, axis=1)
    rows = []
    for index, label in enumerate(labels):
        mask = targets == index
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        flips = int(np.count_nonzero(fp32_top1[mask] != int8_top1[mask]))
        rows.append(
            {
                "label": label,
                "n": count,
                "flips": flips,
                "agreement": 1.0 - flips / count,
            }
        )
    return rows


def command_agreement(rows):
    """Aggregate the per-class rows over the 35 real commands only.

    The two rejection classes carry ~550 clips each and fail differently
    ('_silence_' should be *rejected*, not classified), so folding them into one
    agreement number would hide which half of the decision moved.
    """
    commands = [row for row in rows if row["label"] not in REJECTION_CLASSES]
    count = sum(row["n"] for row in commands)
    flips = sum(row["flips"] for row in commands)
    return {
        "n": count,
        "flips": flips,
        "agreement": (1.0 - flips / count) if count else 1.0,
    }


def benchmark(session, features, rounds=BENCHMARK_ROUNDS):
    """Median, minimum and maximum milliseconds per clip (1 clip/call).

    The median is the headline number, but the spread is reported with it: the two
    graphs are compared inside one process on one CPU, so min/max is what says
    whether a difference is real. Nothing asserts these values - they are a
    property of the host, which is why the report carries them and the test
    re-measures them instead of pinning them.
    """
    for _ in range(10):
        session.run(None, {INPUT_NAME: features[0]})
    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        for feature in features:
            session.run(None, {INPUT_NAME: feature})
        timings.append((time.perf_counter() - start) / len(features) * 1000.0)
    return {
        "median_ms": float(np.median(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "rounds": int(rounds),
        "samples": int(len(features)),
    }


def read_calibration_metadata(onnx_path):
    """Read the ``commandsense.calibration.*`` block back out of a graph.

    The report claims the T/tau of the shipped artifact can be recovered from the
    graph alone, so the claim is verified here: the values are JSON-decoded out of a
    namespace no other producer writes to, exactly as a single-file runtime (the
    v4.0 ONNX Runtime Web client) reads them. An empty dict means the graph
    documents nothing.
    """
    model = onnx.load(onnx_path, load_external_data=False)
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
    metadata = read_calibration_metadata(int8_path)
    if metadata:
        print(
            f"[+] Calibration metadata read back from the graph: "
            f"T={metadata.get('temperature')}, tau={metadata.get('confidence_threshold')}"
        )
    else:
        print("[i] The INT8 graph carries no calibration metadata yet")
    return model


def build_report(context):
    """Render the plain-text quantization report (mirrors report_v3.0.txt).

    Every number comes from ``context``, i.e. from this run: the fidelity and
    decision blocks are measured on ``EVALUATION_SPLIT`` and the calibration block
    on ``CALIBRATION_SPLIT``, both over their whole split, so the report doubles as
    the audit trail for the split provenance and for the T/tau that ships with the
    graph.
    """
    metrics = context["fidelity"]
    compression = 100.0 * (1.0 - context["int8_mb"] / context["fp32_mb"])
    ratio = context["fp32_mb"] / context["int8_mb"]
    int8_ms, fp32_ms = context["int8_ms"], context["fp32_ms"]
    latency_ratio = int8_ms["median_ms"] / fp32_ms["median_ms"]
    flips = int(round((1.0 - metrics["top1_agreement"]) * context["fidelity_samples"]))
    flipped = [row for row in context["per_class"] if row["flips"]]
    fp32_dec, int8_dec = context["decision"]["fp32"], context["decision"]["int8"]
    fp32_acc = context["test_quality"]["fp32"]["accuracy"]
    int8_acc = context["test_quality"]["int8"]["accuracy"]
    retained_gap_pt = 100.0 * (
        int8_dec["retained_accuracy"] - fp32_dec["retained_accuracy"]
    )
    commands = context["command"]
    flip_summary = (
        ", ".join(f"{row['label']} ({row['flips']})" for row in flipped) or "none"
    )
    rejection_rows = [
        f"  {row['label']:17s}: {row['agreement'] * 100:.2f}% "
        f"({row['n']} clips, {row['flips']} flips)"
        for row in context["per_class"]
        if row["label"] in REJECTION_CLASSES
    ]
    lines = [
        "=" * 60,
        f"     STATIC INT8 QUANTIZATION REPORT ({MODEL_NAME} {TARGET_VERSION})",
        "=" * 60,
        "",
        f"Source checkpoint : {context['checkpoint']}",
        f"Source graph      : {context['fp32_path']} (opset {OPSET_VERSION}, float32)",
        f"INT8 graph        : {context['int8_path']}",
        "Quantization      : ONNX Runtime post-training static quantization",
        "                    format=QDQ, per_channel=True, activation=QInt8,",
        "                    weight=QInt8, calibrate_method=MinMax",
        f"Input activation  : [{context['input_min']:.4f}, {context['input_max']:.4f}] "
        "(range observed while calibrating)",
        "",
        "--- CALIBRATION SPLIT PROVENANCE ---",
        f"Activation ranges : '{CALIBRATION_SPLIT}', {CALIBRATION_SAMPLES} of "
        f"{context['calibration_total']} clips, rng seed {SEED}",
        "                    ('testing' would leak the measurement split into the",
        "                    shipped artifact, so it is never calibrated on)",
        "T / tau source    : 'validation' too (evaluate.py fits them there), i.e. the",
        "                    one split used twice. That is why the v3.0 operating",
        "                    point is re-measured below instead of copied on faith.",
        f"Measurement split : '{EVALUATION_SPLIT}', {context['fidelity_samples']} clips "
        "(the whole split)",
        "                    - trained no weight, no activation range and no T/tau,",
        "                    and is never written back into the artifact.",
        "",
        "--- ARTIFACT SIZE ---",
        f"float32 graph     : {context['fp32_mb']:.3f} MB "
        f"({context['parameters']:,} parameters)",
        f"int8 graph        : {context['int8_mb']:.3f} MB",
        f"Compression       : {compression:.1f}% smaller ({ratio:.2f}x smaller file)",
        "",
        f"--- OPERATING POINT AT T={context['temperature']:.6f}, "
        f"tau={context['threshold']:.2f} ('{EVALUATION_SPLIT}', v3.0's pair) ---",
        f"{'':22s}{'float32':>11s}{'int8':>11s}{'delta':>12s}",
        f"{'coverage':22s}{fp32_dec['coverage'] * 100:>10.2f}%"
        f"{int8_dec['coverage'] * 100:>10.2f}%"
        f"{100.0 * (int8_dec['coverage'] - fp32_dec['coverage']):>+11.2f} pt",
        f"{'retained accuracy':22s}{fp32_dec['retained_accuracy'] * 100:>10.2f}%"
        f"{int8_dec['retained_accuracy'] * 100:>10.2f}%"
        f"{retained_gap_pt:>+11.2f} pt",
        f"{'top-1 accuracy':22s}{fp32_acc * 100:>10.2f}%{int8_acc * 100:>10.2f}%"
        f"{100.0 * (int8_acc - fp32_acc):>+11.2f} pt",
        f"{'FRR':22s}{fp32_dec['frr'] * 100:>10.2f}%{int8_dec['frr'] * 100:>10.2f}%"
        f"{100.0 * (int8_dec['frr'] - fp32_dec['frr']):>+11.2f} pt",
        f"{'FAR':22s}{fp32_dec['far'] * 100:>10.2f}%{int8_dec['far'] * 100:>10.2f}%"
        f"{100.0 * (int8_dec['far'] - fp32_dec['far']):>+11.2f} pt",
        f"{'answered clips':22s}{fp32_dec['n_accepted']:>11d}{int8_dec['n_accepted']:>11d}"
        f"{int8_dec['n_accepted'] - fp32_dec['n_accepted']:>+12d}",
        *context["verdict"],
        "",
        f"--- FIDELITY (INT8 vs float32, '{EVALUATION_SPLIT}', "
        f"{context['fidelity_samples']} clips) ---",
        f"Top-1 agreement   : {metrics['top1_agreement'] * 100:.2f}% of clips keep the "
        f"float32 label ({flips} flips)",
        f"Mean |d logit|    : {metrics['mean_logit_gap']:.6f}",
        f"Max  |d logit|    : {metrics['max_logit_gap']:.6f} (informational, see notes)",
        f"Mean |d softmax|  : {metrics['mean_probability_gap']:.6f}",
        f"Max  |d softmax|  : {metrics['max_probability_gap']:.6f}",
        "Per-class top-1 agreement (35 commands + 2 rejection classes):",
        f"  35 commands      : {commands['agreement'] * 100:.2f}% "
        f"({commands['n']} clips, {commands['flips']} flips)",
        *rejection_rows,
        f"  classes with flips ({len(flipped)}): {flip_summary}",
        f"  classes with no flips: {len(context['per_class']) - len(flipped)}",
        "",
        f"--- CALIBRATION QUALITY (NLL / ECE / MCE, T=1 vs T="
        f"{context['temperature']:.6f}) ---",
        "",
        format_quality_block(
            f"int8 on '{CALIBRATION_SPLIT}' (calibration split): "
            f"{context['calibration_quality']['int8']['n_samples']} clips",
            context["calibration_quality"]["int8"],
            context["temperature"],
        ),
        "",
        format_quality_block(
            f"float32 on '{CALIBRATION_SPLIT}' (calibration split): "
            f"{context['calibration_quality']['fp32']['n_samples']} clips",
            context["calibration_quality"]["fp32"],
            context["temperature"],
        ),
        "",
        format_quality_block(
            f"int8 on '{EVALUATION_SPLIT}' (measurement split): "
            f"{context['test_quality']['int8']['n_samples']} clips",
            context["test_quality"]["int8"],
            context["temperature"],
        ),
        "",
        format_quality_block(
            f"float32 on '{EVALUATION_SPLIT}' (measurement split): "
            f"{context['test_quality']['fp32']['n_samples']} clips",
            context["test_quality"]["fp32"],
            context["temperature"],
        ),
        "",
        f"--- LATENCY (median of {int8_ms['rounds']} rounds x {int8_ms['samples']} "
        "clips, 1 clip/call, CPUExecutionProvider) ---",
        f"Host              : {context['host']}",
        f"float32 graph     : {fp32_ms['median_ms']:.3f} ms/clip "
        f"(min {fp32_ms['min_ms']:.3f}, max {fp32_ms['max_ms']:.3f})",
        f"int8 graph        : {int8_ms['median_ms']:.3f} ms/clip "
        f"(min {int8_ms['min_ms']:.3f}, max {int8_ms['max_ms']:.3f})",
        f"int8 / float32    : {latency_ratio:.2f}x -> CPU latency is NOT improved by this",
        "                    quantization; the win of v3.1 is size, see notes.",
        "",
        "--- CALIBRATION ARTIFACT ---",
        f"Served payload    : {context['calibration_json']}",
        f"T / tau           : {context['temperature']:.6f} / {context['threshold']:.2f} "
        f"({context['payload_origin']})",
        f"Resolved by       : calibration.resolve_calibration_file('{TARGET_VERSION}'), a",
        "                    per-version lookup, so v3.0 keeps serving its own file.",
        f"Graph metadata    : {context['metadata_entries']} "
        f"'{calibration.ONNX_METADATA_PREFIX}*' entries, read back as "
        f"T={context['metadata_readback'].get('temperature')}, "
        f"tau={context['metadata_readback'].get('confidence_threshold')}",
        "                    (a self-describing copy for a single-file runtime such as",
        "                    the v4.0 ONNX Runtime Web client; src/inference.py resolves",
        "                    T/tau from the JSON above, by version)",
        "",
        "--- NOTES ---",
        "* Size is the win, and the only one claimed: "
        f"{compression:.1f}% smaller file",
        f"  ({context['fp32_mb']:.3f} MB -> {context['int8_mb']:.3f} MB). CPU latency is "
        "not improved -",
        f"  the int8 median is {int8_ms['median_ms']:.3f} ms/clip against "
        f"{fp32_ms['median_ms']:.3f} ms/clip for float32",
        f"  ({latency_ratio:.2f}x) on this host. ONNX Runtime's CPU QDQ path pays a",
        "  quantize/dequantize pair per activation and has no vectorized int8 kernel for",
        "  these small 3x64x63 activations, so v3.1 makes no speed claim anywhere. The",
        "  v4.0 target is ONNX Runtime Web, whose WASM SIMD/WebNN kernels consume the QDQ",
        "  graph natively; tests/test_v3_1_quantization.py re-measures the ratio wherever",
        "  it runs instead of asserting one.",
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
        "  top-1 agreement, the per-class flips above and mean |d softmax| - the bounds",
        "  tests/test_v3_1_quantization.py asserts, with its agreement floor derived from",
        "  this measurement instead of picked as a round number.",
        "* Split provenance: the activation ranges and T/tau both come from 'validation'",
        "  (the one split used twice, named at the top); every number above is measured",
        "  on 'testing', which fed nothing and is never written back. Same seed, same",
        "  ranges, same decision on a re-run.",
        "* The graph stays raw-logit: T and tau are applied by src/inference.py, so",
        "  changing the threshold never needs a re-export, and the values this graph",
        "  documents under 'commandsense.calibration.*' are exactly the ones the JSON",
        "  payload serves.",
    ]
    return "\n".join(lines) + "\n"


def load_split(subset):
    """Build one split on the raw-int16 cache, with augmentation off.

    ``set_seed(SEED)`` is called first every time: the dataset draws its extra
    ``_silence_`` / ``_unknown_`` clips from the *global* generator, so re-seeding
    makes each split's draw a pure function of ``SEED`` instead of a function of
    how many draws the previously built split happened to consume.
    """
    set_seed(SEED)
    dataset = SpeechCommandsDataset(subset=subset, cache_in_ram=False, return_audio=True)
    print(f"[+] '{subset}' ready: {len(dataset)} clips")
    return dataset


def split_indices(size, limit):
    """Every index of a split, or a seeded sample of ``limit`` of them."""
    if not limit or limit >= size:
        return list(range(size))
    chosen = np.random.default_rng(SEED).choice(size, size=limit, replace=False)
    return [int(index) for index in chosen]


def host_description():
    """Name the machine and the runtime the latency numbers were measured on."""
    return (
        f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical cores, "
        f"{platform.system()} {platform.release()}, python {platform.python_version()}, "
        f"onnxruntime {ort.__version__}"
    )


def served_operating_point(version):
    """The ``(T, tau, payload)`` a version currently ships, or ``None``."""
    payload = calibration.load_calibration(version=version)
    temperature, threshold = payload.get("temperature"), payload.get("confidence_threshold")
    if temperature is None or threshold is None:
        return None
    return float(temperature), float(threshold), payload


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

    print(f"[*] Building the '{CALIBRATION_SPLIT}' split (raw-int16 cache, no augmentation)")
    calibration_split = load_split(CALIBRATION_SPLIT)
    extractor = AudioToThreeChannelMel().eval()
    calibration_indices = split_indices(len(calibration_split), CALIBRATION_SAMPLES)

    print(
        f"[*] Calibrating the activation ranges on {len(calibration_indices)} clips of "
        f"'{CALIBRATION_SPLIT}'..."
    )
    start = time.perf_counter()
    reader = ClipCalibrationReader(calibration_split, extractor, calibration_indices)
    quantize_graph(fp32_path, int8_path, reader)
    print(f"[+] Calibration + quantization: {time.perf_counter() - start:.1f} s")
    verify_int8_graph(int8_path)

    options = ort.SessionOptions()
    sessions = {
        "fp32": ort.InferenceSession(fp32_path, options, providers=CPU_PROVIDERS),
        "int8": ort.InferenceSession(int8_path, options, providers=CPU_PROVIDERS),
    }

    # --------------------------------------------- v3.0's shipped T and tau
    served = served_operating_point(SOURCE_VERSION)
    if served is None:
        print(
            f"[!] No calibration payload for {SOURCE_VERSION}, so its decision rule is "
            f"the raw-logit one (T={DEFAULT_TEMPERATURE}, tau=0). Run evaluate.py if "
            "that is not intended."
        )
        source_temperature, source_threshold = DEFAULT_TEMPERATURE, 0.0
    else:
        source_temperature, source_threshold, _ = served
        print(
            f"[*] {SOURCE_VERSION} ships T={source_temperature:.6f}, "
            f"tau={source_threshold:.2f}; re-measuring it on '{EVALUATION_SPLIT}'..."
        )

    # ------------------------- calibration split: quality block + refit input
    print(
        f"[*] Scoring the whole '{CALIBRATION_SPLIT}' split with both graphs "
        f"({len(calibration_split)} clips)..."
    )
    start = time.perf_counter()
    calibration_logits, calibration_targets = score_split(
        calibration_split, extractor, sessions, split_indices(len(calibration_split), 0)
    )
    del calibration_split
    gc.collect()
    print(f"[+] '{CALIBRATION_SPLIT}' scored in {time.perf_counter() - start:.1f} s")

    # ------------------------ measurement split: fed nothing, written to nothing
    print(f"[*] Building the '{EVALUATION_SPLIT}' split...")
    evaluation_split = load_split(EVALUATION_SPLIT)
    evaluation_indices = split_indices(len(evaluation_split), EVALUATION_SAMPLES)
    print(
        f"[*] Scoring the whole '{EVALUATION_SPLIT}' split with both graphs "
        f"({len(evaluation_indices)} clips)..."
    )
    start = time.perf_counter()
    test_logits, test_targets = score_split(
        evaluation_split, extractor, sessions, evaluation_indices
    )
    labels = list(evaluation_split.labels)
    latency_features, _ = clip_batch(
        evaluation_split, extractor, split_indices(len(evaluation_split), LATENCY_SAMPLES)
    )
    del evaluation_split
    gc.collect()
    print(f"[+] '{EVALUATION_SPLIT}' scored in {time.perf_counter() - start:.1f} s")

    # ------------------------------------------------------- the measurements
    decision = {
        name: operating_point(logits, test_targets, source_temperature, source_threshold)
        for name, logits in test_logits.items()
    }
    coverage_gap = abs(decision["int8"]["coverage"] - decision["fp32"]["coverage"])
    print(
        f"[*] Operating point at T={source_temperature:.6f}, tau={source_threshold:.2f} "
        f"on '{EVALUATION_SPLIT}': coverage float32 "
        f"{decision['fp32']['coverage'] * 100:.2f}% vs int8 "
        f"{decision['int8']['coverage'] * 100:.2f}% (|delta| {coverage_gap * 100:.2f} pt)"
    )

    refit_temperature, refit_threshold, refit_best = fit_operating_point(
        calibration_logits["int8"], calibration_targets
    )
    print(
        f"[*] Re-fitting on the int8 logits of '{CALIBRATION_SPLIT}' would give "
        f"T={refit_temperature:.6f}, tau={refit_threshold:.2f} (coverage "
        f"{refit_best['coverage'] * 100:.2f}%, FRR {refit_best['frr'] * 100:.2f}%, "
        f"FAR {refit_best['far'] * 100:.2f}%)"
    )
    if coverage_gap <= COVERAGE_TOLERANCE:
        temperature, threshold = source_temperature, source_threshold
        payload_origin = f"reused from {SOURCE_VERSION}"
        verdict = [
            f"|d coverage| = {coverage_gap * 100:.2f} pt <= "
            f"{COVERAGE_TOLERANCE * 100:.1f} pt, so v3.0's T/tau transfer to the",
            f"quantized graph and {TARGET_VERSION} ships the same operating point; "
            "re-fitting on the int8",
            f"logits would give T={refit_temperature:.6f}, tau={refit_threshold:.2f} - "
            "the same decision within noise.",
        ]
    else:
        temperature, threshold = refit_temperature, refit_threshold
        payload_origin = f"re-fitted on the {TARGET_VERSION} int8 logits"
        verdict = [
            f"|d coverage| = {coverage_gap * 100:.2f} pt > "
            f"{COVERAGE_TOLERANCE * 100:.1f} pt, so quantization moved the operating",
            "point and T/tau were re-fitted on the int8 logits of "
            f"'{CALIBRATION_SPLIT}' with '{THRESHOLD_CRITERION}'",
            f"(coverage {refit_best['coverage'] * 100:.2f}%, FRR "
            f"{refit_best['frr'] * 100:.2f}%, FAR {refit_best['far'] * 100:.2f}%).",
        ]
    print(f"[*] Shipping T={temperature:.6f}, tau={threshold:.2f} ({payload_origin})")

    calibration_quality = {
        name: quality_block(logits, calibration_targets, temperature)
        for name, logits in calibration_logits.items()
    }
    test_quality = {
        name: quality_block(logits, test_targets, temperature)
        for name, logits in test_logits.items()
    }
    per_class = per_class_agreement(
        test_logits["fp32"], test_logits["int8"], test_targets, labels
    )
    command = command_agreement(per_class)
    metrics = fidelity_metrics(test_logits["fp32"], test_logits["int8"])
    flipped_names = ", ".join(str(row["label"]) for row in per_class if row["flips"])
    print(
        f"[*] Fidelity on '{EVALUATION_SPLIT}' ({len(test_targets)} clips): top-1 "
        f"agreement {metrics['top1_agreement'] * 100:.2f}%, 35-command agreement "
        f"{command['agreement'] * 100:.2f}%, mean |d softmax| "
        f"{metrics['mean_probability_gap']:.6f}, classes with flips: "
        f"{flipped_names or 'none'}"
    )

    payload = calibration.build_calibration_metadata(
        temperature=temperature,
        confidence_threshold=threshold,
        version=TARGET_VERSION,
        n_bins=CALIBRATION_BINS,
        ece_before=calibration_quality["int8"]["ece_before"],
        ece_after=calibration_quality["int8"]["ece_after"],
        nll_before=calibration_quality["int8"]["nll_before"],
        nll_after=calibration_quality["int8"]["nll_after"],
        threshold_criterion=THRESHOLD_CRITERION,
        calibration_split=(
            f"{CALIBRATION_SPLIT} ({len(calibration_targets)} samples): activation "
            f"ranges and T/tau; measured on {EVALUATION_SPLIT} ({len(test_targets)} "
            "samples)"
        ),
        calibration_samples=int(len(calibration_targets)),
        source_checkpoint=checkpoint_path,
        notes=(
            f"INT8 sibling of {SOURCE_VERSION}, so ECE/NLL above are the quantized "
            f"graph's on '{CALIBRATION_SPLIT}'. T/tau {payload_origin}; quantization "
            f"moved the coverage of the held-out '{EVALUATION_SPLIT}' split by "
            f"{coverage_gap * 100:.2f} pt ({COVERAGE_TOLERANCE * 100:.1f} pt "
            f"tolerance). Activation ranges from {CALIBRATION_SAMPLES} "
            f"'{CALIBRATION_SPLIT}' clips; '{EVALUATION_SPLIT}' fed nothing."
        ),
    )
    calibration_json = calibration.save_calibration(payload, version=TARGET_VERSION)
    print(
        f"[+] Saved {calibration_json}  <- resolved by src/inference.py for "
        f"{TARGET_VERSION}"
    )
    # embed_calibration_metadata() stamps the version's own JSON onto the graph, so
    # it runs after the payload above; the read-back then proves the graph carries
    # the operating point this report documents.
    embed_calibration_metadata(int8_path, version=TARGET_VERSION)
    metadata_readback = read_calibration_metadata(int8_path)
    print(
        f"[+] Graph metadata: {len(metadata_readback)} entries, T="
        f"{metadata_readback.get('temperature')}, tau="
        f"{metadata_readback.get('confidence_threshold')}"
    )

    context = {
        "checkpoint": checkpoint_path,
        "fp32_path": fp32_path,
        "int8_path": int8_path,
        # Sized after the metadata embedding above, so the report quotes the bytes
        # of the file that actually ships.
        "fp32_mb": os.path.getsize(fp32_path) / (1024 * 1024),
        "int8_mb": os.path.getsize(int8_path) / (1024 * 1024),
        "parameters": count_parameters(fp32_path),
        "input_min": reader.input_min,
        "input_max": reader.input_max,
        "calibration_total": int(len(calibration_targets)),
        "fidelity_samples": int(len(test_targets)),
        "fidelity": metrics,
        "per_class": per_class,
        "command": command,
        "decision": decision,
        "calibration_quality": calibration_quality,
        "test_quality": test_quality,
        "temperature": temperature,
        "threshold": threshold,
        "payload_origin": payload_origin,
        "verdict": verdict,
        "calibration_json": str(calibration_json),
        "metadata_entries": len(metadata_readback),
        "metadata_readback": metadata_readback,
        "fp32_ms": benchmark(sessions["fp32"], latency_features),
        "int8_ms": benchmark(sessions["int8"], latency_features),
        "host": host_description(),
    }

    report = build_report(context)
    print("\n" + report)
    report_path.write_text(report, encoding="utf-8")
    print(f"[+] Report written -> {report_path}")


if __name__ == "__main__":
    main()
