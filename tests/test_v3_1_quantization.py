"""Smoke test for the shipped INT8 artifact of ``MODEL_VERSION`` (no training, no re-export).

Loads the pair ``export_onnx.py`` leaves in ``checkpoints/model_<version>/`` - the
float32 fidelity reference ``<MODEL_NAME>_<version>_fp32.onnx`` and the deployment
artifact ``<MODEL_NAME>_<version>.onnx`` - and compares them on a seeded draw of the
*testing* split, the split that trained neither the weights, nor the calibration
ranges, nor the temperature/threshold pair. Both graphs take raw 1.0 s waveforms
(the mel/delta front-end is baked into them), so the clips are fed on the same
contract ``src/inference.py`` and the browser client use. It asserts what a
deployable INT8 copy must keep: the input contract (``raw_audio``, rank 2, one
clip per row) and the output shape (``NUM_CLASSES`` raw logits on a dynamic batch
axis), top-1 agreement with the float32 graph of at least ``TOP1_AGREEMENT_MIN``,
per-class agreement including the ``_silence_`` / ``_unknown_`` rejection classes
(so a class that quantizes badly cannot hide in the aggregate), mean absolute
softmax drift below ``SOFTMAX_DRIFT_MAX``, and the operating point the artifact
documents (``commandsense.calibration.*``) matching the version's JSON that
``src/inference.py`` serves and still transferring to the quantized graph within
tolerance on this draw.

Over the whole testing split (12,105 clips) 99.55% of the float32 labels survive,
and the few flips sit on near ties whose float32 top-1/top-2 softmax margin is
mostly below 0.06 - which is why the floor allows a few disagreements instead of
demanding an identical argmax: an exact match is a property of the sample, not of
the quantization. Flips, their margins and the raw-logit deviation are printed for
inspection.

The raw-logit L-infinity deviation is informational only: this checkpoint is very
confident (logits span roughly +/-40), so one int8 step is already ~0.1-0.4 logit
units and any tight absolute bound sits below the quantizer's resolution. The
decision-space bounds above are the ones that characterise a usable copy, because
``src/inference.py`` applies T and tau downstream and only ever sees the softmax.

Latency is measured and printed, never asserted: it is a property of the host and of
the ONNX Runtime build, and on the CPU provider the QDQ graph is *not* faster (see
``reports/model_v4.0/quantization_report.txt``).

Run directly:  python tests/test_v3_1_quantization.py
"""

import os
import platform
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import (  # noqa: E402
    MODEL_VERSION,
    NUM_CLASSES,
    REJECTION_CLASSES,
    SEED,
    TARGET_SAMPLES,
)
from export_onnx import (  # noqa: E402  (the exporter's own artifact helpers)
    INPUT_NAME,
    artifact_paths,
    read_calibration_metadata,
    waveform_batch,
)
from src import calibration  # noqa: E402  (the runtime's own T/tau maths)
from src.dataset import SpeechCommandsDataset  # noqa: E402
from train import set_seed  # noqa: E402  (shared seeding helper, as in evaluate.py)

VERSION = MODEL_VERSION                # release under test: its INT8 graph + float32
SOURCE_VERSION = MODEL_VERSION         # same self-contained release: its JSON has T/tau
EVALUATION_SPLIT = "testing"           # never seen by training, calibration or T/tau
# 1,000 clips is ~27 per class on the 37-label axis: enough for the per-class table
# below to say something, while the split build still dominates the runtime.
EVALUATION_SAMPLES = 1000
CPU_PROVIDERS = ["CPUExecutionProvider"]
# Top-1 agreement measured for the shipped line of artifacts over the *whole* testing
# split (see reports/model_v4.0/quantization_report.txt). The floor is derived from it
# rather than picked: at N = EVALUATION_SAMPLES the binomial standard error of an
# agreement estimate is sqrt(p (1 - p) / N) ~= 0.21 pt, so three standard errors below
# the measured value is ~98.92% - a round 95% would let a real 3-point regression
# through, while asserting the measured value itself would fail on the draw alone.
# Refresh MEASURED_TOP1_AGREEMENT whenever the artifact is rebuilt and re-measured.
MEASURED_TOP1_AGREEMENT = 0.9955
TOP1_AGREEMENT_SIGMA = float(
    np.sqrt(
        MEASURED_TOP1_AGREEMENT
        * (1.0 - MEASURED_TOP1_AGREEMENT)
        / EVALUATION_SAMPLES
    )
)
TOP1_AGREEMENT_MIN = MEASURED_TOP1_AGREEMENT - 3.0 * TOP1_AGREEMENT_SIGMA
# A whole command may not collapse: with ~27 clips per class the derived 3-sigma
# bound is not meaningful class by class, so this is a collapse guard (a class may
# lose at most one clip in ten) and the printed table carries the real detail.
PER_CLASS_AGREEMENT_MIN = 0.90
# Decision-space tolerance: the mean softmax drift must stay far tighter than the
# +/-1 label change that would matter downstream.
SOFTMAX_DRIFT_MAX = 0.05
# Coverage gap the INT8 graph may show against float32 while reusing the shipped
# T/tau. A 1,000-clip draw carries ~0.95 pt of sampling error on a coverage estimate
# (~1.3 pt for an unpaired gap) and this draw is paired (both graphs score the same
# clips), so the guard is generous; the shipped decision is made on the whole split
# inside the exporter's calibration pass, with a 0.5 pt tolerance.
COVERAGE_TOLERANCE_MAX = 0.02
BENCHMARK_ROUNDS = 3


def load_graph(path):
    """Open a graph on the CPU provider, failing with an actionable message."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' is missing. Run 'python export_onnx.py' first: it exports "
            f"the float32 reference next to the INT8 graph."
        )
    return ort.InferenceSession(str(path), providers=CPU_PROVIDERS)


def testing_sample(limit=EVALUATION_SAMPLES):
    """Stream ``limit`` clips of the testing split as deployment-ready waveforms.

    ``set_seed`` also fixes the global generator the dataset draws its extra
    ``_silence_`` / ``_unknown_`` clips from, so both the sampling and the sampled
    content are reproducible from run to run. Clips go through ``waveform_batch`` -
    the exporter's own conversion, so the payload a graph sees here is byte-for-byte
    what the calibration pass and the browser client feed it. The targets and the
    label list come back with the clips because the per-class table below needs them.
    """
    set_seed(SEED)

    dataset = SpeechCommandsDataset(
        subset=EVALUATION_SPLIT, cache_in_ram=False, return_audio=True
    )
    indices = np.random.default_rng(SEED).choice(
        len(dataset), size=limit, replace=False
    )
    print(f"[*] Evaluating on {limit} / {len(dataset)} '{EVALUATION_SPLIT}' clips")

    clips, targets = [], []
    for index in indices:
        audio, target = dataset[int(index)]
        clips.append(waveform_batch(audio))
        targets.append(int(target))
    return clips, np.asarray(targets, dtype=np.int64), list(dataset.labels)


def predict(session, clips):
    """Stack per-clip logits into a single ``[n, NUM_CLASSES]`` matrix."""
    return np.concatenate(
        [session.run(None, {INPUT_NAME: clip})[0] for clip in clips], axis=0
    )


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def softmax_margins(probabilities):
    """Top-1 minus top-2 probability per clip: how decided each decision was."""
    ordered = np.sort(probabilities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def per_class_agreement(state):
    """Top-1 agreement of every class present in the draw, flips included.

    The rejection classes are reported next to the 35 commands instead of being
    folded into them: '_silence_' should be *rejected* rather than classified, so
    its failure mode is a different one and averaging it away would hide it.
    """
    fp32_top1 = np.argmax(state["fp32_logits"], axis=1)
    int8_top1 = np.argmax(state["int8_logits"], axis=1)
    rows = []
    for index, label in enumerate(state["labels"]):
        mask = state["targets"] == index
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


def milliseconds_per_clip(session, clips, rounds=BENCHMARK_ROUNDS):
    """Median, minimum and maximum ms per single-clip call (informational only).

    Median is the headline, but the spread is what says whether a difference
    against the other graph is real on this host; both are printed, neither is
    asserted.
    """
    for _ in range(5):
        session.run(None, {INPUT_NAME: clips[0]})
    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        for clip in clips:
            session.run(None, {INPUT_NAME: clip})
        timings.append((time.perf_counter() - start) / len(clips) * 1000.0)
    return {
        "median_ms": float(np.median(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "rounds": int(rounds),
        "samples": int(len(clips)),
    }


_PREPARED = {}


def prepared():
    """Load both graphs and the evaluation clips once per process.

    Cached so the checks below - and pytest, which calls every ``test_*``
    function independently - share a single split build: decoding the testing
    clips dominates the runtime of this file.
    """
    if not _PREPARED:
        fp32_path, int8_path = artifact_paths(VERSION)
        fp32_session, int8_session = load_graph(fp32_path), load_graph(int8_path)
        clips, targets, labels = testing_sample()
        _PREPARED.update(
            fp32_path=fp32_path,
            int8_path=int8_path,
            fp32_session=fp32_session,
            int8_session=int8_session,
            clips=clips,
            targets=targets,
            labels=labels,
            fp32_logits=predict(fp32_session, clips),
            int8_logits=predict(int8_session, clips),
        )
    return _PREPARED


def test_graph_contract():
    """Both graphs must publish the waveform-in / logits-out contract of the runtime.

    The input side matters as much as the output side: ``src/inference.py`` and the
    browser client decide what to feed a graph from its input *rank*, so a shipped
    artifact that silently went back to rank-4 features (or lost the
    ``raw_audio`` name, or a fixed 16,000-sample axis) would break every consumer
    that already expects the baked front-end.
    """
    state = prepared()
    for name in ("fp32", "int8"):
        inputs = state[f"{name}_session"].get_inputs()
        assert len(inputs) == 1, (
            f"{name} graph exposes {len(inputs)} inputs "
            f"({[value.name for value in inputs]}), but the runtime feeds exactly one"
        )
        graph_input = inputs[0]
        assert graph_input.name == INPUT_NAME, (
            f"{name} input is '{graph_input.name}', but the runtime and the browser "
            f"client bind '{INPUT_NAME}'"
        )
        shape = list(graph_input.shape)
        assert len(shape) == 2, (
            f"{name} input rank is {len(shape)} (shape {shape}), so consumers would "
            "treat it as feature tensors instead of raw waveforms: the mel/delta "
            "front-end is no longer baked into this graph"
        )
        assert graph_input.type == "tensor(float)", (
            f"{name} input type is {graph_input.type}, but clients feed float32 PCM"
        )
        assert not isinstance(shape[0], int), (
            f"{name} input shape {shape} pins the batch size, so the runtime could "
            "not serve N clips in one call"
        )
        assert shape[1] == TARGET_SAMPLES, (
            f"{name} input shape {shape} does not take one {TARGET_SAMPLES}-sample "
            "1.0 s clip per row"
        )

    shape = list(state["int8_session"].get_outputs()[0].shape)
    assert shape[-1] == NUM_CLASSES, (
        f"INT8 output shape {shape} does not end in {NUM_CLASSES} logits"
    )
    assert not isinstance(shape[0], int), (
        f"INT8 graph lost its dynamic batch axis: output shape {shape} pins the "
        "batch size, so the runtime could not serve N clips in one call"
    )
    print(
        f"[OK] graph contract: input '{INPUT_NAME}' [B, {TARGET_SAMPLES}] float32 "
        f"waveforms, INT8 output {shape}"
    )


def test_output_shape():
    """Both graphs must answer one raw-logit vector per clip."""
    state = prepared()
    fp32_logits, int8_logits = state["fp32_logits"], state["int8_logits"]
    expected = (EVALUATION_SAMPLES, NUM_CLASSES)
    assert fp32_logits.shape == expected, (
        f"float32 logits {fp32_logits.shape} != {expected}"
    )
    assert int8_logits.shape == expected, (
        f"int8 logits {int8_logits.shape} != {expected}"
    )
    print(
        f"[OK] identical output shape: float32 {fp32_logits.shape} == "
        f"int8 {int8_logits.shape}"
    )


def test_top1_agreement():
    """Quantization must keep the float32 label on all but a near-tie handful."""
    state = prepared()
    fp32_labels = np.argmax(state["fp32_logits"], axis=1)
    int8_labels = np.argmax(state["int8_logits"], axis=1)
    matched = int(np.sum(fp32_labels == int8_labels))
    agreement = matched / len(fp32_labels)
    assert agreement >= TOP1_AGREEMENT_MIN, (
        f"top-1 agreement {agreement * 100:.2f}% < {TOP1_AGREEMENT_MIN * 100:.2f}% "
        f"({matched}/{len(fp32_labels)} clips keep the float32 label): INT8 changed "
        "the decision by more than the calibration measurement accounts for"
    )
    print(
        f"[OK] top-1 agreement: {agreement * 100:.2f}% "
        f"({matched}/{len(fp32_labels)} clips keep the float32 label) >= "
        f"{TOP1_AGREEMENT_MIN * 100:.2f}% (measured "
        f"{MEASURED_TOP1_AGREEMENT * 100:.2f}% - 3 x {TOP1_AGREEMENT_SIGMA * 100:.3f} "
        f"pt sigma at N={EVALUATION_SAMPLES})"
    )


def test_per_class_agreement():
    """No command may collapse, and both rejection classes must be in the table."""
    state = prepared()
    rows = per_class_agreement(state)
    worst = min(rows, key=lambda row: row["agreement"])
    assert worst["agreement"] >= PER_CLASS_AGREEMENT_MIN, (
        f"class '{worst['label']}' keeps only {worst['agreement'] * 100:.2f}% of its "
        f"{worst['n']} clips ({worst['flips']} flips), below "
        f"{PER_CLASS_AGREEMENT_MIN * 100:.0f}%: INT8 moved a whole class, not a "
        "near tie"
    )
    print(
        f"[OK] per-class agreement: worst '{worst['label']}' "
        f"{worst['agreement'] * 100:.2f}% ({worst['n']} clips, {worst['flips']} flips) "
        f">= {PER_CLASS_AGREEMENT_MIN * 100:.0f}%, over {len(rows)} classes"
    )


def test_calibration_metadata():
    """The artifact must document the very T/tau the runtime serves.

    ``calibration.metadata_to_onnx_props`` is the writer side (used by
    ``export_onnx.embed_calibration_metadata``, which ``export_onnx.main`` calls) and
    ``read_calibration_metadata`` the reader side, so this is the round trip on the
    shipped file: a runtime that only has the graph can recover the operating point
    from ``commandsense.calibration.*`` without the JSON. The payload the runtime loads
    from the version's JSON is printed next to the graph's own copy, so both sides of
    the round trip can be read off one line.
    """
    state = prepared()
    payload = calibration.load_calibration(version=VERSION)
    assert payload, (
        f"no calibration payload for {VERSION}: run evaluate.py or "
        "python export_onnx.py before this test"
    )
    expected = calibration.metadata_to_onnx_props(payload)
    graph = read_calibration_metadata(state["int8_path"])
    # The writer emits the namespaced ONNX keys while the reader returns the payload
    # shape again (prefix stripped, values JSON-decoded), so the two are compared on
    # the stripped names: that is the mapping a runtime would actually apply.
    prefix = calibration.ONNX_METADATA_PREFIX
    served = {key[len(prefix):]: value for key, value in expected.items()}
    missing = sorted(set(served) - set(graph))
    assert not missing, (
        f"the INT8 graph is missing {len(missing)} calibration metadata entries "
        f"({missing}): re-run python export_onnx.py so the artifact describes itself"
    )
    for key in ("temperature", "confidence_threshold"):
        assert abs(float(graph[key]) - float(served[key])) < 1e-9, (
            f"{key}: the graph documents {graph[key]} but the payload serves "
            f"{served[key]}"
        )
    served_json = calibration.load_calibration(version=SOURCE_VERSION) or {}
    print(
        f"[OK] calibration metadata: {len(graph)} "
        f"'{calibration.ONNX_METADATA_PREFIX}*' entries, T="
        f"{float(graph['temperature']):.6f}, tau="
        f"{float(graph['confidence_threshold']):.2f}; "
        f"the {SOURCE_VERSION} JSON serves T={served_json.get('temperature')}, "
        f"tau={served_json.get('confidence_threshold')}"
    )


def test_operating_point_transfer():
    """The INT8 graph reuses the shipped T/tau - which must still hold on this draw.

    Coverage and retained accuracy of both graphs are measured at the payload's own
    (T, tau). The bound is the shipped 0.5 pt tolerance plus the sampling slack of a
    1000-clip draw, so only a real shift of the decision can break it - the case the
    exporter's calibration pass exists to catch (it measures the shipped T/tau on the
    whole split and fails when the INT8 graph has moved the operating point).
    """
    state = prepared()
    payload = calibration.load_calibration(version=VERSION)
    assert payload, f"no calibration payload for {VERSION}"
    temperature = float(payload["temperature"])
    threshold = float(payload["confidence_threshold"])
    measured = {
        name: calibration.threshold_metrics(
            calibration.logits_to_probabilities(state[f"{name}_logits"], temperature),
            state["targets"],
            threshold,
        )
        for name in ("fp32", "int8")
    }
    coverage_gap = abs(
        measured["int8"]["coverage"] - measured["fp32"]["coverage"]
    )
    accuracy_gap = abs(
        measured["int8"]["retained_accuracy"] - measured["fp32"]["retained_accuracy"]
    )
    assert coverage_gap <= COVERAGE_TOLERANCE_MAX, (
        f"coverage gap {coverage_gap * 100:.2f} pt > "
        f"{COVERAGE_TOLERANCE_MAX * 100:.1f} pt at T={temperature:.6f}, "
        f"tau={threshold:.2f}: quantization moved the operating point the shipped T/tau "
        "was fitted for, so the INT8 graph would need its own re-fit"
    )
    assert accuracy_gap <= COVERAGE_TOLERANCE_MAX, (
        f"retained-accuracy gap {accuracy_gap * 100:.2f} pt > "
        f"{COVERAGE_TOLERANCE_MAX * 100:.1f} pt at T={temperature:.6f}, "
        f"tau={threshold:.2f}: the clips INT8 keeps answering got worse"
    )
    print(
        f"[OK] operating point at T={temperature:.6f}, tau={threshold:.2f}: coverage "
        f"{measured['fp32']['coverage'] * 100:.2f}% -> "
        f"{measured['int8']['coverage'] * 100:.2f}% (gap {coverage_gap * 100:.2f} pt), "
        f"retained accuracy "
        f"{measured['fp32']['retained_accuracy'] * 100:.2f}% -> "
        f"{measured['int8']['retained_accuracy'] * 100:.2f}% (gap "
        f"{accuracy_gap * 100:.2f} pt), FRR "
        f"{measured['int8']['frr'] * 100:.2f}%, FAR {measured['int8']['far'] * 100:.2f}%"
    )


def test_softmax_drift():
    """The probabilities fed to the temperature/threshold stage must stay close."""
    state = prepared()
    drift = np.abs(softmax(state["fp32_logits"]) - softmax(state["int8_logits"]))
    assert drift.mean() < SOFTMAX_DRIFT_MAX, (
        f"mean |d softmax| {drift.mean():.6f} >= {SOFTMAX_DRIFT_MAX}: the INT8 "
        "confidence distribution drifted too far for tau to stay valid"
    )
    print(
        f"[OK] softmax drift: mean {drift.mean():.6f} < {SOFTMAX_DRIFT_MAX}, "
        f"max {drift.max():.6f}"
    )


def report_metrics():
    """Print what the report tracks but the assertions deliberately do not bound."""
    state = prepared()
    gap = np.abs(state["fp32_logits"] - state["int8_logits"])
    print(
        f"[i] raw logit deviation (informational): max {gap.max():.4f}, "
        f"mean {gap.mean():.4f} - this checkpoint spans roughly +/-40 logits, so a "
        "single int8 step is ~0.1-0.4 units; see reports/model_v3.1/"
    )
    # Per-class detail. The rejection classes are listed next to the 35 commands
    # (each carries ~5% of the split and fails differently), and the float32 margin
    # of the flipping clips is printed too: a flip on a clip whose own top-1/top-2
    # margin is negligible is noise, not a regression, and the temperature/threshold
    # stage rejects those clips anyway.
    rows = per_class_agreement(state)
    commands = [row for row in rows if row["label"] not in REJECTION_CLASSES]
    rejections = [row for row in rows if row["label"] in REJECTION_CLASSES]
    total = sum(row["n"] for row in commands)
    flips = sum(row["flips"] for row in commands)
    print(
        f"[i] per-class top-1 agreement over {len(rows)} classes - the 35 commands "
        f"combined: {(1.0 - flips / total) * 100:.2f}% ({total} clips, {flips} flips):"
    )
    for row in commands + rejections:
        print(
            f"      {row['label']:>10s}  {row['agreement'] * 100:7.2f}%  "
            f"n={row['n']:<4d} flips={row['flips']}"
        )
    flipped = np.argmax(state["fp32_logits"], axis=1) != np.argmax(
        state["int8_logits"], axis=1
    )
    if flipped.any():
        margin = softmax_margins(softmax(state["fp32_logits"]))
        print(
            f"[i] flipped labels: {int(flipped.sum())}/{len(flipped)}, float32 top-2 "
            f"softmax margin there {np.round(margin[flipped], 4).tolist()}"
        )
    else:
        print("[i] flipped labels: none")
    # Latency is measured, printed and never asserted: it is a property of this host
    # and of the ONNX Runtime build, and the QDQ graph is not the faster one here.
    print(
        "[i] latency on this host (1 clip/call, CPUExecutionProvider, "
        f"{BENCHMARK_ROUNDS} rounds x {len(state['clips'])} clips):"
    )
    for name in ("fp32", "int8"):
        timings = milliseconds_per_clip(state[f"{name}_session"], state["clips"])
        print(
            f"      {name:>5s}  median {timings['median_ms']:.3f} ms/clip  "
            f"(min {timings['min_ms']:.3f}, max {timings['max_ms']:.3f})"
        )
    print(
        f"      host: {platform.processor() or platform.machine()}, "
        f"{os.cpu_count()} logical cores, {platform.system()} {platform.release()}, "
        f"python {platform.python_version()}, onnxruntime {ort.__version__}"
    )
    fp32_mb = os.path.getsize(state["fp32_path"]) / (1024 * 1024)
    int8_mb = os.path.getsize(state["int8_path"]) / (1024 * 1024)
    print(
        f"[i] artifact size: float32 {fp32_mb:.3f} MB -> int8 {int8_mb:.3f} MB "
        f"({100.0 * (1.0 - int8_mb / fp32_mb):.1f}% smaller) - the shipped win"
    )


def main():
    prepared()
    test_graph_contract()
    test_output_shape()
    test_top1_agreement()
    test_per_class_agreement()
    test_softmax_drift()
    test_calibration_metadata()
    test_operating_point_transfer()
    report_metrics()
    print(f"\n[OK] {VERSION} quantization smoke test passed")


if __name__ == "__main__":
    main()
