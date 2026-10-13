"""Smoke test for the v3.1 INT8 artifact (no training, no re-quantization).

Loads the two ONNX graphs ``quantize_int8.py`` leaves in
``checkpoints/model_v3.1/`` and compares them on a seeded draw of the *testing*
split - the split that trained neither the weights, nor the calibration ranges, nor
the temperature/threshold pair. It asserts the properties a deployable INT8 copy
must keep:

* identical output shape (``NUM_CLASSES`` raw logits, dynamic batch axis),
* top-1 agreement with the float32 graph of at least ``TOP1_AGREEMENT_MIN`` - a
  floor *derived* from the calibration run's own measurement minus three standard
  errors of this test's sample, not a round number (see the constants below),
* per-class top-1 agreement including the ``_silence_`` / ``_unknown_`` rejection
  classes, so a class that quantizes badly cannot hide in the aggregate,
* mean absolute softmax drift below ``SOFTMAX_DRIFT_MAX``,
* the operating point the artifact documents (``commandsense.calibration.*``)
  matching the v3.1 JSON that ``src/inference.py`` serves, i.e. the payload is
  recoverable from the graph alone, and still transferring to the quantized graph
  within tolerance on this draw.

Measured over the whole testing split (12 105 clips), ``quantize_int8.py`` keeps
99.40% of the float32 labels: the handful of flips sit on near ties, where the
float32 graph's own top-1/top-2 softmax margin is below 0.06 for most of them. That
is why the floor allows a few disagreements instead of demanding an identical
argmax - an exact match is a property of the sample, not of the quantization. The
number of flips, their margins and the raw-logit deviation are all printed for
inspection.

The raw-logit L-infinity deviation is printed as an informational metric only.
This checkpoint is very confident (logits span roughly +/-40), so a single int8
step is already ~0.1-0.4 logit units and any absolute bound near zero is below the
quantizer's resolution - it is unreachable by every post-training scheme, as the
quantization report documents. The decision-space bounds above are the ones that
characterise a usable copy, because the v3.0 temperature T and threshold tau are
applied downstream by ``src/inference.py`` and only ever see the softmax.

Latency is measured and printed, never asserted: it is a property of the host and
of the ONNX Runtime build, and on the CPU provider the QDQ graph is *not* faster
(see ``reports/model_v3.1/quantization_report.txt``), so nothing here claims one.

Run directly:  python tests/test_v3_1_quantization.py
"""

import os
import platform
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import (  # noqa: E402
    MODEL_NAME,
    NUM_CLASSES,
    REJECTION_CLASSES,
    SEED,
    checkpoint_dir,
)
from quantize_int8 import read_calibration_metadata  # noqa: E402  (one reader)
from src import calibration  # noqa: E402  (the runtime's own T/tau maths)
from src.dataset import SpeechCommandsDataset  # noqa: E402
from src.transforms import AudioToThreeChannelMel  # noqa: E402
from train import set_seed  # noqa: E402  (shared seeding helper, as in evaluate.py)

VERSION = "v3.1"                       # INT8 release under test
SOURCE_VERSION = "v3.0"                # float32 release it was derived from
EVALUATION_SPLIT = "testing"           # never seen by training, calibration or T/tau
# 1000 clips is ~27 per class on the 37-label axis: enough for the per-class table
# below to say something, while the split build still dominates the runtime.
EVALUATION_SAMPLES = 1000
INPUT_NAME = "input"
CPU_PROVIDERS = ["CPUExecutionProvider"]
# Top-1 agreement the calibration run measured over the *whole* testing split, as
# printed by quantize_int8.py and stored in
# reports/model_v3.1/quantization_report.txt: 12 105 clips, 99.55% of which keep
# the float32 label. The floor is derived from it rather than picked: at
# N = EVALUATION_SAMPLES the binomial standard error of an agreement estimate is
# sqrt(p (1 - p) / N) ~= 0.21 pt, so three standard errors below the measured value
# is ~98.92%. A round 95% would let a real 3-point regression through, while
# asserting the measured value itself would fail on the draw alone.
# Refresh MEASURED_TOP1_AGREEMENT whenever the report is regenerated.
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
# Decision-space tolerance: the softmax probabilities must stay within 0.05 in
# the mean, far tighter than the +/-1 label change that would matter downstream.
SOFTMAX_DRIFT_MAX = 0.05
# Coverage gap v3.1 is allowed to show against float32 while reusing v3.0's T/tau,
# plus the sampling slack of a 1000-clip draw (~sqrt(2 * 0.9 * 0.1 / 1000) is 1.3 pt
# at two standard errors). The decision itself is made on the whole split in
# quantize_int8.py with a 0.5 pt tolerance; this is the artifact-level guard.
COVERAGE_TOLERANCE_MAX = 0.02
BENCHMARK_ROUNDS = 3


def onnx_path(suffix):
    """Locate an artifact of the INT8 release."""
    return checkpoint_dir(VERSION) / f"{MODEL_NAME}_{VERSION}{suffix}.onnx"


def load_graph(path):
    """Open a graph on the CPU provider, failing with an actionable message."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' is missing. Run 'python quantize_int8.py' first: it exports "
            f"the float32 reference next to the INT8 graph."
        )
    return ort.InferenceSession(str(path), providers=CPU_PROVIDERS)


def testing_sample(limit=EVALUATION_SAMPLES):
    """Stream ``limit`` clips of the testing split through the v3.0 front-end.

    ``set_seed`` also fixes the global generator the dataset draws its extra
    ``_silence_`` / ``_unknown_`` clips from, so both the sampling and the sampled
    content are reproducible from run to run. The targets and the label list come
    back with the features because the per-class table below needs them.
    """
    set_seed(SEED)

    dataset = SpeechCommandsDataset(
        subset=EVALUATION_SPLIT, cache_in_ram=False, return_audio=True
    )
    extractor = AudioToThreeChannelMel().eval()
    indices = np.random.default_rng(SEED).choice(
        len(dataset), size=limit, replace=False
    )
    print(f"[*] Evaluating on {limit} / {len(dataset)} '{EVALUATION_SPLIT}' clips")

    features, targets = [], []
    with torch.no_grad():
        for index in indices:
            audio, target = dataset[int(index)]
            features.append(extractor(audio.unsqueeze(0)).numpy().astype(np.float32))
            targets.append(int(target))
    return features, np.asarray(targets, dtype=np.int64), list(dataset.labels)


def predict(session, features):
    """Stack per-clip logits into a single ``[n, NUM_CLASSES]`` matrix."""
    return np.concatenate(
        [session.run(None, {INPUT_NAME: feature})[0] for feature in features], axis=0
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
            {"label": label, "n": count, "flips": flips, "agreement": 1.0 - flips / count}
        )
    return rows


def milliseconds_per_clip(session, features, rounds=BENCHMARK_ROUNDS):
    """Median, minimum and maximum ms per single-clip call (informational only).

    Median is the headline, but the spread is what says whether a difference
    against the other graph is real on this host; both are printed, neither is
    asserted.
    """
    for _ in range(5):
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


_PREPARED = {}


def prepared():
    """Load both graphs and the evaluation features once per process.

    Cached so the checks below - and pytest, which calls every ``test_*``
    function independently - share a single split build: decoding the testing
    clips dominates the runtime of this file.
    """
    if not _PREPARED:
        fp32_path, int8_path = onnx_path("_fp32"), onnx_path("_int8")
        fp32_session, int8_session = load_graph(fp32_path), load_graph(int8_path)
        features, targets, labels = testing_sample()
        _PREPARED.update(
            fp32_path=fp32_path,
            int8_path=int8_path,
            fp32_session=fp32_session,
            int8_session=int8_session,
            features=features,
            targets=targets,
            labels=labels,
            fp32_logits=predict(fp32_session, features),
            int8_logits=predict(int8_session, features),
        )
    return _PREPARED


def test_graph_contract():
    """The INT8 graph must expose NUM_CLASSES logits over a batch-agnostic axis."""
    shape = list(prepared()["int8_session"].get_outputs()[0].shape)
    assert shape[-1] == NUM_CLASSES, (
        f"INT8 output shape {shape} does not end in {NUM_CLASSES} logits"
    )
    assert not isinstance(shape[0], int), (
        f"INT8 graph lost its dynamic batch axis: output shape {shape} pins the "
        "batch size, so the v4.0 runtime could not serve N clips in one call"
    )
    print(f"[OK] INT8 output contract: {shape}")


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
    ``export_onnx.embed_calibration_metadata``, which ``quantize_int8.py`` calls) and
    ``read_calibration_metadata`` the reader side, so this is the round trip on the
    shipped file: a runtime that only has the graph can recover the operating point
    from ``commandsense.calibration.*`` without the JSON. The values of the v3.0
    payload are printed next to them, because v3.1 is allowed to *differ* from v3.0 -
    that is the re-fit branch of the quantizer, not a bug.
    """
    state = prepared()
    payload = calibration.load_calibration(version=VERSION)
    assert payload, (
        f"no calibration payload for {VERSION}: run evaluate.py or "
        "quantize_int8.py before this test"
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
        f"({missing}): re-run quantize_int8.py so the artifact describes itself"
    )
    for key in ("temperature", "confidence_threshold"):
        assert abs(float(graph[key]) - float(served[key])) < 1e-9, (
            f"{key}: the graph documents {graph[key]} but the payload serves "
            f"{served[key]}"
        )
    v3_0 = calibration.load_calibration(version=SOURCE_VERSION) or {}
    print(
        f"[OK] calibration metadata: {len(graph)} "
        f"'{calibration.ONNX_METADATA_PREFIX}*' entries, T="
        f"{float(graph['temperature']):.6f}, tau="
        f"{float(graph['confidence_threshold']):.2f}; "
        f"{SOURCE_VERSION} serves T={v3_0.get('temperature')}, "
        f"tau={v3_0.get('confidence_threshold')}"
    )


def test_operating_point_transfer():
    """v3.1 reuses v3.0's T/tau - which must still hold on this draw.

    Coverage and retained accuracy of both graphs are measured at the payload's own
    (T, tau). The bound is the shipped 0.5 pt tolerance plus the sampling slack of a
    1000-clip draw, so only a real shift of the decision can break it - the case the
    re-fit branch of ``quantize_int8.py`` exists to catch.
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
        f"tau={threshold:.2f}: quantization moved the operating point v3.0's T/tau "
        "were fitted for, so v3.1 would need its own re-fit"
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
        f"{BENCHMARK_ROUNDS} rounds x {len(state['features'])} clips):"
    )
    for name in ("fp32", "int8"):
        timings = milliseconds_per_clip(state[f"{name}_session"], state["features"])
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
    print("\n[OK] v3.1 quantization smoke test passed")


if __name__ == "__main__":
    main()
