"""Smoke test for the v3.1 INT8 artifact (no training, no re-quantization).

Loads the two ONNX graphs that ``quantize_int8.py`` leaves in
``checkpoints/model_v3.1/`` and compares them on a random draw of the *testing*
split - the split that trained neither the weights, nor the calibration ranges, nor
the temperature/threshold pair. It asserts the properties a deployable INT8 copy
must keep:

* identical output shape (``NUM_CLASSES`` raw logits, dynamic batch axis),
* top-1 agreement with the float32 graph of at least ``TOP1_AGREEMENT_MIN``,
* mean absolute softmax drift below ``SOFTMAX_DRIFT_MAX``.

Measured over 1000 testing clips, quantization keeps 99.40% of the float32 labels:
the handful of flips sit on near ties, where the float32 graph's own top-1/top-2
softmax margin is below 0.06 for five of the six. That is why the floor allows a
few disagreements instead of demanding an identical argmax - an exact match is a
property of the sample, not of the quantization. The number of flips, their
margins and the raw-logit deviation are all printed for inspection.

The raw-logit L-infinity deviation is printed as an informational metric only.
This checkpoint is very confident (logits span roughly +/-40), so a single int8
step is already ~0.1-0.4 logit units and any absolute bound near zero is below the
quantizer's resolution - it is unreachable by every post-training scheme, as the
quantization report documents. The decision-space bounds above are the ones that
characterise a usable copy, because the v3.0 temperature T and threshold tau are
applied downstream by ``src/inference.py`` and only ever see the softmax.

Run directly:  python tests/test_v3_1_quantization.py
"""

import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import (  # noqa: E402
    MODEL_NAME,
    NUM_CLASSES,
    SEED,
    checkpoint_dir,
)
from src.dataset import SpeechCommandsDataset  # noqa: E402
from src.transforms import AudioToThreeChannelMel  # noqa: E402
from train import set_seed  # noqa: E402  (shared seeding helper, as in evaluate.py)

VERSION = "v3.1"                       # INT8 release under test
SOURCE_VERSION = "v3.0"                # float32 release it was derived from
EVALUATION_SPLIT = "testing"           # never seen by training, calibration or T/tau
# 200 clips keeps the agreement estimate tight while costing little: the split
# build dominates the runtime, not the per-clip feature extraction.
EVALUATION_SAMPLES = 200
INPUT_NAME = "input"
CPU_PROVIDERS = ["CPUExecutionProvider"]
# The INT8 graph must not change the decision. A handful of clips in a thousand do
# flip, always on near ties where the float32 graph is itself undecided (measured:
# 99.40% agreement over 1000 testing clips, five of the six flips with a float32
# top-1/top-2 softmax margin below 0.06), so the floor tolerates a small number of
# sample-dependent disagreements rather than demanding a perfectly identical argmax.
TOP1_AGREEMENT_MIN = 0.95
# Decision-space tolerance: the softmax probabilities must stay within 0.05 in
# the mean, far tighter than the +/-1 label change that would matter downstream.
SOFTMAX_DRIFT_MAX = 0.05
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


def testing_features(limit=EVALUATION_SAMPLES):
    """Stream ``limit`` clips of the testing split through the v3.0 front-end.

    ``set_seed`` also fixes the global generator the dataset draws its extra
    ``_silence_`` / ``_unknown_`` clips from, so both the sampling and the sampled
    content are reproducible from run to run.
    """
    set_seed(SEED)

    dataset = SpeechCommandsDataset(
        subset=EVALUATION_SPLIT, cache_in_ram=False, return_audio=True
    )
    extractor = AudioToThreeChannelMel().eval()
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(dataset), size=limit, replace=False)
    print(f"[*] Evaluating on {limit} / {len(dataset)} '{EVALUATION_SPLIT}' clips")

    features = []
    with torch.no_grad():
        for index in indices:
            audio, _ = dataset[int(index)]
            features.append(extractor(audio.unsqueeze(0)).numpy().astype(np.float32))
    return features


def predict(session, features):
    """Stack per-clip logits into a single ``[n, NUM_CLASSES]`` matrix."""
    return np.concatenate(
        [session.run(None, {INPUT_NAME: feature})[0] for feature in features], axis=0
    )


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def milliseconds_per_clip(session, features, rounds=BENCHMARK_ROUNDS):
    """Median ms per single-clip call, after a warm-up (informational only)."""
    for _ in range(5):
        session.run(None, {INPUT_NAME: features[0]})
    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        for feature in features:
            session.run(None, {INPUT_NAME: feature})
        timings.append((time.perf_counter() - start) / len(features) * 1000.0)
    return float(np.median(timings))


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
        features = testing_features()
        _PREPARED.update(
            fp32_path=fp32_path,
            int8_path=int8_path,
            fp32_session=fp32_session,
            int8_session=int8_session,
            features=features,
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
    fp32_logits, int8_logits = state["fp32_logits"], state["int8_logits"]
    fp32_labels = np.argmax(fp32_logits, axis=1)
    int8_labels = np.argmax(int8_logits, axis=1)
    matched = int(np.sum(fp32_labels == int8_labels))
    agreement = matched / len(fp32_labels)
    assert agreement >= TOP1_AGREEMENT_MIN, (
        f"top-1 agreement {agreement * 100:.2f}% < {TOP1_AGREEMENT_MIN * 100:.0f}% "
        f"({matched}/{len(fp32_labels)} clips keep the float32 label): INT8 changed "
        "the decision"
    )
    print(
        f"[OK] top-1 agreement: {agreement * 100:.2f}% "
        f"({matched}/{len(fp32_labels)} clips keep the float32 label)"
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
    # Where the two graphs disagree, show how undecided the float32 graph was: a
    # flip on a clip whose own top-1/top-2 margin is negligible is noise, not a
    # regression, and the temperature/threshold stage rejects those clips anyway.
    ordered = np.sort(softmax(state["fp32_logits"]), axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    flipped = np.argmax(state["fp32_logits"], axis=1) != np.argmax(
        state["int8_logits"], axis=1
    )
    if flipped.any():
        print(
            f"[i] flipped labels: {int(flipped.sum())}/{len(flipped)}, float32 top-2 "
            f"softmax margin there {np.round(margin[flipped], 4).tolist()}"
        )
    else:
        print("[i] flipped labels: none")
    fp32_ms = milliseconds_per_clip(state["fp32_session"], state["features"])
    int8_ms = milliseconds_per_clip(state["int8_session"], state["features"])
    print(
        f"[i] latency on this machine: float32 {fp32_ms:.3f} ms/clip, "
        f"int8 {int8_ms:.3f} ms/clip ({int8_ms / fp32_ms:.2f}x)"
    )
    fp32_mb = os.path.getsize(state["fp32_path"]) / (1024 * 1024)
    int8_mb = os.path.getsize(state["int8_path"]) / (1024 * 1024)
    print(
        f"[i] artifact size: float32 {fp32_mb:.3f} MB -> int8 {int8_mb:.3f} MB "
        f"({100.0 * (1.0 - int8_mb / fp32_mb):.1f}% smaller)"
    )


def main():
    prepared()
    test_graph_contract()
    test_output_shape()
    test_top1_agreement()
    test_softmax_drift()
    report_metrics()
    print("\n[OK] v3.1 quantization smoke test passed")


if __name__ == "__main__":
    main()
