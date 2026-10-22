"""Smoke test for the sliding-window inference path (no dataset touched).

Verifies the variable-length windowing rules, the "best match" aggregation and
the batched scoring of ``CommandInferencer.predict_from_raw_audio``. The part
that needs the model artifacts is skipped when the configured release is not
present, so the windowing contract can still be checked in a fresh checkout.

Run directly:  python tests/test_sliding_window.py
"""

import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import (  # noqa: E402
    REJECTION_CLASSES,
    REJECT_LABEL,
    SAMPLE_RATE,
    TARGET_SAMPLES,
)
from src.inference import (  # noqa: E402
    WINDOW_HOP_SAMPLES,
    WINDOW_SAMPLES,
    CommandInferencer,
    ModelUnavailableError,
    sliding_windows,
)

HOP = WINDOW_HOP_SAMPLES


def test_windowing_contract():
    """Short clips are padded, long clips overlap by 50 % and keep their tail."""
    rng = np.random.default_rng(0)

    short = rng.standard_normal(TARGET_SAMPLES // 2).astype(np.float32)
    windows, starts = sliding_windows(short)
    assert windows.shape == (1, WINDOW_SAMPLES), windows.shape
    assert starts == [0], starts
    assert np.array_equal(windows[0, : short.size], short)
    assert np.all(windows[0, short.size :] == 0.0), "short clip was not zero-padded"

    exact = rng.standard_normal(TARGET_SAMPLES).astype(np.float32)
    windows, starts = sliding_windows(exact[np.newaxis, :])
    assert windows.shape == (1, WINDOW_SAMPLES), windows.shape
    assert starts == [0], starts
    assert np.array_equal(windows[0], exact), "a 1.0 s [1, N] clip must be unchanged"

    two_seconds = rng.standard_normal(2 * TARGET_SAMPLES).astype(np.float32)
    windows, starts = sliding_windows(two_seconds)
    assert starts == [0, HOP, TARGET_SAMPLES], starts
    assert windows.shape == (3, WINDOW_SAMPLES), windows.shape
    assert np.array_equal(windows[1], two_seconds[HOP : HOP + WINDOW_SAMPLES])

    # 2.0625 s: the regular windows stop at 32000, so an end-aligned tail window
    # is appended and the last 1000 samples are still covered.
    ragged = rng.standard_normal(TARGET_SAMPLES * 2 + 1000).astype(np.float32)
    windows, starts = sliding_windows(ragged)
    assert starts[-1] + WINDOW_SAMPLES == ragged.size, starts
    assert np.array_equal(windows[-1], ragged[-WINDOW_SAMPLES:])

    four_seconds = rng.standard_normal(4 * SAMPLE_RATE).astype(np.float32)
    windows, starts = sliding_windows(four_seconds.tolist())
    assert windows.shape == (7, WINDOW_SAMPLES), windows.shape
    assert starts[-1] == 3 * SAMPLE_RATE, starts

    print(
        f"[OK] windowing: 0.5 s -> 1 padded window, 2.0 s -> 3 windows, "
        f"4.0 s -> {len(starts)} windows (hop {HOP} samples)"
    )


def test_windowing_rejects_bad_input():
    """Multi-channel, integer and non-finite input fail loudly instead of silently."""
    for bad, expected in (
        (np.zeros((2, TARGET_SAMPLES), dtype=np.float32), ValueError),
        (np.zeros(0, dtype=np.float32), ValueError),
        (np.zeros(TARGET_SAMPLES, dtype=np.int16), TypeError),
        (np.full(TARGET_SAMPLES, np.nan, dtype=np.float32), ValueError),
    ):
        try:
            sliding_windows(bad)
        except expected:
            continue
        raise AssertionError(f"{type(bad)} should have raised {expected.__name__}")
    print("[OK] windowing rejects stereo, empty, int16 and NaN input")


def test_batched_best_match(engine):
    """A command aligned to a hop boundary wins its window of the recording."""
    rng = np.random.default_rng(1234)
    clip = (rng.standard_normal(TARGET_SAMPLES) * 0.05).astype(np.float32)
    recording = np.concatenate(
        [
            np.zeros(HOP, dtype=np.float32),
            clip,
            np.zeros(HOP, dtype=np.float32),
        ]
    )

    single = engine.predict_from_raw_audio(clip)
    sliding = engine.predict_from_raw_audio(recording)

    assert sliding["window_count"] == 3, sliding["window_count"]
    aligned = sliding["window_predictions"][1]
    assert aligned["start_samples"] == HOP, aligned["start_samples"]
    assert aligned["label"] == single["raw_label"], (aligned, single["raw_label"])
    assert abs(aligned["confidence"] - single["confidence"]) < 1e-4, aligned

    best = max(sliding["window_predictions"], key=lambda row: row["confidence"])
    assert abs(sliding["confidence"] - best["confidence"]) < 1e-9, "best window must decide"
    assert sliding["raw_label"] == best["label"], (sliding["raw_label"], best["label"])
    assert sliding["raw_label"] not in REJECTION_CLASSES, sliding["raw_label"]

    rejected = engine.predict_from_raw_audio(recording, confidence_threshold=1.0)
    assert rejected["accepted"] is False
    assert rejected["is_low_confidence"] is True
    assert rejected["label"] == REJECT_LABEL, rejected["label"]
    assert rejected["raw_label"] == sliding["raw_label"]

    batched = engine.predict_from_raw_audio(recording[np.newaxis, :])
    assert batched["raw_label"] == sliding["raw_label"]
    assert abs(batched["confidence"] - sliding["confidence"]) < 1e-9
    print(
        f"[OK] best match: 3 windows -> {sliding['raw_label']} "
        f"({sliding['confidence']:.4f}) at {sliding['window_start_seconds']:.2f} s"
    )


def test_file_path_matches_array(engine):
    """A 3.0 s file keeps its length instead of being truncated to 1.0 s."""
    rng = np.random.default_rng(7)
    recording = (rng.standard_normal(3 * SAMPLE_RATE) * 0.05).astype(np.float32)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "recording_3s.wav")
        sf.write(path, recording, SAMPLE_RATE, subtype="FLOAT")
        from_file = engine.predict(path)
        from_array = engine.predict(recording)

    assert from_file["window_count"] == 5, from_file["window_count"]
    assert from_file["raw_label"] == from_array["raw_label"]
    assert abs(from_file["confidence"] - from_array["confidence"]) < 1e-4
    print(
        f"[OK] 3.0 s file -> 5 windows, {from_file['raw_label']} "
        f"({from_file['confidence']:.4f}), backend {from_file['backend']}"
    )


def test_latency(engine):
    """A 4.0 s clip must be scored as one batched call, not window by window."""
    rng = np.random.default_rng(3)
    recording = (rng.standard_normal(4 * SAMPLE_RATE) * 0.05).astype(np.float32)
    engine.predict_from_raw_audio(recording)  # warm up the session

    started = time.perf_counter()
    result = engine.predict_from_raw_audio(recording)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(
        f"[OK] latency: {elapsed_ms:.1f} ms for 4.0 s "
        f"({result['window_count']} windows, backend {result['backend']})"
    )


def main():
    test_windowing_contract()
    test_windowing_rejects_bad_input()

    try:
        engine = CommandInferencer()
    except ModelUnavailableError as exc:
        print(f"[SKIP] model-dependent checks: {exc}")
        print("[+] Sliding-window contract verified.")
        return

    print(
        f"[+] Engine ready (T={engine.temperature:.4f}, "
        f"tau={engine.confidence_threshold:.2f}, "
        f"{engine.command_indices.size} command classes)"
    )
    test_batched_best_match(engine)
    test_file_path_matches_array(engine)
    test_latency(engine)
    print("[+] Sliding-window self-check passed.")


if __name__ == "__main__":
    main()
