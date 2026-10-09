"""Throwaway smoke test for the evaluate.py helpers (no dataset touched)."""
import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
 
import evaluate
from configs.config import MODEL_VERSION, THRESHOLD_CRITERION, THRESHOLD_MIN_COVERAGE
from src import calibration

rng = np.random.default_rng(7)
n_classes = 37
targets = rng.integers(0, n_classes, 1500).astype(np.int64)
logits = rng.normal(0.0, 1.0, size=(1500, n_classes))
logits[np.arange(1500), targets] += rng.normal(1.6, 0.5, size=1500)  # over-confident model

temperature = calibration.fit_temperature(logits, targets)
print(f"T = {temperature:.6f}")
calibrated = calibration.logits_to_probabilities(logits, temperature)
sweep = calibration.sweep_thresholds(calibrated, targets)
best = calibration.select_threshold(
    sweep, criterion=THRESHOLD_CRITERION, min_coverage=THRESHOLD_MIN_COVERAGE
)
tau = float(best["tau"])
print(f"tau* = {tau:.2f}")

summary = evaluate.split_summary(logits, targets, temperature, tau)
class_names = [f"C{i}" for i in range(n_classes)]
predictions = np.argmax(logits, axis=1)
class_report = evaluate.rejection_recalls(predictions, targets, class_names)
print("recalls:", class_report)

print(evaluate.format_quality_block("DEMO", summary, temperature))
print(evaluate.format_sweep_table(sweep, tau))

# simulate "interrupted" (no evaluation split) and complete runs
partial = evaluate.build_report_text(
    {
        "checkpoint": "demo.pth",
        "device": "cpu",
        "temperature": temperature,
        "tau": tau,
        "interrupted": True,
        "cmd_accuracy": 0.0,
        "cmd_correct": 0,
        "cmd_total": 0,
    },
    summary,
    None,
    sweep,
    "not computed",
    {},
)
assert "PARTIAL RUN" in partial
assert "THRESHOLD SWEEP" not in partial
print("partial report ok:", len(partial.splitlines()), "lines")

full = evaluate.build_report_text(
    {
        "checkpoint": "demo.pth",
        "device": "cpu",
        "temperature": temperature,
        "tau": tau,
        "interrupted": False,
        "cmd_accuracy": 91.5,
        "cmd_correct": 900,
        "cmd_total": 984,
    },
    summary,
    summary,
    sweep,
    "              precision    recall\n   C0            0.9  0.8",
    {"_silence_": (99.0, 99, 100), "_unknown_": (55.0, 44, 80)},
)
assert "CONFIDENCE THRESHOLD DECISION" in full and "<- tau*" in full
print("full report ok:", len(full.splitlines()), "lines")
print("\n".join(full.splitlines()[:26]))

out_dir = tempfile.mkdtemp(prefix="cs_smoke_")
evaluate.plot_reliability_diagram(summary, temperature, "demo", os.path.join(out_dir, "r.png"))
evaluate.plot_coverage_vs_accuracy(sweep, tau, "demo", os.path.join(out_dir, "c.png"))
evaluate.plot_confusion_matrix(targets, predictions, class_names, "demo", os.path.join(out_dir, "m.png"))
print("plots:", sorted(os.listdir(out_dir)))

metadata = calibration.build_calibration_metadata(
    temperature=temperature,
    confidence_threshold=tau,
    version=MODEL_VERSION,
    n_bins=15,
    ece_before=summary["ece_before"],
    ece_after=summary["ece_after"],
    nll_before=summary["nll_before"],
    nll_after=summary["nll_after"],
    threshold_criterion=THRESHOLD_CRITERION,
    calibration_split="validation (1500 samples)",
    calibration_samples=1500,
    source_checkpoint="demo.pth",
    reject_label="_unknown_",
)
path = calibration.save_calibration(metadata, path=os.path.join(out_dir, "cal.json"))
print("json keys:", sorted(calibration.load_calibration(path).keys()))
print("[OK] smoke test passed")
