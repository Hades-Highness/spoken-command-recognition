"""Calibration-aware evaluation of CommandSense (v2.2).

The run has two halves, and they never share a split:

1. **Fit** the temperature ``T`` and the confidence threshold ``tau`` on the
   *validation* split, which the model never trained on.
2. **Measure** the effect on the *testing* split: ECE before/after, coverage /
   retained accuracy / FRR / FAR at ``tau*``, and the same 37x37 confusion
   matrix as v2.1 so the two versions stay comparable.

Artifacts written to ``reports/model_v2.2/``::

    report_v2.2.txt                 human-readable metrics
    calibration_v2.2.png            confidence vs accuracy, before/after T
    coverage_vs_accuracy_v2.2.png   coverage / retained accuracy / FRR / FAR vs tau
    confusion_matrix_v2.2.png       37x37 confusion matrix of the evaluation split

The deployment artifact read by ``src/inference.py`` and ``app.py`` is written to
``configs/calibration/CommandSense_calibration_v2.2.json``: it is versioned
configuration rather than a weight, so it is kept out of both the report folder
and the git-ignored ``checkpoints/`` tree.

Like ``train.py``, an interrupted run still writes what it has, under an
``_interrupted`` file-name suffix, and skips the deployment JSON so a partial fit
can never be served by accident.
"""

import gc
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from train import set_seed

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs.config import (  # noqa: E402
    BATCH_SIZE,
    CALIBRATION_BINS,
    CHECKPOINT_DIR,
    DEFAULT_TEMPERATURE,
    LABELS_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    NUM_CLASSES,
    NUM_WORKERS,
    REJECTION_CLASSES,
    REJECT_LABEL,
    SEED,
    THRESHOLD_CRITERION,
    THRESHOLD_MIN_COVERAGE,
    calibration_path,
    checkpoint_dir,
    ensure_version_dirs,
    reports_dir,
)
from src import calibration  # noqa: E402
from src.dataset import SpeechCommandsDataset  # noqa: E402
from src.models import CommandSense  # noqa: E402
import src.utils  # noqa: E402,F401  (installs the torchaudio.load fallback early)

CALIBRATION_SUBSET = "validation"   # fits T and tau; never seen during training
EVALUATION_SUBSET = "testing"       # reports every headline number


def resolve_checkpoint(version=MODEL_VERSION):
    """Return the weights of ``version`` to evaluate, or ``None`` when absent.

    v3.0 is a self-contained release that trains its own weights, so only the
    artifacts of ``version`` are considered - there is no base release to borrow
    a checkpoint from.
    """
    candidates = [
        checkpoint_dir(version) / f"{MODEL_NAME}_{version}.pth",
        CHECKPOINT_DIR / f"{MODEL_NAME}_{version}.pth",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return str(candidate)
    return None


def verify_class_mapping(class_names):
    """Fail loudly when ``configs/labels.json`` disagrees with the dataset order.

    A drifted mapping would silently make every reported number meaningless.
    """
    if not os.path.exists(LABELS_PATH):
        print(f"[!] {LABELS_PATH} not found: index order is unverified.")
        return
    with open(LABELS_PATH, "r", encoding="utf-8") as handle:
        declared_labels = json.load(handle)
    if declared_labels != class_names:
        raise RuntimeError(
            f"Class mapping mismatch: '{LABELS_PATH}' differs from the split's "
            "label order. Delete configs/labels.json and re-run train.py."
        )
    print(f"[+] Class mapping verified against {LABELS_PATH} ({len(class_names)} classes)")


def collect_split_logits(model, device, subset, logit_parts, target_parts):
    """Append the raw logits of ``subset`` batch by batch.

    The features are cached in RAM split by split and the dataset is released
    before the caller reads the next split, so a run never holds two splits at
    once. Appending instead of concatenating means Ctrl+C keeps every completed
    batch, exactly like the epoch history of ``train.py``. The class order of the
    split is returned so the caller can verify it before trusting any index.
    """
    print(f"[*] Caching and scoring the '{subset}' split (raw logits)...")
    # evaluate.py scores pre-computed features (legacy contract); training serves
    # raw int16 audio and extracts features on the GPU via src/transforms.py.
    dataset = SpeechCommandsDataset(
        subset=subset, cache_in_ram=True, return_audio=False
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    class_names = list(dataset.labels)
    try:
        with torch.no_grad():
            for features, targets in loader:
                logit_parts.append(model(features.to(device)).cpu().numpy())
                target_parts.append(targets.numpy())
    finally:
        del loader, dataset
    print(f"[+] Scored {sum(part.shape[0] for part in logit_parts)} clips from '{subset}'")
    return class_names


def stack_parts(parts):
    """Concatenate accumulated batches, or return ``None`` when nothing arrived."""
    if not parts:
        return None
    return np.concatenate(parts, axis=0).astype(np.float64)


def split_summary(logits, targets, temperature, threshold):
    """Every metric reported for one split, calibrated with ``temperature``."""
    probabilities_before = calibration.logits_to_probabilities(logits, DEFAULT_TEMPERATURE)
    probabilities_after = calibration.logits_to_probabilities(logits, temperature)
    diagram_before = calibration.reliability_diagram(
        probabilities_before, targets, n_bins=CALIBRATION_BINS
    )
    diagram_after = calibration.reliability_diagram(
        probabilities_after, targets, n_bins=CALIBRATION_BINS
    )
    return {
        "n_samples": int(targets.shape[0]),
        "accuracy": calibration.accuracy(probabilities_after, targets),
        "nll_before": calibration.negative_log_likelihood(
            logits, targets, DEFAULT_TEMPERATURE
        ),
        "nll_after": calibration.negative_log_likelihood(logits, targets, temperature),
        "ece_before": diagram_before["ece"],
        "ece_after": diagram_after["ece"],
        "mce_before": diagram_before["mce"],
        "mce_after": diagram_after["mce"],
        "diagram_before": diagram_before,
        "diagram_after": diagram_after,
        "probabilities_after": probabilities_after,
        "threshold": calibration.threshold_metrics(probabilities_after, targets, threshold),
    }


def format_quality_block(title, summary, temperature):
    """NLL/ECE/MCE before and after temperature scaling, for the text report."""
    lines = [
        f"--- {title} ---",
        f"Samples: {summary['n_samples']}",
        "Top-1 accuracy (unchanged by temperature scaling): "
        f"{summary['accuracy'] * 100:.4f}%",
        "",
        f"{'':14s}{'before (T=1)':>16s}{'after (T=' + format(temperature, '.4f') + ')':>22s}",
        f"{'NLL':14s}{summary['nll_before']:>16.6f}{summary['nll_after']:>22.6f}",
        f"{'ECE':14s}{summary['ece_before']:>16.6f}{summary['ece_after']:>22.6f}",
        f"{'MCE':14s}{summary['mce_before']:>16.6f}{summary['mce_after']:>22.6f}",
        "",
    ]
    return "\n".join(lines)


def format_sweep_table(sweep, threshold):
    """Coverage / accuracy / FRR / FAR at round thresholds plus ``tau*``."""
    grid = np.round(np.arange(0.0, 1.0, 0.1), 6)
    # tau* is part of the sweep it was selected from, so drop the grid point it
    # coincides with and keep a single, marked row for it at the bottom.
    wanted = [tau for tau in grid if abs(tau - float(threshold)) > 1e-9]
    wanted.append(round(float(threshold), 6))
    lines = [
        f"{'tau':>6s}{'coverage':>11s}{'retained acc':>14s}{'FRR':>9s}{'FAR':>9s}",
    ]
    for tau in wanted:
        row = min(sweep, key=lambda entry: abs(entry["tau"] - tau))
        marker = " <- tau*" if abs(row["tau"] - float(threshold)) < 1e-9 else ""
        lines.append(
            f"{row['tau']:>6.2f}{row['coverage'] * 100:>10.2f}%"
            f"{row['retained_accuracy'] * 100:>13.2f}%"
            f"{row['frr'] * 100:>8.2f}%{row['far'] * 100:>8.2f}%{marker}"
        )
    return "\n".join(lines)


def plot_reliability_diagram(summary, temperature, title, out_path):
    """Two-panel reliability diagram: raw softmax (T=1) next to calibrated."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d1117")
    panels = (
        (axes[0], summary["diagram_before"], "T = 1 (raw softmax)", "#f85149"),
        (axes[1], summary["diagram_after"], f"T = {temperature:.4f} (fitted)", "#58a6ff"),
    )
    for ax, diagram, label, color in panels:
        ax.set_facecolor("#0d1117")
        centres = (diagram["bin_edges"][:-1] + diagram["bin_edges"][1:]) / 2.0
        width = (diagram["bin_edges"][1] - diagram["bin_edges"][0]) * 0.9
        ax.bar(
            centres,
            diagram["bin_accuracies"],
            width=width,
            color=color,
            alpha=0.75,
            label="Accuracy",
        )
        ax.plot([0, 1], [0, 1], "--", color="#8b949e", linewidth=1.5, label="Perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence (top-1 probability)", color="#8b949e")
        ax.set_ylabel("Accuracy", color="#8b949e")
        ax.set_title(
            f"{label}\nECE = {diagram['ece']:.4f} | MCE = {diagram['mce']:.4f}",
            color="#c9d1d9",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(title, color="#c9d1d9", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_path}")


def plot_coverage_vs_accuracy(sweep, threshold, title, out_path):
    """Coverage / retained accuracy / FRR / FAR as a function of tau."""
    taus = np.array([row["tau"] for row in sweep])
    coverage = np.array([row["coverage"] for row in sweep]) * 100.0
    accuracy = np.array([row["retained_accuracy"] for row in sweep]) * 100.0
    frr = np.array([row["frr"] for row in sweep]) * 100.0
    far = np.array([row["far"] for row in sweep]) * 100.0

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d1117")
    ax1.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")

    ax1.plot(taus, coverage, color="#58a6ff", linewidth=2, label="Coverage")
    ax1.plot(taus, accuracy, color="#f85149", linewidth=2, label="Retained accuracy")
    ax1.set_xlabel("Confidence threshold tau", color="#8b949e")
    ax1.set_ylabel("Percent", color="#8b949e")
    ax1.set_title("Coverage vs Accuracy", color="#c9d1d9", fontsize=12, fontweight="bold")

    ax2.plot(taus, frr, color="#58a6ff", linewidth=2, label="FRR (correct rejected)")
    ax2.plot(taus, far, color="#f85149", linewidth=2, label="FAR (wrong served)")
    ax2.set_xlabel("Confidence threshold tau", color="#8b949e")
    ax2.set_ylabel("Percent", color="#8b949e")
    ax2.set_title(
        "False Rejection vs False Acceptance", color="#c9d1d9", fontsize=12, fontweight="bold"
    )

    for ax in (ax1, ax2):
        ax.axvline(threshold, color="#c9d1d9", linestyle="--", linewidth=1.5)
        ax.annotate(
            f"tau* = {threshold:.2f}",
            xy=(threshold, ax.get_ylim()[1]),
            xytext=(4, -12),
            textcoords="offset points",
            color="#c9d1d9",
            fontsize=9,
        )
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend(fontsize=9)

    fig.suptitle(title, color="#c9d1d9", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_path}")


def plot_confusion_matrix(targets, predictions, class_names, title, out_path):
    """The same 37x37 raw-argmax matrix v2.1 reports, for comparability."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 14), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    sns.heatmap(
        confusion_matrix(targets, predictions),
        annot=False,
        fmt="d",
        cmap="mako",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    plt.title(title, color="#c9d1d9", fontsize=16, pad=20)
    plt.xlabel("Predicted Label", color="#8b949e", fontsize=12)
    plt.ylabel("True Label", color="#8b949e", fontsize=12)
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_path}")





def build_report_text(meta, calibration_summary, evaluation_summary, sweep, class_report, recalls):
    """Assemble the plain-text report written next to the plots."""
    rule = "=" * 60
    lines = [
        rule,
        f"     CALIBRATION & EVALUATION REPORT ({MODEL_NAME} {MODEL_VERSION})",
        rule,
        "",
        f"Model version    : {MODEL_VERSION} (trained weights, calibrated post-hoc)",
        f"Checkpoint       : {meta['checkpoint']}",
        f"Device           : {meta['device']}",
        f"Calibration split: '{CALIBRATION_SUBSET}' "
        f"({calibration_summary['n_samples']} samples) - fits T and tau",
        f"Evaluation split : '{EVALUATION_SUBSET}'"
        + (f" ({evaluation_summary['n_samples']} samples)" if evaluation_summary else "")
        + " - untouched by the fit",
        f"Temperature      : T = {meta['temperature']:.6f} (NLL minimisation, "
        f"{CALIBRATION_BINS} bins)",
        f"Threshold        : tau* = {meta['tau']:.6f} (criterion '{THRESHOLD_CRITERION}', "
        f"minimum coverage {THRESHOLD_MIN_COVERAGE * 100:.2f}%)",
        f"Reject label     : {REJECT_LABEL}",
        "",
    ]

    if meta.get("interrupted"):
        lines += [
            "!! PARTIAL RUN: interrupted with Ctrl+C, so these numbers describe the",
            "!! clips collected so far. No deployment JSON was written for this run.",
            "",
        ]

    lines.append(format_quality_block(
        f"CALIBRATION QUALITY ({CALIBRATION_SUBSET} split, where T was fitted)",
        calibration_summary,
        meta["temperature"],
    ))

    if evaluation_summary is not None:
        lines.append(format_quality_block(
            f"CALIBRATION QUALITY ({EVALUATION_SUBSET} split, out-of-sample)",
            evaluation_summary,
            meta["temperature"],
        ))

        decision = evaluation_summary["threshold"]
        all_correct = evaluation_summary["accuracy"]
        lines += [
            f"--- CONFIDENCE THRESHOLD DECISION ({EVALUATION_SUBSET} split, "
            f"tau* = {meta['tau']:.2f}) ---",
            f"Coverage (clips answered)       : {decision['coverage'] * 100:.4f}% "
            f"({decision['n_accepted']}/{decision['n_total']})",
            "Retained accuracy (answered)    : "
            f"{decision['retained_accuracy'] * 100:.4f}% "
            f"({decision['accepted_correct']}/{decision['n_accepted']})",
            f"False rejection rate (FRR)      : {decision['frr'] * 100:.4f}% of correct "
            "predictions discarded",
            f"False acceptance rate (FAR)     : {decision['far'] * 100:.4f}% of incorrect "
            "predictions served",
            f"Without any threshold (argmax)  : {all_correct * 100:.4f}% of all clips correct",
            "End-to-end yield                : "
            f"{decision['coverage'] * decision['retained_accuracy'] * 100:.4f}% of all "
            "clips answered and correct",
            f"Rejected as uncertain           : {decision['rejection_rate'] * 100:.4f}% "
            f"served as '{REJECT_LABEL}'",
            "",
            f"--- THRESHOLD SWEEP ({EVALUATION_SUBSET} split, calibrated confidence) ---",
            format_sweep_table(sweep, meta["tau"]),
            "",
            f"--- SEPARATED METRICS (raw argmax, no threshold) ---",
            f"Top-1 Accuracy (35 Commands): {meta['cmd_accuracy']:.4f}% "
            f"({meta['cmd_correct']}/{meta['cmd_total']})",
        ]
        for label, (recall, correct, total) in recalls.items():
            lines.append(f"Recall '{label}':               {recall:.4f}% ({correct}/{total})")
        lines += [
            "",
            f"--- DETAILED CLASSIFICATION REPORT (raw argmax) ---",
            class_report.rstrip("\n"),
            "",
        ]

    lines += [
        rule,
        "NOTES",
        rule + "",
        "* Temperature scaling is applied to the logits before the softmax, so it never",
        "  changes the argmax: the top-1 accuracy of v2.1 and v2.2 is identical by",
        "  construction, and only the reported confidence changes.",
        "* FRR and FAR are conditional rates: FRR is measured over the correct (raw)",
        "  predictions and FAR over the incorrect ones, so they are directly comparable",
        "  between operating points.",
        f"* The exported ONNX graph keeps raw logits; T and tau are applied by the Python",
        f"  runtime, so the threshold can be changed without re-exporting the model.",
        "",
    ]
    return "\n".join(lines) + "\n"


def rejection_recalls(predictions, targets, class_names):
    """Per-class recall of ``_silence_`` / ``_unknown_``, as reported by v2.1."""
    recalls = {}
    for label in REJECTION_CLASSES:
        if label not in class_names:
            continue
        index = class_names.index(label)
        mask = targets == index
        total = int(np.count_nonzero(mask))
        correct = int(np.count_nonzero(predictions[mask] == index)) if total else 0
        recalls[label] = (100.0 * correct / total if total else 0.0, correct, total)
    return recalls


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware device: {device}")

    checkpoint_path = resolve_checkpoint()
    if checkpoint_path is None:
        print(
            f"[!] No checkpoint found at '{checkpoint_dir(MODEL_VERSION)}/"
            f"{MODEL_NAME}_{MODEL_VERSION}.pth'. Download the release asset or run train.py."
        )
        return
    print(f"[*] Loading model checkpoint: {checkpoint_path}")
    checkpoint_reference = os.path.relpath(
        checkpoint_path, os.path.dirname(__file__)
    ).replace(os.sep, "/")

    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # ------------------------------------------------- fit on the val split
    interrupted = False
    calibration_logit_parts, calibration_target_parts = [], []
    class_names = None
    try:
        class_names = collect_split_logits(
            model,
            device,
            CALIBRATION_SUBSET,
            calibration_logit_parts,
            calibration_target_parts,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[!] Interrupted while reading the calibration split.")
    if class_names is None:
        print("[!] The calibration split was never read; nothing to write.")
        return
    verify_class_mapping(class_names)

    calibration_logits = stack_parts(calibration_logit_parts)
    if calibration_logits is None:
        print("[!] No calibration sample was collected; nothing to write.")
        return
    calibration_targets = np.concatenate(calibration_target_parts, axis=0).astype(np.int64)
    del calibration_logit_parts, calibration_target_parts

    temperature = calibration.fit_temperature(calibration_logits, calibration_targets)
    print(f"[+] Fitted temperature on the '{CALIBRATION_SUBSET}' split: T = {temperature:.6f}")

    calibrated_probabilities = calibration.logits_to_probabilities(
        calibration_logits, temperature
    )
    calibration_sweep = calibration.sweep_thresholds(
        calibrated_probabilities, calibration_targets
    )
    best = calibration.select_threshold(
        calibration_sweep,
        criterion=THRESHOLD_CRITERION,
        min_coverage=THRESHOLD_MIN_COVERAGE,
    )
    tau = float(best["tau"])
    print(
        f"[+] Threshold tau* = {tau:.2f} ('{THRESHOLD_CRITERION}'): "
        f"coverage {best['coverage'] * 100:.2f}%, "
        f"retained accuracy {best['retained_accuracy'] * 100:.2f}%, "
        f"FRR {best['frr'] * 100:.2f}%, FAR {best['far'] * 100:.2f}%"
    )
    calibration_summary = split_summary(
        calibration_logits, calibration_targets, temperature, tau
    )
    # Release the calibration split before the evaluation split is cached in RAM.
    del calibration_logits, calibrated_probabilities
    gc.collect()

    # ------------------------------------------- measure on the test split
    evaluation_logit_parts, evaluation_target_parts = [], []
    evaluation_logits, evaluation_targets = None, None
    evaluation_labels = None
    if not interrupted:
        try:
            evaluation_labels = collect_split_logits(
                model,
                device,
                EVALUATION_SUBSET,
                evaluation_logit_parts,
                evaluation_target_parts,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("\n[!] Interrupted while reading the evaluation split.")
        evaluation_logits = stack_parts(evaluation_logit_parts)
        if evaluation_target_parts:
            evaluation_targets = np.concatenate(
                evaluation_target_parts, axis=0
            ).astype(np.int64)
        del evaluation_logit_parts, evaluation_target_parts
        if evaluation_labels is not None and evaluation_labels != class_names:
            raise RuntimeError(
                f"'{CALIBRATION_SUBSET}' and '{EVALUATION_SUBSET}' disagree on the class "
                "order, so the calibrated scores would be meaningless."
            )

    evaluation_summary = None
    evaluation_sweep = calibration_sweep
    class_report = "Not computed: the evaluation split was not read to the end."
    recalls = {}
    command_correct, command_total = 0, 0
    if evaluation_logits is not None and evaluation_targets is not None:
        evaluation_summary = split_summary(
            evaluation_logits, evaluation_targets, temperature, tau
        )
        evaluation_sweep = calibration.sweep_thresholds(
            evaluation_summary["probabilities_after"], evaluation_targets
        )
        predictions = np.argmax(evaluation_logits, axis=1)
        class_report = classification_report(
            evaluation_targets, predictions, target_names=class_names, digits=4
        )
        recalls = rejection_recalls(predictions, evaluation_targets, class_names)
        rejection_indices = [
            class_names.index(label) for label in REJECTION_CLASSES if label in class_names
        ]
        command_mask = ~np.isin(evaluation_targets, rejection_indices)
        command_correct = int(
            np.count_nonzero(predictions[command_mask] == evaluation_targets[command_mask])
        )
        command_total = int(np.count_nonzero(command_mask))

    # ------------------------------------------------------- console block
    print("\n" + "=" * 60)
    print(f"       EVALUATION REPORT ({MODEL_NAME} {MODEL_VERSION})       ")
    print("=" * 60)
    print(
        f"Temperature (fitted on '{CALIBRATION_SUBSET}'): T = {temperature:.6f}  |  "
        f"tau* = {tau:.2f} ({THRESHOLD_CRITERION})"
    )
    print("-" * 60)
    print(
        f"Calibration ({CALIBRATION_SUBSET}): ECE {calibration_summary['ece_before']:.4f} -> "
        f"{calibration_summary['ece_after']:.4f} | NLL "
        f"{calibration_summary['nll_before']:.4f} -> {calibration_summary['nll_after']:.4f}"
    )
    if evaluation_summary is not None:
        decision = evaluation_summary["threshold"]
        print(
            f"Calibration ({EVALUATION_SUBSET}): ECE {evaluation_summary['ece_before']:.4f} -> "
            f"{evaluation_summary['ece_after']:.4f} | NLL "
            f"{evaluation_summary['nll_before']:.4f} -> {evaluation_summary['nll_after']:.4f}"
        )
        print(
            f"Overall Accuracy (37 classes): {evaluation_summary['accuracy'] * 100:.2f}% "
            f"({int(round(evaluation_summary['accuracy'] * evaluation_summary['n_samples']))}"
            f"/{evaluation_summary['n_samples']}) - unchanged by T (logit rescaling)"
        )
        print(
            f"35 Command Words Top-1 Acc:    {100.0 * command_correct / command_total:.2f}% "
            f"({command_correct}/{command_total})"
        )
        for label, (recall, correct, total) in recalls.items():
            print(f"Rejection Recall '{label}':    {recall:.2f}% ({correct}/{total})")
        print("-" * 60)
        print(
            f"At tau* = {tau:.2f}: coverage {decision['coverage'] * 100:.2f}% | retained "
            f"accuracy {decision['retained_accuracy'] * 100:.2f}% | FRR "
            f"{decision['frr'] * 100:.2f}% | FAR {decision['far'] * 100:.2f}%"
        )
        print(
            "End-to-end yield: "
            f"{decision['coverage'] * decision['retained_accuracy'] * 100:.2f}% of clips "
            "answered and correct"
        )
    if interrupted:
        print("-" * 60)
        print("PARTIAL RUN: Ctrl+C stopped the sweep, so the numbers above are incomplete.")
    print("=" * 60 + "\n")

    # --------------------------------------------------------- artifacts
    ensure_version_dirs()
    out_dir = reports_dir(MODEL_VERSION)
    suffix = "_interrupted" if interrupted else ""

    meta = {
        "checkpoint": checkpoint_reference,
        "device": str(device),
        "temperature": temperature,
        "tau": tau,
        "interrupted": interrupted,
        "cmd_accuracy": 100.0 * command_correct / command_total if command_total else 0.0,
        "cmd_correct": command_correct,
        "cmd_total": command_total,
    }
    report_path = os.path.join(out_dir, f"report_{MODEL_VERSION}{suffix}.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(
            build_report_text(
                meta,
                calibration_summary,
                evaluation_summary,
                evaluation_sweep,
                class_report,
                recalls,
            )
        )
    print(f"[+] Saved {report_path}")

    split_for_plots = EVALUATION_SUBSET if evaluation_summary is not None else CALIBRATION_SUBSET
    plot_reliability_diagram(
        evaluation_summary or calibration_summary,
        temperature,
        f"Reliability Diagram - {MODEL_NAME} {MODEL_VERSION} ({split_for_plots} split)",
        os.path.join(out_dir, f"calibration_{MODEL_VERSION}{suffix}.png"),
    )
    plot_coverage_vs_accuracy(
        evaluation_sweep,
        tau,
        f"Coverage vs Accuracy - {MODEL_NAME} {MODEL_VERSION} ({split_for_plots} split)",
        os.path.join(out_dir, f"coverage_vs_accuracy_{MODEL_VERSION}{suffix}.png"),
    )
    if evaluation_summary is not None:
        plot_confusion_matrix(
            evaluation_targets,
            np.argmax(evaluation_summary["probabilities_after"], axis=1),
            class_names,
            f"Confusion Matrix - {MODEL_NAME} {MODEL_VERSION} ({EVALUATION_SUBSET} split)",
            os.path.join(out_dir, f"confusion_matrix_{MODEL_VERSION}{suffix}.png"),
        )
    del evaluation_logits, evaluation_targets
    gc.collect()

    # ---------------------------------------------- deployment artifact
    reported = evaluation_summary or calibration_summary
    metadata = calibration.build_calibration_metadata(
        temperature=temperature,
        confidence_threshold=tau,
        version=MODEL_VERSION,
        n_bins=CALIBRATION_BINS,
        ece_before=reported["ece_before"],
        ece_after=reported["ece_after"],
        nll_before=reported["nll_before"],
        nll_after=reported["nll_after"],
        threshold_criterion=THRESHOLD_CRITERION,
        calibration_split=f"{CALIBRATION_SUBSET} ({calibration_summary['n_samples']} samples)",
        calibration_samples=calibration_summary["n_samples"],
        source_checkpoint=checkpoint_reference,
        reject_label=REJECT_LABEL,
    )
    if interrupted:
        partial_path = calibration_path(MODEL_VERSION).with_name(
            f"{MODEL_NAME}_calibration_{MODEL_VERSION}{suffix}.json"
        )
        print(
            "[!] Partial run: the calibration JSON is written as\n"
            f"    '{partial_path}' only, so a half-fitted temperature is never\n"
            "    served by src/inference.py."
        )
        calibration.save_calibration(metadata, path=partial_path)
    else:
        json_path = calibration.save_calibration(metadata)
        print(f"[+] Saved {json_path}  <- loaded by app.py and src/inference.py")
    print(
        "[*] T and tau are applied to the raw logits at inference time; the ONNX graph "
        "itself stays raw-logit."
    )


if __name__ == "__main__":
    main()
