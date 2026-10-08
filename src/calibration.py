"""Post-hoc confidence calibration and confidence thresholding (v2.2).

This module implements the whole v2.2 feature set without touching a single
model weight - the v2.1 checkpoint is frozen and only its logits are rescaled:

* :class:`TemperatureScaler` - temperature scaling (Guo et al., 2017). A single
  scalar ``T > 0`` divides the logits before the softmax and is fitted on the
  held-out validation split by minimising the negative log-likelihood.
* :func:`expected_calibration_error` and :func:`reliability_diagram` - the
  confidence calibration diagnostics, measured with ``M`` equal-width
  probability bins.
* :func:`threshold_metrics` and :func:`sweep_thresholds` - coverage, retained
  accuracy, false rejection rate (FRR) and false acceptance rate (FAR) for a
  confidence threshold ``tau``; :func:`select_threshold` picks the operating
  point according to a named criterion.
* :func:`apply_confidence_threshold` - the online decision used by
  ``src/inference.py`` and ``app.py`` (``max softmax < tau`` -> reject).
* :func:`save_calibration` / :func:`load_calibration` - the deployment artifact
  consumed by the inferencer and mirrored into the ONNX metadata.

Every helper accepts NumPy arrays or PyTorch tensors and is free of dataset
imports, so the offline calibration pass (``evaluate.py``) and the served
prediction share exactly the same arithmetic.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import (  # noqa: E402
    CALIBRATION_BINS,
    CONFIDENCE_THRESHOLD_STEP,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TEMPERATURE,
    MODEL_NAME,
    MODEL_VERSION,
    REJECT_LABEL,
    TEMPERATURE_GRID,
    TEMPERATURE_LR,
    TEMPERATURE_MAX_ITER,
    TEMPERATURE_STEPS,
    TEMPERATURE_TOL,
    THRESHOLD_CRITERION,
    THRESHOLD_MIN_COVERAGE,
    calibration_path,
    checkpoint_dir,
)

THRESHOLD_CRITERIA = ("coverage_accuracy", "min_cost", "target_accuracy")

# Keys written by save_calibration(); load_calibration() tolerates missing keys
# so an older or hand-edited file degrades to the configured defaults.
CALIBRATION_METADATA_KEYS = (
    "model_name",
    "model_version",
    "created_utc",
    "temperature",
    "confidence_threshold",
    "reject_label",
    "n_bins",
    "ece_before",
    "ece_after",
    "nll_before",
    "nll_after",
    "nll_before_train",
    "threshold_criterion",
    "calibration_split",
    "calibration_samples",
    "source_checkpoint",
    "notes",
)

# ONNX metadata is a string dictionary, so the calibration payload is nested
# under a namespace that cannot collide with torch/onnxruntime producer fields.
ONNX_METADATA_PREFIX = "commandsense.calibration."


# ---------------------------------------------------------------------------
# Small conversion / maths helpers
# ---------------------------------------------------------------------------
def as_numpy(array) -> np.ndarray:
    """Return a detached ``float64`` NumPy view of a tensor or array."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64)


def softmax(logits, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax that also accepts raw Python sequences."""
    values = as_numpy(logits)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def logits_to_probabilities(logits, temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """Softmax after dividing the logits by ``temperature``."""
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    return softmax(as_numpy(logits) / temperature, axis=-1)


def negative_log_likelihood(logits, targets, temperature: float = DEFAULT_TEMPERATURE) -> float:
    """Mean negative log-likelihood of ``targets`` under the scaled softmax."""
    probabilities = logits_to_probabilities(logits, temperature)
    targets = as_numpy(targets).astype(np.int64)
    picked = probabilities[np.arange(targets.shape[0]), targets]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))


def accuracy(probabilities, targets) -> float:
    """Top-1 accuracy as a fraction in ``[0, 1]``."""
    probabilities = as_numpy(probabilities)
    targets = as_numpy(targets).astype(np.int64)
    return float(np.mean(np.argmax(probabilities, axis=-1) == targets))


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
class TemperatureScaler(nn.Module):
    """Single-parameter temperature fitted on a held-out split.

    The scalar is stored in log-space so that unconstrained gradient descent
    cannot drive the temperature to a non-positive value.
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(float(np.log(float(temperature))), dtype=torch.float32)
        )

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.exp(self.log_temperature)

    def fit(
        self,
        logits,
        targets,
        max_iter: int = TEMPERATURE_MAX_ITER,
        steps: int = TEMPERATURE_STEPS,
        lr: float = TEMPERATURE_LR,
        tol: float = TEMPERATURE_TOL,
        bounds: Sequence[float] = TEMPERATURE_GRID,
    ) -> "TemperatureScaler":
        """Minimise the NLL of the frozen model's logits.

        Only ``log_temperature`` receives gradients, so no weight of the
        underlying classifier is ever updated. Optimisation runs a bounded
        number of LBFGS steps (the NLL is convex in ``log T``); it stops early
        once the loss stops moving by more than ``tol``.
        """
        logits_tensor = torch.as_tensor(as_numpy(logits), dtype=torch.float32)
        targets_tensor = torch.as_tensor(
            as_numpy(targets).astype(np.int64), dtype=torch.long
        )
        if logits_tensor.shape[0] == 0:
            raise ValueError("Cannot fit a temperature on an empty logits array.")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=float(lr),
            max_iter=int(max_iter),
            tolerance_grad=float(tol),
            tolerance_change=float(tol),
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_tensor), targets_tensor)
            loss.backward()
            return loss

        previous_loss = None
        for _ in range(max(1, int(steps))):
            loss = optimizer.step(closure)
            current_loss = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
            if previous_loss is not None and abs(previous_loss - current_loss) < tol:
                break
            previous_loss = current_loss

        lower, upper = float(bounds[0]), float(bounds[1])
        if lower <= 0:
            raise ValueError(f"temperature lower bound must be > 0, got {lower}.")
        with torch.no_grad():
            self.log_temperature.clamp_(
                min=float(np.log(lower)), max=float(np.log(upper))
            )
        return self


def fit_temperature(logits, targets, **kwargs) -> float:
    """Convenience wrapper around :class:`TemperatureScaler` returning a float."""
    return TemperatureScaler().fit(logits, targets, **kwargs).temperature


# ---------------------------------------------------------------------------
# Expected Calibration Error and reliability diagrams
# ---------------------------------------------------------------------------
def reliability_diagram(probabilities, targets, n_bins: int = CALIBRATION_BINS) -> dict:
    """Bin-wise calibration statistics of a softmax distribution.

    ``probabilities`` is ``[N, C]`` and ``targets`` the matching class indices.
    Confidence is the top-1 probability, accuracy the top-1 correctness. Returns
    the per-bin average confidence/accuracy (0.0 for empty bins) along with the
    ECE, the maximum calibration error (MCE) and the bin population.
    """
    probabilities = as_numpy(probabilities)
    targets = as_numpy(targets).astype(np.int64)
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be 2-D [N, C], got {probabilities.shape}.")
    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError(
            f"probabilities and targets disagree on N: "
            f"{probabilities.shape[0]} vs {targets.shape[0]}."
        )
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")

    confidences = np.max(probabilities, axis=-1)
    correct = (np.argmax(probabilities, axis=-1) == targets).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Confidence 1.0 belongs to the last bin, hence the clip on the digitized index.
    indices = np.clip(np.digitize(confidences, edges[1:-1]), 0, n_bins - 1)

    counts = np.zeros(n_bins, dtype=np.int64)
    bin_confidences = np.zeros(n_bins, dtype=np.float64)
    bin_accuracies = np.zeros(n_bins, dtype=np.float64)
    total = confidences.shape[0]
    ece = 0.0
    mce = 0.0
    for bin_index in range(n_bins):
        mask = indices == bin_index
        counts[bin_index] = int(np.count_nonzero(mask))
        if counts[bin_index] == 0:
            continue
        bin_confidences[bin_index] = float(np.mean(confidences[mask]))
        bin_accuracies[bin_index] = float(np.mean(correct[mask]))
        gap = abs(bin_accuracies[bin_index] - bin_confidences[bin_index])
        weight = counts[bin_index] / float(total)
        ece += weight * gap
        mce = max(mce, gap)

    return {
        "n_bins": n_bins,
        "n_samples": int(total),
        "ece": float(ece),
        "mce": float(mce),
        "bin_edges": edges,
        "bin_counts": counts,
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_gaps": bin_accuracies - bin_confidences,
        "bin_weights": counts / float(total),
    }


def expected_calibration_error(
    probabilities, targets, n_bins: int = CALIBRATION_BINS
) -> float:
    """Expected Calibration Error: ``sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|``."""
    return reliability_diagram(probabilities, targets, n_bins=n_bins)["ece"]


def top_label_confidence(probabilities):
    """Return ``(confidence, index)`` of the top-1 class for each sample."""
    probabilities = as_numpy(probabilities)
    index = np.argmax(probabilities, axis=-1)
    confidence = probabilities[np.arange(probabilities.shape[0]), index]
    return confidence, index


# ---------------------------------------------------------------------------
# Confidence thresholding
# ---------------------------------------------------------------------------
def default_thresholds(step: float = CONFIDENCE_THRESHOLD_STEP) -> np.ndarray:
    """Evenly spaced thresholds covering ``[0.0, 1.0]`` inclusive."""
    step = float(step)
    if not 0.0 < step <= 1.0:
        raise ValueError(f"step must be in (0, 1], got {step}.")
    return np.round(np.arange(0.0, 1.0 + step / 2.0, step), 6)


def threshold_metrics(probabilities, targets, threshold: float) -> dict:
    """Rejection metrics of one confidence threshold ``tau``.

    A sample is *accepted* when its top-1 probability reaches ``tau``:

    * ``coverage`` - share of samples that survive the threshold, i.e. the share
      of clips the system still answers.
    * ``retained_accuracy`` - accuracy measured on the accepted samples only.
    * ``frr`` (false rejection rate) - share of the **correct** predictions that
      are thrown away by the threshold: lost commands.
    * ``far`` (false acceptance rate) - share of the **incorrect** predictions
      that still pass the threshold: wrong answers served as if they were right.
    """
    probabilities = as_numpy(probabilities)
    targets = as_numpy(targets).astype(np.int64)
    threshold = float(threshold)

    confidences, indices = top_label_confidence(probabilities)
    correct = indices == targets
    accepted = confidences >= threshold

    n_total = int(confidences.shape[0])
    n_accepted = int(np.count_nonzero(accepted))
    n_correct = int(np.count_nonzero(correct))
    n_incorrect = n_total - n_correct
    accepted_correct = int(np.count_nonzero(correct & accepted))
    rejected_correct = int(np.count_nonzero(correct & ~accepted))
    accepted_incorrect = int(np.count_nonzero(~correct & accepted))

    return {
        "tau": threshold,
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_rejected": n_total - n_accepted,
        "coverage": (n_accepted / n_total) if n_total else 0.0,
        "retained_accuracy": (accepted_correct / n_accepted) if n_accepted else 0.0,
        "rejection_rate": ((n_total - n_accepted) / n_total) if n_total else 0.0,
        "frr": (rejected_correct / n_correct) if n_correct else 0.0,
        "far": (accepted_incorrect / n_incorrect) if n_incorrect else 0.0,
        "accepted_correct": accepted_correct,
        "accepted_incorrect": accepted_incorrect,
        "rejected_correct": rejected_correct,
        "rejected_incorrect": n_incorrect - accepted_incorrect,
    }


def sweep_thresholds(
    probabilities, targets, thresholds: Optional[Sequence[float]] = None
) -> List[dict]:
    """Evaluate :func:`threshold_metrics` over a grid of thresholds."""
    if thresholds is None:
        thresholds = default_thresholds()
    return [threshold_metrics(probabilities, targets, tau) for tau in thresholds]


def select_threshold(
    sweep: Sequence[dict],
    criterion: str = THRESHOLD_CRITERION,
    min_coverage: float = THRESHOLD_MIN_COVERAGE,
    target_accuracy: Optional[float] = None,
) -> dict:
    """Pick the operating point ``tau*`` from a threshold sweep.

    Criteria:

    * ``"min_cost"`` (default) - minimise ``FRR + FAR``: the balanced operating
      point where a discarded command and a served mistake cost the same.
    * ``"coverage_accuracy"`` - maximise ``coverage x retained accuracy``, i.e.
      the number of clips that are answered *and* right, which is what an actual
      deployment gains.
    * ``"target_accuracy"`` - smallest ``tau`` (largest coverage) whose retained
      accuracy reaches ``target_accuracy``.

    Rows below ``min_coverage`` are discarded first so the chosen threshold
    cannot silence almost the whole system; if no row qualifies, the whole sweep
    is considered. Ties are broken toward the *lower* threshold, i.e. toward
    answering more clips.
    """
    sweep = list(sweep)
    if not sweep:
        raise ValueError("Cannot select a threshold from an empty sweep.")
    if criterion not in THRESHOLD_CRITERIA:
        raise ValueError(
            f"Unknown criterion '{criterion}'. Expected one of {THRESHOLD_CRITERIA}."
        )

    candidates = [row for row in sweep if row["coverage"] >= float(min_coverage)]
    if not candidates:
        candidates = sweep

    if criterion == "target_accuracy":
        target = 0.0 if target_accuracy is None else float(target_accuracy)
        reached = [row for row in candidates if row["retained_accuracy"] >= target]
        pool = reached if reached else candidates
        return dict(min(pool, key=lambda row: (row["tau"], -row["retained_accuracy"])))

    if criterion == "min_cost":
        return dict(
            min(candidates, key=lambda row: (row["frr"] + row["far"], row["tau"]))
        )

    return dict(
        max(
            candidates,
            key=lambda row: (row["coverage"] * row["retained_accuracy"], -row["tau"]),
        )
    )


def apply_confidence_threshold(
    probabilities,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    labels: Optional[Sequence[str]] = None,
    reject_label: Optional[str] = REJECT_LABEL,
) -> dict:
    """Turn one probability row into the deployed decision.

    Returns the top-1 class (``predicted_index`` / ``predicted_label``), its
    calibrated probability (``confidence``), the ``is_low_confidence`` rejection
    flag and the label actually served (``label``): the reject label when
    ``confidence < threshold``, the predicted class otherwise.
    """
    row = as_numpy(probabilities)
    if row.ndim != 1:
        raise ValueError(f"probabilities must be 1-D [C], got {row.shape}.")
    threshold = float(threshold)

    index = int(np.argmax(row))
    confidence = float(row[index])
    is_low_confidence = bool(confidence < threshold)
    predicted_label = labels[index] if labels is not None else index
    served_label = (
        reject_label if (is_low_confidence and reject_label) else predicted_label
    )

    return {
        "predicted_index": index,
        "predicted_label": predicted_label,
        "label": served_label,
        "confidence": confidence,
        "is_low_confidence": is_low_confidence,
        "accepted": not is_low_confidence,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Calibration artifact (deployment metadata)
# ---------------------------------------------------------------------------
def build_calibration_metadata(
    temperature: float,
    confidence_threshold: float,
    *,
    version: str = MODEL_VERSION,
    n_bins: int = CALIBRATION_BINS,
    ece_before: Optional[float] = None,
    ece_after: Optional[float] = None,
    nll_before: Optional[float] = None,
    nll_after: Optional[float] = None,
    nll_before_train: Optional[float] = None,
    threshold_criterion: str = THRESHOLD_CRITERION,
    calibration_split: Optional[str] = None,
    calibration_samples: Optional[int] = None,
    source_checkpoint: Optional[str] = None,
    reject_label: Optional[str] = REJECT_LABEL,
    notes: Optional[str] = None,
) -> dict:
    """Assemble the JSON payload shared by the evaluator, the inferencer and ONNX."""
    return {
        "model_name": MODEL_NAME,
        "model_version": version,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "temperature": round(float(temperature), 6),
        "confidence_threshold": round(float(confidence_threshold), 6),
        "reject_label": reject_label,
        "n_bins": int(n_bins),
        "ece_before": None if ece_before is None else round(float(ece_before), 6),
        "ece_after": None if ece_after is None else round(float(ece_after), 6),
        "nll_before": None if nll_before is None else round(float(nll_before), 6),
        "nll_after": None if nll_after is None else round(float(nll_after), 6),
        "nll_before_train": (
            None if nll_before_train is None else round(float(nll_before_train), 6)
        ),
        "threshold_criterion": threshold_criterion,
        "calibration_split": calibration_split,
        "calibration_samples": (
            None if calibration_samples is None else int(calibration_samples)
        ),
        "source_checkpoint": None if source_checkpoint is None else str(source_checkpoint),
        "notes": notes
        or (
            "Post-hoc temperature scaling and confidence thresholding. The logits "
            "stay raw in the exported ONNX graph: T and tau are applied by the "
            "Python runtime so that the threshold can be changed without re-export."
        ),
    }


def save_calibration(
    metadata: dict,
    path=None,
    version: str = MODEL_VERSION,
) -> Path:
    """Write the calibration metadata as JSON, creating its folder if needed."""
    if path is None:
        path = calibration_path(version)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)
        handle.write("\n")
    return path


def resolve_calibration_file(version: str = MODEL_VERSION) -> Optional[Path]:
    """Return the first existing calibration file for ``version``, if any."""
    for candidate in (
        calibration_path(version),
        Path(checkpoint_dir(version)) / f"{MODEL_NAME}_calibration_{version}.json",
    ):
        if candidate.exists():
            return candidate
    return None


def load_calibration(path=None, version: str = MODEL_VERSION) -> dict:
    """Load the calibration metadata, or ``{}`` when no usable file exists.

    Missing keys are tolerated downstream, so a partially written file cannot
    crash the server: callers fall back to the configured defaults.
    """
    if path is None:
        path = resolve_calibration_file(version)
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def metadata_to_onnx_props(metadata: dict) -> dict:
    """Flatten the calibration metadata into ``key -> string`` ONNX metadata.

    ONNX metadata is a string dictionary, so values are JSON-encoded and
    namespaced under ``commandsense.calibration.``.
    """
    props = {}
    for key in CALIBRATION_METADATA_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        props[f"{ONNX_METADATA_PREFIX}{key}"] = (
            value if isinstance(value, str) else json.dumps(value)
        )
    return props


if __name__ == "__main__":
    # Self-check on synthetic logits: rescaling a logit matrix by a constant c
    # must make the fitted temperature c times larger, because T and the logit
    # scale are reciprocal.
    rng = np.random.default_rng(1234)
    n_samples, n_classes = 4000, 37
    targets = rng.integers(0, n_classes, n_samples)
    logits = rng.normal(size=(n_samples, n_classes))
    logits[np.arange(n_samples), targets] += 3.0

    scale = 2.5
    temperature = fit_temperature(logits * scale, targets)
    expected = scale * fit_temperature(logits, targets)
    relative_error = abs(temperature - expected) / expected

    probabilities_before = logits_to_probabilities(logits * scale, DEFAULT_TEMPERATURE)
    probabilities_after = logits_to_probabilities(logits * scale, temperature)
    ece_before = expected_calibration_error(probabilities_before, targets, n_bins=15)
    ece_after = expected_calibration_error(probabilities_after, targets, n_bins=15)

    print("--- CommandSense calibration self-check ---")
    print(f"Input shape        : {logits.shape} (37 classes)")
    print(f"Fitted temperature : {temperature:.4f} (expected ~{expected:.4f})")
    print(f"ECE before         : {ece_before:.4f}")
    print(f"ECE after          : {ece_after:.4f}")
    assert relative_error < 0.05, "temperature scaling did not recover the logit scale"
    assert ece_after <= ece_before, "calibration must not degrade the ECE"

    labels = [f"class_{index}" for index in range(n_classes)]
    sweep = sweep_thresholds(probabilities_after, targets)
    best = select_threshold(sweep)
    decision = apply_confidence_threshold(
        probabilities_after[0], threshold=0.90, labels=labels
    )
    print(
        f"tau* (coverage x acc): {best['tau']:.2f} | coverage {best['coverage']:.4f} | "
        f"retained acc {best['retained_accuracy']:.4f} | FRR {best['frr']:.4f} | "
        f"FAR {best['far']:.4f}"
    )
    print(
        f"Single decision    : label={decision['label']} "
        f"confidence={decision['confidence']:.4f} "
        f"low_confidence={decision['is_low_confidence']}"
    )
    print("[+] Calibration self-check passed.")
