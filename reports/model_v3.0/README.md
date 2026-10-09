# CommandSense v3.0 - Model Card

<p align="center">
  <img src="https://img.shields.io/badge/Model%20Version-v3.0.0-7c3aed?style=for-the-badge" alt="Model Version">
  <img src="https://img.shields.io/badge/Val%20Acc-96.36%25-green?style=for-the-badge" alt="Validation Accuracy">
  <img src="https://img.shields.io/badge/Test%20Acc%20(35%20cmd)-95.87%25-green?style=for-the-badge" alt="Test Accuracy">
  <img src="https://img.shields.io/badge/Temperature-1.3116-orange?style=for-the-badge" alt="Fitted Temperature">
  <img src="https://img.shields.io/badge/Threshold%20(tau)-0.89-blue?style=for-the-badge" alt="Confidence Threshold">
  <img src="https://img.shields.io/badge/Classes-37-blue?style=for-the-badge" alt="37 Classes">
</p>

---

## ★ Model Overview

* **Model Name**: CommandSense v3.0
* **Architecture**: Residual CNN with three residual stages (`in_conv` + `layer1..3` + `AdaptiveAvgPool2d` + `Dropout(0.3)` + `Linear`), fed by a three-channel Log-Mel front-end. **1,217,349 parameters** (≈4.87 MB in float32).
* **Weights**: trained from scratch for **30 epochs** on the official speaker-disjoint splits. No pre-trained checkpoint is loaded and no weight is reused.
* **Front-end**: `AudioToThreeChannelMel` (`src/transforms.py`) builds the Log-Mel + Δ + Δ² tensor **on the accelerator**, on the fly, from the raw `int16` waveforms that the dataset yields.
* **Task**: Multi-class Keyword Spotting (KWS) & Spoken Command Recognition, with explicit rejection of silence, out-of-vocabulary speech **and** low-confidence predictions.
* **Target Classes (37)**: `_silence_`, `_unknown_`, and the 35 command words `backward`, `bed`, `bird`, `cat`, `dog`, `down`, `eight`, `five`, `follow`, `forward`, `four`, `go`, `happy`, `house`, `learn`, `left`, `marvin`, `nine`, `no`, `off`, `on`, `one`, `right`, `seven`, `sheila`, `six`, `stop`, `three`, `tree`, `two`, `up`, `visual`, `wow`, `yes`, `zero`.
* **Artifacts**: the v3.0 release is a self-contained bundle — `CommandSense_v3.0.pth` (trained weights), `CommandSense_v3.0.onnx` (the same model exported for ONNX Runtime, with `T` and `tau` in its `metadata_props`) and `CommandSense_calibration_v3.0.json` (the fitted calibration, kept in `configs/calibration/` because it is versioned configuration rather than a weight).
* **Scope**: this card documents the v3.0 weights and the calibrated deployment layer fitted on top of them — the `int16` RAM cache, the GPU-side feature extraction, the GPU-native conditional augmentation and the 30-epoch training schedule that produce them, and the temperature `T` and threshold `tau` that serve them. It reports v3.0 in isolation and makes no claim about any other release.

---

## ★ Technical Innovations

v3.0 keeps the residual network and rebuilds the data pipeline around it. Four mechanisms define the release, and none of them adds a parameter.

| Mechanism | Implementation | Effect |
| :--- | :--- | :--- |
| **`int16` RAM cache** | `src/dataset.py` caches the raw waveform as `torch.int16` — 16,000 samples × 2 bytes ≈ 32 KB per clip — instead of pre-computed float32 spectrograms. | The resident dataset is ≈75 % smaller, so the corpus fits in RAM alongside the model and the batches. |
| **GPU-side feature extraction** | `AudioToThreeChannelMel` (`src/transforms.py`) builds Log-Mel + Δ + Δ² on the accelerator, on the fly, from the `int16` tensors; the CPU never materialises a spectrogram. | The front-end stays inside the per-batch budget (≈25–30 s per epoch on a single T4 at batch size 256) and cannot drift between training, evaluation and serving. |
| **GPU-native conditional augmentation** | Spectral shift (STFT-domain `stft → torch.roll → istft`), time shift (`torch.roll`) and dynamic SpecAugment (independent per-clip frequency and time masks, re-drawn every epoch) run per batch, in `train()` mode only, and are **strictly disabled for `_silence_`**. | The spectral shift is ≈180× cheaper than the phase vocoder it replaces, and excluding `_silence_` keeps the reject boundary sharp instead of blurring it. |
| **Resilient provisioning** | `ensure_speech_commands` / `ensure_librispeech` centralise and idempotently retry the downloads, with a TLS fallback when a CDN chain fails local verification (`CERTIFICATE_VERIFY_FAILED`). | An unattended install completes without a manual workaround, while genuine network errors are still re-raised and stay visible. |

**These mechanisms act as a bundle.** They are introduced together in a single release, and every metric on this card is the outcome of the combination. Each mechanism is documented separately above, but no per-mechanism accuracy contribution is claimed: isolating any one of them would require a dedicated training run per variant, which was not done.

---

## ★ Training Recipe & History

v3.0 is trained from scratch for **30 epochs** on the official speaker-disjoint splits. `train.py` caches the corpus as `int16` and builds spectrograms on the GPU, so an epoch costs ≈25–30 s on a single T4.

| Hyperparameter | Value |
| :--- | :--- |
| Objective | Cross-entropy over 37 classes |
| Optimiser | AdamW |
| Learning rate | `1e-3`, constant (no scheduler) |
| Weight decay | `1e-4` |
| Batch size | 256 |
| Epochs | 30 |
| Seed | `42` (global, set by `set_seed`) |
| Mixed precision | AMP (`torch.amp.autocast` + `GradScaler`) |
| Hardware | 1× NVIDIA T4, ≈25–30 s per epoch |

![Training Curves](training_curves_v3.0.png)

| Epoch | Train loss | Train acc | Val loss | Val acc |
| :---: | ---: | ---: | ---: | ---: |
| 1 | 1.728152 | 53.56 % | 0.644610 | 81.18 % |
| 10 | — | — | — | 95.14 % |
| 20 | — | — | — | 95.97 % |
| 30 | 0.111208 | 96.53 % | 0.149190 | 96.12 % |

The best checkpoint of the run is the last one: validation accuracy peaks at **96.3567 %** on epoch 29 (96.1199 % on epoch 30), and validation loss is lowest on the same epoch (0.141437). The curve is still improving slowly at epoch 30 — the last ten epochs buy ≈0.38 points of validation accuracy — so the 30-epoch budget is mildly under-trained rather than over-trained, and the train/val gap stays small (96.53 % vs 96.12 % for accuracy, 0.111 vs 0.149 for loss).

**Note on the two validation numbers.** The history JSON records 96.3567 % (best epoch, training-time validation loader) while the calibration report measures **96.3749 %** on the same 10,979-clip validation split. The two passes differ because the evaluation loader builds its inputs through the dedicated eval path in `src/transforms.py` and re-draws the synthetic `_silence_` / `_unknown_` clips at load time. The headline, **96.36 %**, is the calibration report figure.

---

## ★ Evaluation Results

All numbers below are on the official speaker-disjoint splits, with **raw argmax** decisions unless a threshold is named. The `T`-scaled and `T = 1` predictions are identical by construction (see [Calibration](#-calibration)).

| Metric | Split | Clips | Value |
| :--- | :--- | ---: | ---: |
| Overall accuracy (37 classes) | testing | 12,105 | **96.1751 %** |
| Top-1 on the 35 commands | testing | 11,005 | **95.8655 %** (10,550) |
| Top-1 (37 classes) | validation | 10,979 | 96.3749 % |
| Macro F1, 35 command classes | testing | 11,005 | **0.9541** |
| Macro recall, 35 command classes | testing | 11,005 | 0.9542 |
| Macro F1, 37 classes | testing | 12,105 | 0.9560 |
| Recall `_silence_` | testing | 550 | **100.0000 %** (550) |
| Recall `_unknown_` | testing | 550 | **98.5455 %** (542) |

![Confusion Matrix](confusion_matrix_v3.0.png)

The per-class report in [`report_v3.0.txt`](report_v3.0.txt) shows the residual error is concentrated in a handful of commands rather than spread evenly: `six` (F1 0.9873), `stop` (0.9854) and `yes` (0.9856) are essentially solved, while `tree` (0.8855), `learn` (0.8896) and `bed` (0.9059) are the weakest three. `bed` and `tree` are recall-limited (0.8599 and 0.9016): the model misses those words more often than it over-predicts them, which is the expected cost of a rejection-heavy training set.

Two caveats on reading this table:

* **The rejection-class recalls are samples, not constants.** The 1,100 `_silence_` / `_unknown_` evaluation clips are not stored on disk — `src/dataset.py` slices them out of the held-out `_background_noise_` recordings and out of LibriSpeech `dev-clean` at load time under the global RNG. Re-running the evaluation with a different seed moves those two cells. The command columns are unaffected.
* **`Overall` mixes two populations.** It is computed over the command clips plus the rejection clips, so it responds to the composition of the split as much as to raw accuracy. For a pure command metric read the 35-command Top-1; for the reject behaviour read the two per-class recalls.

---


## ★ Calibration

v3.0 ships a fitted post-hoc calibration: a single scalar temperature `T` fitted on held-out validation data, plus a confidence threshold `tau` that turns the calibrated probability into an accept/reject decision. Both values are produced by `evaluate.py` and stored in `configs/calibration/CommandSense_calibration_v3.0.json`, which the inferencer loads at start-up.

**Temperature scaling** (Guo et al., 2017) rescales the logits `z` with a single scalar `T > 0`:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

`T` is fitted on the **validation split** (10,979 clips) by minimising the negative log-likelihood with LBFGS (50 evaluations per step, 12 steps, `tol = 1e-9`, clamped to `[0.05, 20.0]`). Two diagnostics are reported with `M = 15` equal-width probability bins: **NLL** (the quantity being minimised), **ECE** (expected calibration error, the bin-weighted gap between mean confidence and empirical accuracy) and **MCE** (maximum calibration error, the worst single bin).

| Split | Samples | Top-1 accuracy (unchanged by `T`) | Metric | Before (`T = 1`) | After (`T = 1.3116`) | Change |
| :--- | ---: | :---: | :--- | ---: | ---: | :---: |
| Validation (fit) | 10,979 | 96.3749 % | NLL | 0.139316 | 0.132453 | −4.9 % |
| | | | ECE | 0.009217 | **0.006232** | **−32.4 %** |
| | | | MCE | 0.168483 | 0.138063 | −18.1 % |
| Testing (out of sample) | 12,105 | 96.1751 % | NLL | 0.132331 | 0.127490 | −3.7 % |
| | | | ECE | 0.010317 | **0.005332** | **−48.3 %** |
| | | | MCE | 0.192082 | 0.125716 | −34.5 % |

**`T = 1.3116` is a mild correction, and that is the interesting part.** Before any fitting, the raw logits already score `ECE = 0.0103` on the out-of-sample split — a small miscalibration to begin with — so the fitted scalar stays close to `1` and has limited head-room to exploit. `T > 1` still **softens** the distribution, which is the direction an over-confident network needs, so the reported confidence moves closer to the empirical accuracy while not a single prediction changes.

### - Reliability Diagram

![Reliability Diagram](calibration_v3.0.png)

Before calibration the reliability curve already sits close to the diagonal, and after dividing the logits by `T = 1.3116` the residual gap narrows further, with the maximum calibration error falling from 0.192 to 0.126 on the out-of-sample split. The test ECE falls by **48.3 %** even though `T` was fitted on a different split: with ≈11 k clips and one free parameter there is essentially nothing to overfit, which is why the out-of-sample improvement tracks the in-sample one so closely.

The practical consequence is modest but real: at 90 % confidence the model is now right about 90 % of the time, so the calibrated number can be shown to a user as-is, and the same number is what the confidence threshold operates on.

---


## ★ Confidence Threshold Decision

A clip is answered only when its calibrated top-1 probability reaches `tau`; otherwise it is served as `_unknown_`, the project's existing reject class.

$$\hat{y} = \begin{cases} \arg\max_i p_i, & \max_i p_i \ge \tau \\ \texttt{\_unknown\_}, & \text{otherwise} \end{cases}$$

`tau` is swept over `[0, 1]` in steps of `0.01` on the validation split, and the selection criterion is `min_cost`, which minimises `FRR + FAR` subject to the coverage floor `THRESHOLD_MIN_COVERAGE = 90 %`. The selected value is **`tau* = 0.89`** — a deliberately strict operating point: the model is accurate and well calibrated, so a high bar can be demanded before an answer is served.

| Metric (testing split, 12,105 clips) | Value |
| :--- | ---: |
| Coverage (clips answered) | **89.7646 %** (10,866 / 12,105) |
| Retained accuracy (answered clips only) | **99.4754 %** (10,809 / 10,866) |
| False rejection rate (correct predictions discarded) | 7.1551 % of correct predictions |
| False acceptance rate (incorrect predictions served) | 12.3110 % of incorrect predictions |
| No threshold at all (raw argmax) | 96.1751 % of all clips correct |
| End-to-end yield (answered **and** correct) | 89.2937 % of all clips |
| Rejected as uncertain | 10.2354 % (served as `_unknown_`) |

Reading the numbers in counts, which is more honest than reading the percentages:

| | Correct | Incorrect | Total |
| :--- | ---: | ---: | ---: |
| Above `tau = 0.89` (served) | 10,809 | 57 | 10,866 |
| Below `tau = 0.89` (rejected) | 833 | 406 | 1,239 |
| **Total** | **11,642** | **463** | **12,105** |

What the threshold buys is **precision, not accuracy**: the answers given at `tau*` are correct 99.48 % of the time instead of 96.18 % (+3.30 points), and **406 of the split's 463 errors** are suppressed before they reach a caller. What it costs is **coverage**: 833 correct clips (6.88 points of the full split) are discarded, so the end-to-end yield of correct and answered clips is 89.29 %, **below** the 96.18 % the threshold-free model scores on every clip. The 57 wrong predictions that stay above `tau` are still served with no warning — a confidence threshold reduces errors, it does not eliminate them.

### - Threshold Sweep

Every row is measured on the testing split with the calibrated probabilities, so the sweep is a fair out-of-sample view of the trade-off. `tau*` is the validation-selected operating point.

| `tau` | Coverage | Retained acc | FRR | FAR |
| :---: | ---: | ---: | ---: | ---: |
| 0.00 | 100.00 % | 96.18 % | 0.00 % | 100.00 % |
| 0.10 | 99.74 % | 96.41 % | 0.02 % | 93.52 % |
| 0.20 | 99.45 % | 96.68 % | 0.03 % | 86.39 % |
| 0.30 | 99.18 % | 96.88 % | 0.09 % | 80.99 % |
| 0.40 | 98.73 % | 97.13 % | 0.29 % | 74.08 % |
| 0.50 | 98.08 % | 97.50 % | 0.57 % | 64.15 % |
| 0.60 | 96.67 % | 98.12 % | 1.37 % | 47.52 % |
| 0.70 | 95.22 % | 98.70 % | 2.28 % | 32.40 % |
| 0.80 | 93.14 % | 99.11 % | 4.02 % | 21.60 % |
| 0.90 | 89.26 % | **99.55 %** | 7.61 % | **10.58 %** |
| **0.89** | **89.76 %** | **99.48 %** | **7.16 %** | **12.31 %** ← `tau*` |

Below `tau = 0.70` the threshold removes very few errors: coverage stays above 95 % but the FAR is still above 30 %, so most of the remaining mistakes are served anyway. The useful range is `0.80–0.90`, where the FAR collapses from 21.60 % to 10.58 % for ≈3.9 points of coverage (93.14 % → 89.26 %). The `0.90` row is very slightly more accurate on retained accuracy (99.55 %) but sits just below the 90 % coverage floor, which is exactly why `min_cost` lands on `0.89`.

### - Coverage vs Accuracy

![Coverage vs Accuracy](coverage_vs_accuracy_v3.0.png)

Below `tau ≈ 0.5` almost nothing is rejected (coverage 98–100 %) while most of the remaining errors are still served (FAR 64–100 %); above `tau ≈ 0.9` the threshold starts discarding correct answers (FRR 7.61 %) faster than it removes errors. The useful range is narrow — retained accuracy only moves from 99.11 % to 99.55 % between `tau = 0.80` and `tau = 0.90` — which is the flip side of the better calibrated base: there is simply less error left for a threshold to remove.

### - Interaction with the Learned Rejection Classes

The two rejection mechanisms are complementary rather than redundant:

* The learned classes `_silence_` / `_unknown_` cover the **two known non-command populations** and are decided by the argmax. They are measured on clips drawn from the same distributions the model was trained on, and v3.0 scores 100.00 % / 98.55 % on them.
* The threshold catches **everything else the model is unsure about** — including real command clips it cannot recognise, which is precisely the failure mode a learned rejection class cannot fix.

Both are reported in the response, so a caller can tell them apart: `raw_label` always carries the raw argmax and `is_low_confidence` flags the threshold rejection.

---


## ★ Deployment

### - Loading the Artifacts

Nothing has to be wired by hand: `CommandInferencer` (used by `app.py`) resolves its own paths at start-up, in this order.

| What | Resolution order |
| :--- | :--- |
| Weights | `checkpoints/model_v3.0/CommandSense_v3.0.pth` |
| ONNX | `checkpoints/model_v3.0/CommandSense_v3.0.onnx` → `onnx/` |
| `T` / `tau` | `configs/calibration/CommandSense_calibration_v3.0.json` → `checkpoints/model_v3.0/*_calibration_v3.0.json` (legacy) |

A startup log line names the file that was actually read, and every response carries the `temperature` and `threshold` in force, so a deployment can always prove which calibration it is serving.

**Fallback.** If the calibration file is absent, unreadable, malformed or holds values the runtime cannot apply (`T <= 0`, `tau` outside `[0, 1]`), the inferencer prints a warning and falls back on `T = 1.0` / `tau = 0.0`. That is the raw-logit decision rule with no low-confidence rejection. **Model weights are different:** the inferencer only loads assets for the configured version (`MODEL_VERSION = "v3.0"`). If neither the matching `.pth` nor `.onnx` file exists, initialization raises an error explaining where to place the versioned release assets; it does not substitute weights from another version.

The exported ONNX graph keeps emitting raw logits and stores `T` and `tau` in its `metadata_props` under `commandsense.calibration.*`, so the operating point can be changed without re-exporting the model. `src/inference.py` applies `softmax(logits / T)` in Python, identically for the ONNX and PyTorch backends, and `is_low_confidence` distinguishes a threshold rejection from the learned `_unknown_` class.

### - Files in this folder

| File | Contents |
| :--- | :--- |
| `report_v3.0.txt` | Full evaluation and calibration report: per-class table, sweep, before/after metrics |
| `history_v3.0.json` | 30-epoch training history (train/val loss and accuracy) |
| `training_curves_v3.0.png` | Loss and accuracy curves for the run above |
| `confusion_matrix_v3.0.png` | 37×37 confusion matrix on the testing split |
| `calibration_v3.0.png` | Reliability diagram before and after temperature scaling |
| `coverage_vs_accuracy_v3.0.png` | Coverage and retained accuracy across the `tau` sweep |

Reproduce them with:
```text
python train.py      # trains v3.0 for 30 epochs
python evaluate.py   # fits T and tau -> configs/calibration/ + this folder
```

`evaluate.py` seeds the RNG (`SEED = 42`), so a verification pass reproduces the fitted `T = 1.311633` and `tau* = 0.89` exactly.

---

## ★ Caveats

* **The mechanisms act as a bundle.** The RAM representation, the front-end placement, the augmentation strategy, the provisioning fallback and the epoch budget are all part of one release. The published metrics are the outcome of the combination; this card does not attribute them to any single mechanism.
* **The improvement is a pipeline improvement, not a capacity improvement.** The architecture is a plain 1,217,349-parameter residual CNN: no layer was added, removed or resized. What changed is how the data reaches the model.
* **The rejection clips are drawn at load time.** The 10,979-clip validation split and the 12,105-clip testing split (11,005 commands + 1,100 rejection clips) each contain 1,100 synthetic `_silence_` / `_unknown_` clips that `src/dataset.py` slices out of the held-out recordings rather than storing on disk. The two rejection-class recalls are therefore samples rather than constants: re-running the evaluation with a different seed moves those cells, while the 35 command columns do not move.
* **The `Val Acc` column vs the calibration report.** The training history records 96.3567 % at its best epoch; the calibration report measures 96.3749 % on the same split through the evaluation path. The headline **96.36 %** follows the report.
* **The threshold trades coverage for precision, and it loses on net yield.** At `tau* = 0.89` the answers served are 99.48 % correct, but the end-to-end yield of *answered and correct* clips (89.29 %) is below the 96.18 % the threshold-free model scores on every clip. Choose `tau` from the sweep table, not from the headline accuracy.
* **This card describes v3.0 only.** It is written to be read on its own and makes no claim about any other release: no other version appears in it, and no metric is stated as a delta against one. Comparisons across versions live in the repository's root `README.md`, where every version sits in a single table.

