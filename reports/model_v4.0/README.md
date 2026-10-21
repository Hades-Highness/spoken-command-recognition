# CommandSense v4.0 - Model Card

<p align="center">
  <img src="https://img.shields.io/badge/Model%20Version-v4.0.0-7c3aed?style=for-the-badge" alt="Model Version">
  <img src="https://img.shields.io/badge/PyTorch-Supported-EE4C2C?style=for-the-badge" alt="PyTorch">
  <img src="https://img.shields.io/badge/INT8%20ONNX-SUPPORTED-green?style=for-the-badge" alt="INT8 ONNX">
  <img src="https://img.shields.io/badge/FP32%20ONNX-SUPPORTED-005ced?style=for-the-badge" alt="FP32 ONNX">
  <img src="https://img.shields.io/badge/Classes-37-blue?style=for-the-badge" alt="37 Classes">
</p>

---

## ★ Model Overview

- **Model Name**: CommandSense v4.0
- **Architecture**: Residual CNN (3 residual stages) with a three-channel Log-Mel front-end. **1,217,349 parameters** (≈4.87 MB in float32).
- **Weights**: trained from scratch for **30 epochs** on the official speaker-disjoint splits with SNR-controlled noise mixing.
- **Front-end**: `AudioToThreeChannelMel` (`src/transforms.py`) builds the Log-Mel + Δ + Δ² tensor **on the accelerator**, on the fly, from the raw `int16` waveforms.
- **Task**: Multi-class Keyword Spotting (KWS) & Spoken Command Recognition, with explicit rejection of silence, out-of-vocabulary speech **and** low-confidence predictions. Training includes controlled ambient-noise and background-speech mixing. A held-out noise evaluation found substantially higher command accuracy than v3.0 at the tested SNRs; see the results and protocol below.
- **Target Classes (37)**: `_silence_`, `_unknown_`, and the 35 command words `backward`, `bed`, `bird`, `cat`, `dog`, `down`, `eight`, `five`, `follow`, `forward`, `four`, `go`, `happy`, `house`, `learn`, `left`, `marvin`, `nine`, `no`, `off`, `on`, `one`, `right`, `seven`, `sheila`, `six`, `stop`, `three`, `tree`, `two`, `up`, `visual`, `wow`, `yes`, `zero`.
- **Artifacts**: the v4.0 release provides `CommandSense_v4.0.pth` (trained weights), `CommandSense_v4.0.onnx` (exported for ONNX Runtime, with `T` and `tau` in its `metadata_props`) and `CommandSense_calibration_v4.0.json` (the fitted calibration, kept in `configs/calibration/` because it is versioned configuration rather than a weight).
- **Scope**: this card documents the v4.0 weights and calibrated deployment layer, including the SNR-controlled noise mixing, `int16` RAM cache, GPU-side feature extraction, conditional augmentation and 30-epoch training schedule. It also reports a held-out noisy-condition comparison with v3.0; other version comparisons are outside its scope.

---

## ★ Technical Innovations

v4.0 keeps the residual network and adds **SNR-controlled noise mixing** to the training pipeline. In the held-out evaluation, v4.0 outperforms v3.0 on the same noisy command clips at all four tested SNRs (5–20 dB). This result is specific to the noise sources and mixing protocol described below.

| Mechanism | Implementation | Effect |
| :-- | :-- | :-- |
| **SNR-controlled noise mixing** | 50% of eligible training clips are mixed with noise sources at controlled signal-to-noise ratios. Ambient noise (`_background_noise_`) is mixed at 5–15 dB SNR, LibriSpeech speech at 15–25 dB SNR. Mixing happens on the GPU at the waveform stage, before feature extraction, and never touches the cached int16 clips. | Exposes the model to command audio with ambient noise and background speech. On the held-out benchmark below, this training version scores higher than v3.0 at each tested noise level. |
| **`_silence_` exclusion** | `_silence_` clips are **never** mixed with additional noise. They remain pure background noise. | The reject boundary stays sharp. `_silence_` teaches the model what "no command present" sounds like, and adding noise would blur that boundary. |
| **Noise banks in RAM** | 256 ambient noise clips and 512 LibriSpeech clips are cached as int16 in RAM and promoted to GPU buffers once before training. | Mixing is fully GPU-resident, no disk I/O during training, no CPU overhead. |
| **`int16` RAM cache** | `src/dataset.py` caches the raw waveform as `torch.int16` — 16,000 samples × 2 bytes ≈ 32 KB per clip — instead of pre-computed float32 spectrograms. | The resident dataset is ≈75% smaller, so the corpus fits in RAM alongside the model and the batches. |
| **GPU-side feature extraction** | `AudioToThreeChannelMel` (`src/transforms.py`) builds Log-Mel + Δ + Δ² on the accelerator, on the fly, from the `int16` tensors; the CPU never materialises a spectrogram. | The front-end stays inside the per-batch budget (≈25–30 s per epoch on a single T4 at batch size 256) and cannot drift between training, evaluation and serving. |
| **GPU-native conditional augmentation** | Spectral shift (STFT-domain `stft → torch.roll → istft`), time shift (`torch.roll`) and dynamic SpecAugment (independent per-clip frequency and time masks, re-drawn every epoch) run per batch, in `train()` mode only, and are **strictly disabled for `_silence_`**. | The spectral shift is ≈180× cheaper than the phase vocoder it replaces, and excluding `_silence_` keeps the reject boundary sharp instead of blurring it. |

**These mechanisms act as a bundle.** They are introduced together in a single release, and every metric on this card is the outcome of the combination. Each mechanism is documented separately above, but no per-mechanism accuracy contribution is claimed: isolating any one of them would require a dedicated training run per variant, which was not done.

---

## ★ Training Recipe & History

v4.0 is trained from scratch for **30 epochs** on the official speaker-disjoint splits. `train.py` caches the corpus as `int16` and builds spectrograms on the GPU, so an epoch costs ≈25–30 s on a single T4.

| Hyperparameter | Value |
| :-- | :-- |
| Objective | Cross-entropy over 37 classes |
| Optimiser | AdamW |
| Learning rate | `1e-3`, constant (no scheduler) |
| Weight decay | `1e-4` |
| Batch size | 256 |
| Epochs | 30 |
| Seed | `42` (global, set by `set_seed`) |
| Mixed precision | AMP (`torch.amp.autocast` + `GradScaler`) |
| Hardware | 1× NVIDIA T4, ≈25–30 s per epoch |
| Noise mixing probability | 50% of eligible training clips |
| Ambient noise SNR | 5–15 dB |
| Speech noise SNR | 15–25 dB |
| Speech noise source share | 50% LibriSpeech, 50% ambient |

| Epoch | Train loss | Train acc | Val loss | Val acc |
| :-: | --: | --: | --: | --: |
| 1 | 2.1555 | 41.94% | 0.6988 | 80.17% |
| 5 | 0.4236 | 87.60% | 0.2307 | 93.41% |
| 10 | 0.3085 | 90.98% | 0.1730 | 95.45% |
| 15 | 0.2619 | 92.13% | 0.1807 | 95.84% |
| 20 | 0.2314 | 93.17% | 0.1846 | 95.22% |
| 25 | 0.2070 | 93.84% | 0.1637 | 95.97% |
| 30 | 0.1909 | 94.21% | 0.1599 | 96.14% |

The highest validation accuracy in the recorded history is **96.4204%** at epoch 26; validation loss is also lowest on epoch 26 (0.1339). The epoch-30 validation accuracy is 96.1381%. `train.py` saves the checkpoint when validation accuracy improves, so the best-accuracy checkpoint is from epoch 26, not epoch 30. Validation accuracy fluctuates after epoch 20, so these results alone do not establish that the 30-epoch schedule is under- or over-trained.

**Note on the two validation numbers.** The history JSON records 96.4204% (best epoch, training-time validation loader) while the calibration report measures **96.4113%** on the same 10,979-clip validation split. The two passes differ because the evaluation loader builds its inputs through the dedicated eval path in `src/transforms.py` and re-draws the synthetic `_silence_` / `_unknown_` clips at load time. The headline, **96.41%**, is the calibration report figure.

---

## ★ Evaluation Results

All numbers below are on the official speaker-disjoint splits, with **raw argmax** decisions unless a threshold is named. The `T`-scaled and `T = 1` predictions are identical by construction (see Calibration).

| Metric | Split | Clips | Value |
| :-- | :-- | --: | --: |
| Overall accuracy (37 classes) | testing | 12,105 | **95.9025%** |
| Top-1 on the 35 commands | testing | 11,005 | **95.6111%** (10,522) |
| Top-1 (37 classes) | validation | 10,979 | 96.4113% |
| Macro F1, 35 command classes | testing | 11,005 | **0.9533** |
| Macro recall, 35 command classes | testing | 11,005 | 0.9511 |
| Macro F1, 37 classes | testing | 12,105 | 0.9547 |
| Recall `_silence_` | testing | 550 | **100.0000%** (550) |
| Recall `_unknown_` | testing | 550 | **97.6364%** (537) |

The per-class report in `report_v4.0.txt` shows the residual error is concentrated in a handful of commands rather than spread evenly: `six` (F1 0.9724), `stop` (0.9817) and `yes` (0.9832) are the highest-scoring three, while `forward` (0.8767), `follow` (0.9046) and `tree` (0.9130) are the lowest-scoring three. `forward` and `tree` are recall-limited (0.8258 and 0.8705): the model misses those words more often than it over-predicts them.

**Comparison with v3.0:**
- Top-1 on 35 commands: **95.61%** (v4.0) vs 95.87% (v3.0) — a **0.26 pt decrease**
- `_silence_` recall: **100.00%** (unchanged)
- `_unknown_` recall: **97.64%** (v4.0) vs 98.55% (v3.0) — a **0.91 pt decrease**
- Validation accuracy: **96.41%** (v4.0) vs 96.36% (v3.0) — a **0.05 pt increase**

The clean-test command accuracy and `_unknown_` recall are lower than the corresponding v3.0 figures, while validation accuracy is slightly higher. The separate held-out noise benchmark shows a substantial command-accuracy improvement over v3.0 for its tested mixtures. These version-level results do not isolate noise mixing from other training-run variation.

---

## ★ Noise Robustness Evaluation

`robustness_report_v4.0.txt` re-scores all **11,005 command clips** from the official, speaker-disjoint `testing` split at four fixed SNR levels. It compares v4.0 with v3.0 using the **same clean command clips and the same noise mixture** for each condition. Top-1 is raw argmax over the 35 command classes and is independent of the confidence threshold.

| SNR | v4.0 Top-1 | v3.0 Top-1 | Difference | v4.0 rejection rate | v3.0 rejection rate |
| :-: | --: | --: | --: | --: | --: |
| 5 dB | 75.24% | 50.96% | **+24.28 pp** | 41.73% | 29.48% |
| 10 dB | 88.45% | 63.54% | **+24.91 pp** | 28.97% | 26.67% |
| 15 dB | 92.69% | 74.27% | **+18.43 pp** | 19.97% | 22.08% |
| 20 dB | 93.94% | 84.10% | **+9.84 pp** | 15.86% | 19.04% |
| **Mean, 5–20 dB** | **87.58%** | **68.22%** | **+19.36 pp** | — | — |
| Clean | 95.61% | 95.87% | **−0.25 pp** | 11.63% | 9.16% |

The mean is the unweighted average across the four tested SNR values. On these fixed held-out mixtures, v4.0 improves raw command accuracy by **9.84–24.91 percentage points** over v3.0. At 5 dB both models remain substantially affected by noise: v4.0 reaches 75.24% Top-1, compared with 50.96% for v3.0.

### - Served operating point

The following v4.0 results apply its shipped calibration (`T = 1.252585`, `tau = 0.92`) to command clips only. Rejection rate is the share of command clips below `tau` and served as `_unknown_`; it is not the learned `_silence_` / `_unknown_` class recall. Accepted accuracy is conditional on clips that pass the threshold, and correct yield is the share of all command clips that are both accepted and correct.

| Condition | Accepted accuracy | Coverage | Correct yield | Mean confidence |
| :-- | --: | --: | --: | --: |
| 5 dB | 90.32% | 58.27% | 52.63% | 84.15% |
| 10 dB | 98.23% | 71.03% | 69.78% | 89.22% |
| 15 dB | 99.28% | 80.03% | 79.45% | 92.54% |
| 20 dB | 99.42% | 84.14% | 83.65% | 93.97% |
| Clean | 99.57% | 88.37% | 87.99% | 95.54% |

The robustness report's clean coverage (88.37%) and rejection rate (11.63%) use only the 11,005 command clips. They therefore differ from the calibration report's 88.95% coverage over all 12,105 command and rejection-class clips.

In the v4.0/v3.0 comparison table, each model's rejection rate uses its own shipped calibration: v4.0 uses `T = 1.252585`, `tau = 0.92`; v3.0 uses `T = 1.311633`, `tau = 0.89`. Top-1 accuracy remains threshold-independent for both.

### - Evaluation protocol and scope

- Noise sources are held out from the v4.0 training noise bank: **121 non-overlapping one-second ambient chunks** from the held-out `_background_noise_` recordings, and **512 one-second LibriSpeech chunks** from `dev-clean`. The evaluated ambient recordings are `running_tap.wav` and `white_noise.wav`; the other four listed background-noise recordings are reserved for training.
- Each command clip is mixed with a randomly selected ambient or speech clip at a 50/50 source probability, at the target SNR shown. The report records seed `42`; the speech noise bank came from `dev-clean`, because `test-clean` was not available in that run.
- The report records one evaluation run. It tests four SNR values and a finite set of held-out sources; results should not be generalized to all environments, noise types, speakers or SNRs. The clean-to-noisy change within v4.0 is descriptive, not a controlled estimate of causal effect.
- Reproduce with `python evaluate_robustness.py` for v4.0 only, or `python evaluate_robustness.py --compare_v3` for the v4.0/v3.0 comparison. See the generated [`robustness_report_v4.0.txt`](robustness_report_v4.0.txt) for the full run metadata and tables.

Two caveats on reading this table:

- **The rejection-class recalls are samples, not constants.** The 1,100 `_silence_` / `_unknown_` evaluation clips are not stored on disk — `src/dataset.py` slices them out of the held-out `_background_noise_` recordings and out of LibriSpeech `dev-clean` at load time under the global RNG. Re-running the evaluation with a different seed moves those two cells. The command columns are unaffected.
- **`Overall` mixes two populations.** It is computed over the command clips plus the rejection clips, so it responds to the composition of the split as much as to raw accuracy. For a pure command metric read the 35-command Top-1; for the reject behaviour read the two per-class recalls.

---

## ★ Calibration

v4.0 ships a fitted post-hoc calibration: a single scalar temperature `T` fitted on held-out validation data, plus a confidence threshold `tau` that turns the calibrated probability into an accept/reject decision. Both values are produced by `evaluate.py` and stored in `configs/calibration/CommandSense_calibration_v4.0.json`, which the inferencer loads at start-up.

**Temperature scaling** (Guo et al., 2017) rescales the logits `z` with a single scalar `T > 0`:

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

`T` is fitted on the **validation split** (10,979 clips) by minimising the negative log-likelihood with LBFGS (50 evaluations per step, 12 steps, `tol = 1e-9`, clamped to `[0.05, 20.0]`). Two diagnostics are reported with `M = 15` equal-width probability bins: **NLL** (the quantity being minimised), **ECE** (expected calibration error, the bin-weighted gap between mean confidence and empirical accuracy) and **MCE** (maximum calibration error, the worst single bin).

| Split | Samples | Top-1 accuracy (unchanged by `T`) | Metric | Before (`T = 1`) | After (`T = 1.2526`) | Change |
| :-- | --: | :-: | :-- | --: | --: | :-: |
| Validation (fit) | 10,979 | 96.4113% | NLL | 0.133312 | 0.128907 | −3.3% |
|  |  |  | ECE | 0.007027 | **0.004150** | **−40.9%** |
|  |  |  | MCE | 0.148710 | 0.116857 | −21.4% |
| Testing (out of sample) | 12,105 | 95.9025% | NLL | 0.145185 | 0.140124 | −3.5% |
|  |  |  | ECE | 0.008544 | **0.003833** | **−55.1%** |
|  |  |  | MCE | 0.189815 | 0.089088 | −53.1% |

**`T = 1.2526` is a mild correction.** Before fitting, the raw logits score `ECE = 0.0085` on the out-of-sample split. The fitted value is greater than `1`, which softens the probability distribution; the reported test ECE also decreases, while the top-1 predictions remain unchanged.

### - Reliability Diagram

Before calibration the reliability curve already sits close to the diagonal, and after dividing the logits by `T = 1.2526` the reported residual gap narrows, with the maximum calibration error falling from 0.190 to 0.089 on the out-of-sample split. The test ECE falls by **55.1%** in this evaluation; this single run and split do not establish that the same improvement will hold for other data or recording conditions.

The practical consequence is that the calibrated probability is the confidence value used by the thresholding rule. Calibration is measured on these evaluation splits and should not be assumed to transfer unchanged to other recording conditions.

---

## ★ Confidence Threshold Decision

A clip is answered only when its calibrated top-1 probability reaches `tau`; otherwise it is served as `_unknown_`, the project's existing reject class.

```
ŷ = argmax_i p_i  if max_i p_i ≥ τ
    _unknown_     otherwise
```

`tau` is swept over `[0, 1]` in steps of `0.01` on the validation split, and the selection criterion is `min_cost`, which minimises `FRR + FAR` subject to the validation coverage floor `THRESHOLD_MIN_COVERAGE = 90%`. The selected value is **`tau* = 0.92`**. On the separate test split, this threshold gives 88.95% coverage; the test coverage is below 90%, which reflects the difference between the validation and test distributions and does not determine the validation-selected threshold.

| Metric (testing split, 12,105 clips) | Value |
| :-- | --: |
| Coverage (clips answered) | **88.9467%** (10,767 / 12,105) |
| Retained accuracy (answered clips only) | **99.5728%** (10,721 / 10,767) |
| False rejection rate (correct predictions discarded) | 7.6492% of correct predictions |
| False acceptance rate (incorrect predictions served) | 9.2742% of incorrect predictions |
| No threshold at all (raw argmax) | 95.9025% of all clips correct |
| End-to-end yield (answered **and** correct) | 88.5667% of all clips |
| Rejected as uncertain | 11.0533% (served as `_unknown_`) |

Reading the numbers in counts, which is more honest than reading the percentages:

|  | Correct | Incorrect | Total |
| :-- | --: | --: | --: |
| Above `tau = 0.92` (served) | 10,721 | 46 | 10,767 |
| Below `tau = 0.92` (rejected) | 879 | 459 | 1,338 |
| **Total** | **11,600** | **505** | **12,105** |

What the threshold buys is **precision, not accuracy**: the answers given at `tau*` are correct 99.57% of the time instead of 95.90% (+3.67 points), and **459 of the split's 505 errors** are suppressed before they reach a caller. What it costs is **coverage**: 879 correct clips (7.26 points of the full split) are discarded, so the end-to-end yield of correct and answered clips is 88.57%, **below** the 95.90% the threshold-free model scores on every clip. The 46 wrong predictions that stay above `tau` are still served with no warning — a confidence threshold reduces errors, it does not eliminate them.

### - Threshold Sweep

Every row is measured on the testing split with the calibrated probabilities, so the sweep is a fair out-of-sample view of the trade-off. `tau*` is the validation-selected operating point.

| `tau` | Coverage | Retained acc | FRR | FAR |
| :-: | --: | --: | --: | --: |
| 0.00 | 100.00% | 95.90% | 0.00% | 100.00% |
| 0.10 | 99.69% | 96.19% | 0.01% | 92.74% |
| 0.20 | 99.36% | 96.47% | 0.05% | 85.69% |
| 0.30 | 98.93% | 96.78% | 0.16% | 77.62% |
| 0.40 | 98.30% | 97.12% | 0.46% | 69.15% |
| 0.50 | 97.50% | 97.51% | 0.87% | 59.27% |
| 0.60 | 96.08% | 98.07% | 1.75% | 45.16% |
| 0.70 | 94.35% | 98.68% | 2.92% | 30.44% |
| 0.80 | 92.58% | 99.05% | 4.38% | 21.37% |
| 0.90 | 89.90% | 99.48% | 6.74% | 11.49% |
| **0.92** | **88.95%** | **99.57%** | **7.65%** | **9.27%** ← `tau*` |

Below `tau = 0.70` the threshold removes very few errors: test coverage stays above 94% but the FAR is still above 30%, so most of the remaining mistakes are served anyway. The useful range is `0.80–0.92`, where the test FAR falls from 21.37% to 9.27% as coverage decreases from 92.58% to 88.95%. The `0.92` row is the validation-selected operating point; its test-split coverage is not the coverage value used by `min_cost`.

### - Coverage vs Accuracy

Below `tau ≈ 0.5` almost nothing is rejected (coverage 97–100%) while most of the remaining errors are still served (FAR 59–100%); above `tau ≈ 0.9` the threshold starts discarding correct answers (FRR 7.65%) faster than it removes errors. The useful range is narrow — retained accuracy only moves from 99.05% to 99.57% between `tau = 0.80` and `tau = 0.92` — illustrating the coverage-versus-error trade-off at these tested thresholds.

### - Interaction with the Learned Rejection Classes

The two rejection mechanisms are complementary rather than redundant:

- The learned classes `_silence_` / `_unknown_` cover the **two known non-command populations** and are decided by the argmax. They are measured on clips drawn from the same distributions the model was trained on, and v4.0 scores 100.00% / 97.64% on them.
- The threshold catches **everything else the model is unsure about** — including real command clips it cannot recognise, which is precisely the failure mode a learned rejection class cannot fix.

Both are reported in the response, so a caller can tell them apart: `raw_label` always carries the raw argmax and `is_low_confidence` flags the threshold rejection.

---

## ★ Deployment

### - Loading the Artifacts

Nothing has to be wired by hand: `CommandInferencer` (used by `app.py`) resolves its own paths at start-up, in this order.

| What | Resolution order |
| :-- | :-- |
| Weights | `checkpoints/model_v4.0/CommandSense_v4.0.pth` |
| ONNX | `checkpoints/model_v4.0/CommandSense_v4.0.onnx` → `onnx/` |
| `T` / `tau` | `configs/calibration/CommandSense_calibration_v4.0.json` → `checkpoints/model_v4.0/*_calibration_v4.0.json` (legacy) |

A startup log line names the file that was actually read, and every response carries the `temperature` and `threshold` in force, so a deployment can always prove which calibration it is serving.

**Fallback.** If the calibration file is absent, unreadable, malformed or holds values the runtime cannot apply (`T <= 0`, `tau` outside `[0, 1]`), the inferencer prints a warning and falls back on `T = 1.0` / `tau = 0.0`. That is the raw-logit decision rule with no low-confidence rejection. **Model weights are different:** the inferencer only loads assets for the configured version (`MODEL_VERSION = "v4.0"`). If neither the matching `.pth` nor `.onnx` file exists, initialization raises an error explaining where to place the versioned release assets; it does not substitute weights from another version.

The exported ONNX graph keeps emitting raw logits and stores `T` and `tau` in its `metadata_props` under `commandsense.calibration.*`, so the operating point can be changed without re-exporting the model. `src/inference.py` applies `softmax(logits / T)` in Python, identically for the ONNX and PyTorch backends, and `is_low_confidence` distinguishes a threshold rejection from the learned `_unknown_` class.

### - Files in this folder

| File | Contents |
| :-- | :-- |
| `report_v4.0.txt` | Full evaluation and calibration report: per-class table, sweep, before/after metrics |
| `history_v4.0.json` | 30-epoch training history (train/val loss and accuracy) |
| `training_curves_v4.0.png` | Loss and accuracy curves for the run above |
| `confusion_matrix_v4.0.png` | 37×37 confusion matrix on the testing split |
| `calibration_v4.0.png` | Reliability diagram before and after temperature scaling |
| `coverage_vs_accuracy_v4.0.png` | Coverage and retained accuracy across the `tau` sweep |
| `robustness_report_v4.0.txt` | Held-out noise evaluation, SNR results and v4.0/v3.0 comparison |

Reproduce them with:

```
python train.py      # trains v4.0 for 30 epochs with SNR-controlled noise mixing
python evaluate.py   # fits T and tau -> configs/calibration/ + this folder
python evaluate_robustness.py --compare_v3  # compares v4.0 and v3.0 on held-out noise
```

`evaluate.py` seeds the RNG (`SEED = 42`), so a verification pass reproduces the fitted `T = 1.252585` and `tau* = 0.92` exactly.

---

## ★ Caveats

- **The mechanisms act as a bundle.** The noise mixing strategy, the RAM representation, the front-end placement, the augmentation strategy, the provisioning fallback and the epoch budget are all part of one release. The published metrics are the outcome of the combination; this card does not attribute them to any single mechanism.
- **Noise-test scope.** The held-out benchmark shows v4.0 ahead of v3.0 by 9.84–24.91 points at 5–20 dB SNR on the tested mixtures. It uses one run, a finite held-out bank (121 ambient and 512 speech chunks), and seed 42; broader noise types, SNRs, speakers and real recordings remain to be tested. These results establish the measured comparison for this protocol, not universal robustness or causality for noise mixing alone.
- **Clean-audio score difference.** On the clean test split, v4.0's 35-command Top-1 accuracy is 95.61% (vs 95.87% in v3.0), and `_unknown_` recall is 97.64% (vs 98.55%). The clean-split difference is observed alongside the noisy-condition gains; this single comparison does not isolate its cause.
- **The improvement is a training-data improvement, not a capacity improvement.** The architecture is a plain 1,217,349-parameter residual CNN: no layer was added, removed or resized. What changed is how the data reaches the model during training.
- **The rejection clips are drawn at load time.** The 10,979-clip validation split and the 12,105-clip testing split (11,005 commands + 1,100 rejection clips) each contain 1,100 synthetic `_silence_` / `_unknown_` clips that `src/dataset.py` slices out of the held-out recordings rather than storing on disk. The two rejection-class recalls are therefore samples rather than constants: re-running the evaluation with a different seed moves those cells, while the 35 command columns do not move.
- **The `Val Acc` column vs the calibration report.** The training history records 96.4204% at its best epoch; the calibration report measures 96.4113% on the same split through the evaluation path. The headline **96.41%** follows the report.
- **The threshold trades coverage for precision, and it loses on net yield.** At `tau* = 0.92` the answers served are 99.57% correct, but the end-to-end yield of _answered and correct_ clips (88.57%) is below the 95.90% the threshold-free model scores on every clip. Choose `tau` from the sweep table, not from the headline accuracy.
- **Version comparison context.** The clean and noisy comparisons here are specifically against v3.0; the full version progression is in the repository's root `README.md`.
