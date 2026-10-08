# CommandSense - Spoken Command Recognition

<p align="center">
  <a href="https://github.com/Hades-Highness/spoken-command-recognition/releases/latest">
    <img src="https://img.shields.io/github/v/release/Hades-Highness/spoken-command-recognition?color=7c3aed&label=Latest-Release&style=for-the-badge" alt="Latest Release">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ONNX%20Runtime-Supported-005ced?style=for-the-badge" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

An end-to-end deep learning pipeline for spoken keyword spotting and audio command recognition, featuring a 16 kHz high-resolution three-channel Log-Mel spectrogram front-end, a dual-backend deployment architecture, and explicit rejection of silence and out-of-vocabulary speech.

*The multi-class keyword evolution of [spoken-digit-recognition](https://github.com/Hades-Highness/spoken-digit-recognition).*

---

## - Note on Model Versions & Source Code

The current repository source code and the interactive web interface **(`app.py`)** are optimized **exclusively** for the **CommandSense v2.x** family: **v2.1** (35 command words plus `_silence_` and `_unknown_`, 37 output classes, 3-channel 16 kHz spectrograms) and its calibration release **v2.2**, which reuses the v2.1 weights byte-for-byte and adds nothing but a fitted temperature `T` and a confidence threshold `tau` on top of them. A single v2.1 checkpoint therefore reproduces every number published in this repository.

Model artifacts **(`.pth` and `.onnx`)** for official releases are available in the **GitHub Releases** section to keep the source repository lightweight and version-controlled. Older checkpoints remain downloadable from their own releases, and their results stay in the repository under `reports/model_v<version>/`.

---

## - Audio Technical Specifications

* **Sampling Rate ($f_s$)**: 16,000 Hz (mono)
* **Normalized Duration**: 1.0 second (16,000 samples)
* **Spectral Representation**: Three-Channel Log-Mel Spectrogram
  * `n_mels`: 64
  * `n_fft` / `win_length`: 1024 (~64 ms window)
  * `hop_length`: 256 (~16 ms step)
  * **Frames**: 63
  * **Channels**: 3 — Log-Mel, Delta ($\Delta$), Delta-Delta ($\Delta^2$)
  * **Normalization**: Per-Channel Log Scaling ($\log(S + 10^{-6})$)
  * **Input Tensor Shape**: `[Batch, 3, 64, 63]`

---

## - Version Progression

Every version is trained with the same recipe (residual CNN, AdamW at `1e-3`, weight decay `1e-4`, batch size 256, 20 epochs, no scheduler) and evaluated on the same official speaker-disjoint splits, so the columns below are directly comparable. Only the change named in the second column differs from one version to the next.

* **`Test Acc (35 cmd)`** is Top-1 over the 35 command words on the official `testing_list.txt` split — **11,005 clips**, identical for every version. This is the column to compare across rows.
* **`Rejection recall`** is the per-class recall on the two non-command classes, measured on 550 clips each.
* **`Overall`** is the accuracy over the full test split, which grows once rejection classes are added; it is reported per row below but is not comparable between rows.

| Version | Key change | Params | Val Acc | Test Acc (35 cmd) | Test Macro F1 | Rejection recall | Overall | Δ Test | Model Card |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **v1.0** | Baseline: single-channel Log-Mel front-end | 1,216,259 | 94.66% | 94.4571% | 0.9406 | — | 94.4571% | — | [v1.0 card](reports/model_v1.0/README.md) |
| **v2.0** | Three-channel front-end: Log-Mel + $\Delta$ + $\Delta^2$ | 1,216,835 | 94.77% | 94.7842% | 0.9436 | — | 94.7842% | +0.3271 | [v2.0 card](reports/model_v2.0/README.md) |
| **v2.1** | Rejection classes: `_silence_` (background noise) + `_unknown_` (LibriSpeech OOV speech) | 1,217,349 | 94.96%  | **94.1481%** | **0.9363** | `_silence_` 97.82%<br>`_unknown_` 98.36% | 94.5064% | **−0.6361** | [v2.1 card](reports/model_v2.1/README.md) |
| **v2.2** | Calibration release: temperature scaling + confidence threshold, **no new weights** | 1,217,349 | 94.94% | **94.1481%** | 0.9364 | `_silence_` 98.36%<br>`_unknown_` 99.09% | 94.5642% | **0.0000** ‡ | [v2.2 card](reports/model_v2.2/README.md) |

† The v2.1 validation split contains 10,979 clips (9,981 commands + 998 rejection samples), so its `Val Acc` is not comparable to the rows above. The comparable figures are the test columns.

‡ v2.2 takes **no gradient step at all**: `Δ Test` is zero by construction, not by measurement, because temperature scaling cannot change an argmax. All 11,005 command clips are classified exactly as in the v2.1 pass — the per-class recall of every one of the 35 commands is identical row-for-row in the two reports — so every accuracy cell on the command task is a copy of the v2.1 cell. Only the columns that mix in the redrawn rejection clips move — `Test Macro F1` (0.9363 → 0.9364, through precision), `Overall` (94.5064 % → 94.5642 %) and `Rejection recall` — and they move only because `src/dataset.py` slices the 1,100 synthetic `_silence_` / `_unknown_` clips out of the held-out recordings at load time instead of storing them on disk. Treat the v2.2 row as a *calibration and rejection* result, not an accuracy result.

### - How to read the progression

* **v2.1 trades command accuracy for rejection capability.** Top-1 on the 35 commands falls by **0.6361 pt** against v2.0 (−70 clips). The standard error of that unpaired difference is ≈0.31 pt, so this cost is at the edge of significance — larger than the noise floor, unlike the v2.0 gain.
* **Decomposition of the 70 lost clips**: 37 are commands wrongly rejected as silence or unknown (3 + 34), and 33 are commands now confused with other commands. The **false-rejection rate on commands is 0.34 %** (37 of 11,005).
* **v2.1 is indistinguishable from v1.0** on the command task (94.1481 % vs 94.4571 %, z ≈ 0.99).
* **A `Δ Test` below ~0.4 points is not resolvable.** The standard error on a single accuracy estimate at this level is ≈0.21 points; differences smaller than that should not be read as improvements or regressions.
* **Per-version detail lives in the model cards, not here.** Each card carries its own epoch-by-epoch history, full per-class table, confusion matrix and caveats, and each card is written to stand alone — cards are not compared against each other.
* **Watch the `Params` column.** Parameters are capacity. A version that changes both the architecture and the front-end at once is not a single-variable experiment.
* **v2.2 is the one row that cannot move.** It changes how the model's *confidence* is reported, not what the model computes, so its accuracy cells are copies of the v2.1 cells by construction. Read it for the calibration error and the confidence-threshold behaviour, which the other rows do not have at all.

---

## - Model Architecture

All versions share the same residual network; only the first convolution and the classification head change with the class count.

| Stage | Configuration | Output shape |
| :--- | :--- | :--- |
| Input | — | `[B, 3, 64, 63]` |
| `in_conv` | `Conv2d(3→32, 3×3)`, BN, ReLU | `[B, 32, 64, 63]` |
| `layer1` | `ResidualBlock(32→64, stride 2)` | `[B, 64, 32, 32]` |
| `layer2` | `ResidualBlock(64→128, stride 2)` | `[B, 128, 16, 16]` |
| `layer3` | `ResidualBlock(128→256, stride 2)` | `[B, 256, 8, 8]` |
| Pooling | `AdaptiveAvgPool2d((1, 1))` | `[B, 256, 1, 1]` |
| Head | Flatten → `Dropout(0.3)` → `Linear(256→N)` | `[B, N]` |

Each `ResidualBlock` is two 3×3 convolutions with batch normalization and a 1×1 projection shortcut where the shape changes. `N` is 35 for v1.0 and v2.0, and 37 for v2.1 and v2.2 — v2.2 introduces no layer, no parameter and no change to the graph.

---

## - Project Structure

```text
.
├── checkpoints/                  # Model weight releases (git-ignored; created at runtime)
│   ├── model_v2.1/               # CommandSense_v2.1.pth, CommandSense_v2.1.onnx
│   └── model_v2.2/               # CommandSense_calibration_v2.2.json (T + tau; ships no weights)
├── configs/                      # Global hyperparameters, paths and the class mapping
│   ├── config.py                 # Training, audio and calibration settings
│   └── labels.json               # Index -> class name mapping (37 entries for v2.1)
├── data/                         # Raw datasets (git-ignored, auto-downloaded)
│   ├── SpeechCommands/           # Google Speech Commands v0.02
│   └── LibriSpeech/              # Out-of-vocabulary speech for _unknown_
├── reports/                      # Versioned evaluation cards, histories, reports and plots
│   ├── model_v1.0/
│   ├── model_v2.0/
│   ├── model_v2.1/
│   └── model_v2.2/               # Calibration report, tau sweep, reliability & coverage plots
├── src/                          # Source code modules
│   ├── calibration.py            # Temperature scaling, ECE / reliability, tau sweep (v2.2)
│   ├── dataset.py                # Speech Commands + rejection-class construction
│   ├── inference.py              # Dual-backend (ONNX & PyTorch) inferencer applying T and tau
│   ├── models.py                 # AudioFeatureExtractor + ResidualBlock + CommandSense
│   └── utils.py                  # Audio loading and preprocessing (single source of truth)
├── app.py                        # Interactive Gradio Web UI with a live tau slider
├── evaluate.py                   # Metrics, classification report, calibration fit and tau sweep
├── export_onnx.py                # PyTorch to ONNX exporter with strict shape validation
├── requirements.txt              # Python environment dependencies
├── train.py                      # Training pipeline with AMP and graceful termination
└── LICENSE                       # MIT
```

---

## - Quick Start

#### - Clone repository
```text
git clone https://github.com/Hades-Highness/spoken-command-recognition.git
cd spoken-command-recognition
```

#### - Install dependencies
```text
pip install -r requirements.txt
```

#### - Download the model weights
Checkpoints are not committed, so place the release assets into the versioned folder:
```text
checkpoints/model_v2.1/CommandSense_v2.1.pth
checkpoints/model_v2.1/CommandSense_v2.1.onnx   (optional, for the ONNX backend)
```

#### - Launch interactive Web UI
```text
python app.py
```

#### - Reproduce the published metrics
```text
python train.py       # trains v2.1; downloads Speech Commands and LibriSpeech on first run
python evaluate.py    # fits T/tau and writes reports/model_v2.2/ + checkpoints/model_v2.2/
python export_onnx.py # optional: re-export the v2.1 graph with T/tau in its metadata_props
```
The first training run downloads Speech Commands v0.02 (≈2.3 GB) and LibriSpeech `train-clean-100` (≈6.3 GB). v2.2 requires no training of its own: `evaluate.py` fits its single temperature on the validation split and never takes a gradient step.

---

## - Rejection & Confidence

v2.1 introduces two non-command classes so the model can decline to answer:

| Class | Training source | Evaluation source | Recall |
| :--- | :--- | :--- | :---: |
| `_silence_` | 4 of the 6 `_background_noise_` recordings | the 2 held-out recordings | 97.82% (538/550) |
| `_unknown_` | LibriSpeech `train-clean-100` (OOV read speech) | LibriSpeech `dev-clean` (disjoint speakers) | 98.36% (541/550) |

Rejection is a learned class, not a confidence threshold: the model outputs `_silence_` or `_unknown_` as the argmax label. When it does so on a real command, that command is lost — currently **0.34 % of command clips** (37 of 11,005). The recalls above are from the v2.1 pass; v2.2 re-draws the same 1,100 evaluation clips and measures 98.36 % (541/550) and 99.09 % (545/550), so read those two cells as samples rather than constants.

v2.2 adds the second, orthogonal half of the story: a **calibrated confidence threshold** that rejects the clips the model is unsure about — including command clips the learned classes cannot catch. See [Calibration & Confidence Thresholding](#-calibration--confidence-thresholding-v22).

---

## - Calibration & Confidence Thresholding (v2.2)

v2.1 reports `softmax(logits)` as its confidence, and that number is systematically **too high**: the clips it labels at 90 % confidence are right far less often than 90 % of the time. **v2.2 fixes the confidence, not the accuracy.** It is a post-hoc calibration release — it trains nothing, owns no parameters and loads the v2.1 checkpoint unchanged.

| | v2.1 | v2.2 |
| :--- | :--- | :--- |
| Weights | 1,217,349 parameters | identical, frozen (no gradient step is ever taken) |
| Reported confidence | raw `softmax(logits)`, over-confident | `softmax(logits / T)`, `T = 1.8074` fitted on the validation split |
| Rejection | learned classes `_silence_` / `_unknown_`, decided by the argmax | learned classes **plus** a calibrated confidence threshold `tau* = 0.80` |
| ONNX graph | raw logits | raw logits, unchanged, with `T` and `tau` in `metadata_props` |
| Serving | `app.py` shows the argmax label | `app.py` adds the calibrated confidence, an accept/reject badge and a live `tau` slider |

**Temperature scaling** (Guo et al., 2017) divides every logit by the same positive scalar before the softmax, so the ranking of the classes — and therefore the argmax — is mathematically unchanged. Top-1 on the 35 commands is **identical to v2.1 by construction**: 94.1481 % (10,361 / 11,005). Only the confidences move.

| Split | Clips | NLL (`T = 1` → `T = 1.8074`) | ECE (→) | MCE (→) |
| :--- | ---: | ---: | ---: | ---: |
| Validation — where `T` was fitted | 10,979 | 0.240624 → 0.190525 | 0.024844 → **0.005109** | 0.269240 → 0.100179 |
| Testing — out of sample | 12,105 | 0.239402 → 0.195932 | 0.025993 → **0.007664** | 0.226169 → 0.115049 |

The test ECE falls by 70.5 % even though `T` was fitted on a different split, which is the practical advantage of temperature scaling: one parameter, ≈11 k clips, nothing to overfit.

### - Confidence Threshold Decision

A clip is answered only when its calibrated top-1 probability reaches `tau`; otherwise it is served as `_unknown_`, the project's existing reject class. `tau` is swept over `[0, 1]` in steps of `0.01` on the validation split and selected by `min_cost` (`FRR + FAR`) under a 90 % coverage floor, which picks `tau* = 0.80`.

| Metric (testing split, 12,105 clips) | Value |
| :--- | ---: |
| Coverage (clips answered) | **89.3350 %** (10,814 / 12,105) |
| Retained accuracy (answered clips only) | **98.6591 %** (10,669 / 10,814) |
| False rejection rate — correct clips discarded | 6.7965 % |
| False acceptance rate — incorrect clips served | 22.0365 % |
| Accuracy with no threshold at all | 94.5642 % of all clips |
| End-to-end yield — answered **and** correct | 88.1371 % of all clips |
| Rejected as uncertain | 10.6650 % (served as `_unknown_`) |

The threshold buys **precision, not accuracy**. At `tau*` the answers that are given are correct 98.66 % of the time instead of 94.56 %, and 513 of the split's 658 errors are suppressed before they reach a caller. The price is coverage: 778 correct clips are discarded, so the end-to-end yield of *correct and answered* clips (88.14 %) is **below** the 94.56 % the threshold-free model scores on every clip. The 145 wrong predictions that still clear `tau` are served with no warning — a threshold reduces errors, it does not remove them.

#### - Deployment

* `checkpoints/model_v2.2/CommandSense_calibration_v2.2.json` is the deployment payload (the fitted `T` and `tau`). It carries no weights.
* The exported graph keeps emitting **raw logits** and stores the same values in its `metadata_props` under `commandsense.calibration.*`, so `T` and `tau` can be changed without re-exporting the model.
* `src/inference.py` applies `softmax(logits / T)` in Python, identically for the ONNX and PyTorch backends, and every response carries `raw_label` (the raw argmax) and `is_low_confidence`, so a threshold rejection can be told apart from the learned `_unknown_` class.
* Verified end to end: `_background_noise_/doing_the_dishes.wav` is answered `_silence_` with confidence `0.9994` on both backends, and a command clip scored `0.9613` is served at `tau = 0.80` but becomes `_unknown_` with `raw_label = backward` at `tau = 0.99`.

The reliability diagram, the coverage/accuracy curve, the full `tau` sweep and the before/after counts live in the [v2.2 model card](reports/model_v2.2/README.md).

---

## - Datasets & Citation

| Dataset | Content | Source rate | Licence |
| :--- | :--- | :---: | :--- |
| [Google Speech Commands v0.02](https://www.tensorflow.org/datasets/catalog/speech_commands) | 105,829 clips, 35 command words + background noise, 2,618 speakers | 16 kHz | CC BY 4.0 |
| [LibriSpeech](https://www.openslr.org/12/) (`train-clean-100`, `dev-clean`) | 16 kHz read English speech used to build `_unknown_` | 16 kHz | CC BY 4.0 |

Google Speech Commands is loaded through `torchaudio.datasets.SPEECHCOMMANDS`. The **training / validation / testing** split comes from the dataset's own `validation_list.txt` and `testing_list.txt`, which are speaker-disjoint by construction — this is why the validation and test numbers in this repository agree so closely, and why they are not inflated by speaker identity.

LibriSpeech is loaded through `torchaudio.datasets.LIBRISPEECH` using **disjoint subsets**: `train-clean-100` for training and `dev-clean` for validation and testing. Their speakers do not overlap, and neither overlaps the Speech Commands speakers, so the `_unknown_` measurement is not contaminated by voices the model has already seen.

```bib
@article{warden2018speech,
  title   = {Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition},
  author  = {Warden, Pete},
  journal = {arXiv preprint arXiv:1804.03209},
  year    = {2018},
  url     = {https://arxiv.org/abs/1804.03209}
}

@inproceedings{panayotov2015librispeech,
  title     = {LibriSpeech: An ASR corpus based on public domain audio books},
  author    = {Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle = {2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {5206--5210},
  year      = {2015},
  doi       = {10.1109/ICASSP.2015.7178964}
}
```

Both datasets are released under **CC BY 4.0**, so attribution is required when the data is redistributed or used in published work.

---

## - Licence

This project is released under the **MIT Licence**. The datasets keep their own terms (both CC BY 4.0). Model checkpoints published as release assets are covered by this project's MIT licence, but any model retrained or redistributed using the datasets must also respect their licences.
