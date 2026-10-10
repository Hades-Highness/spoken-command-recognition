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

An end-to-end deep learning pipeline for spoken keyword spotting and audio command recognition, featuring a 16 kHz high-resolution three-channel Log-Mel spectrogram front-end, a dual-backend deployment architecture, and explicit rejection of silence and out-of-vocabulary speech. **v3.0** moves the whole signal front-end onto the accelerator: raw `torch.int16` audio now sits in RAM and the Log-Mel + Δ + Δ² features are built on the fly on the GPU, together with GPU-native conditional augmentation.

*The multi-class keyword evolution of [spoken-digit-recognition](https://github.com/Hades-Highness/spoken-digit-recognition).*

---

## - What's New in v3.0

**CommandSense v3.0** reaches **96.36 % validation accuracy** (up from 94.94 % in v2.2) with the **same parameter count — 1,217,349 parameters**. The gain comes from a rebuilt data pipeline, not a bigger network: the residual CNN is untouched, and the four changes below are the whole story.

#### - Memory-Optimized Pipeline

* The dataset caches **raw audio as `torch.int16`** instead of pre-computed float32 spectrograms — `16,000 samples × 2 bytes ≈ 32 KB per clip`, a **~75 % reduction in RAM footprint** against the old 3-channel float32 cache.
* Feature extraction (**Log-Mel + Δ + Δ²**) is now **computed dynamically on the GPU** (`src/transforms.py`, `AudioToThreeChannelMel`), so the CPU never materialises a spectrogram and the freed RAM is left to the model and the data.
* The int16 cache is the single source of truth: training, evaluation and serving all derive their inputs from it, so the front-end cannot drift between phases.

#### - Conditional Data Augmentation

* Three **GPU-native** augmentations run per batch, **only in `train()` mode**:
  * **Spectral Shift** — an STFT-domain pitch shift (`stft → torch.roll → istft`) that replaces the heavy phase vocoder;
  * **Time Shift** — a single `torch.roll` on the waveform;
  * **SpecAugment** — independent per-clip frequency and time masks.
* They are **strictly disabled for the `_silence_` class**. `_silence_` teaches the reject (OOD) boundary from raw background noise, so shifting or masking those clips would only blur the frontier between silence and out-of-vocabulary speech.

#### - Resilient Data Provisioning

* The downloader carries a **secure SSL fallback**: when the TensorFlow CDN serves a chain the local trust store cannot verify (`CERTIFICATE_VERIFY_FAILED`, a known intermittent issue), the transfer is retried once over an unverified TLS context. Every other network error is re-raised untouched, so genuine failures stay visible.
* Provisioning is centralized and idempotent (`ensure_speech_commands` / `ensure_librispeech`), so an unattended install completes without a manual workaround.

#### - Performance

* The **Spectral Shift** removes the phase vocoder from the hot path — it is **~180× cheaper** — which keeps the entire front-end inside the per-batch budget.
* Training runs at **≈25–30 s per epoch on a single T4** at batch size 256.

**Net effect on the official splits:** 35-command Top-1 rises from **94.1481 %** (v2.2) to **95.8655 %** (v3.0), and both rejection classes improve (`_silence_` **100.00 %**, `_unknown_` **98.55 %**).

---

## - Note on Model Versions & Source Code

The current repository source code and the interactive web interface **(`app.py`)** are optimized **exclusively** for the **CommandSense v3.0** release — no longer for the v2.x line: 37 output classes (35 command words plus `_silence_` and `_unknown_`), a GPU-resident three-channel 16 kHz front-end, and a fitted temperature `T` with a confidence threshold `tau` applied on top of the trained weights. A single `CommandSense_v3.0.pth` serves the whole repository, so the CLI (`src/inference.py`) and the web UI (`app.py`) always report exactly the numbers published below. The v2.x family — up to and including the **v2.2** calibration release, which reuses the v2.1 weights byte-for-byte and adds nothing but `T` and `tau` — remains available from its own tags and releases, and its results stay under `reports/model_v2*/`.

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

Every version is trained with the same recipe (residual CNN, AdamW at `1e-3`, weight decay `1e-4`, batch size 256, no scheduler) and evaluated on the same official speaker-disjoint splits, so the columns below are directly comparable. v1.0–v2.1 were trained for 20 epochs; **v3.0 extends the schedule to 30 epochs** (and v2.2 trains nothing at all). Apart from that, only the change named in the second column differs from one version to the next.

* **`Test Acc (35 cmd)`** is Top-1 over the 35 command words on the official `testing_list.txt` split — **11,005 clips**, identical for every version. This is the column to compare across rows.
* **`Rejection recall`** is the per-class recall on the two non-command classes, measured on 550 clips each.
* **`Overall`** is the accuracy over the full test split, which grows once rejection classes are added; it is reported per row below but is not comparable between rows.
* **`Test Macro F1`** is the unweighted mean F1 over the 35 command classes (rejection classes excluded).

| Version | Key change | Params | Val Acc | Test Acc (35 cmd) | Test Macro F1 | Rejection recall | Overall | Δ Test | Model Card |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **v1.0** | Baseline: single-channel Log-Mel front-end | 1,216,259 | 94.66% | 94.4571% | 0.9406 | — | 94.4571% | — | [v1.0 card](reports/model_v1.0/README.md) |
| **v2.0** | Three-channel front-end: Log-Mel + $\Delta$ + $\Delta^2$ | 1,216,835 | 94.77% | 94.7842% | 0.9436 | — | 94.7842% | +0.3271 | [v2.0 card](reports/model_v2.0/README.md) |
| **v2.1** | Rejection classes: `_silence_` (background noise) + `_unknown_` (LibriSpeech OOV speech) | 1,217,349 | 94.96% † | **94.1481%** | **0.9363** | `_silence_` 97.82%<br>`_unknown_` 98.36% | 94.5064% | **−0.6361** | [v2.1 card](reports/model_v2.1/README.md) |
| **v2.2** | Calibration release: temperature scaling + confidence threshold, **no new weights** | 1,217,349 | 94.94% | **94.1481%** | 0.9364 | `_silence_` 98.36%<br>`_unknown_` 99.09% | 94.5642% | **0.0000** ‡ | [v2.2 card](reports/model_v2.2/README.md) |
| **v3.0** | Memory-optimized pipeline: int16 RAM cache + GPU front-end + conditional GPU augmentation (30-epoch schedule) | 1,217,349 | 96.36% | **95.8655%** | 0.9541 § | `_silence_` 100.00%<br>`_unknown_` 98.55% | 96.1751% | **+1.7174** | [v3.0 card](reports/model_v3.0/README.md) |

† The v2.1 validation split contains 10,979 clips (9,981 commands + 998 rejection samples), so its `Val Acc` is not comparable to the rows above. The comparable figures are the test columns.

‡ v2.2 takes **no gradient step at all**: `Δ Test` is zero by construction, not by measurement, because temperature scaling cannot change an argmax. All 11,005 command clips are classified exactly as in the v2.1 pass — the per-class recall of every one of the 35 commands is identical row-for-row in the two reports — so every accuracy cell on the command task is a copy of the v2.1 cell. Only the columns that mix in the redrawn rejection clips move — `Test Macro F1` (0.9363 → 0.9364, through precision), `Overall` (94.5064 % → 94.5642 %) and `Rejection recall` — and they move only because `src/dataset.py` slices the 1,100 synthetic `_silence_` / `_unknown_` clips out of the held-out recordings at load time instead of storing them on disk. Treat the v2.2 row as a *calibration and rejection* result, not an accuracy result.

§ v3.0's `Test Macro F1` is the unweighted mean of the 35 command-class F1 scores printed in [`reports/model_v3.0/report_v3.0.txt`](reports/model_v3.0/report_v3.0.txt); the same report gives the 37-class macro average as `0.9560`. Unlike v2.2, v3.0 trains new weights, so its row is a genuine accuracy result.

### - How to read the progression

* **v3.0 is the first version to move the accuracy needle since v2.0.** 35-command Top-1 climbs **+1.7174 pt** (94.1481 % → 95.8655 %, +189 clips) — well above the ±0.4-pt resolution floor — while rejection improves at the same time (`_silence_` 100.00 %, `_unknown_` 98.55 %). That gain is the **net effect of a bundle**, because v3.0 moves four things at once: the audio cache becomes `int16` (≈75 % less RAM), the Log-Mel + Δ + Δ² front-end moves to the GPU, the CPU augmentation stack is replaced by GPU-native augmentation (spectral shift, time shift, **dynamic SpecAugment** — disabled for `_silence_`), and the schedule extends to **30 epochs**. No single change is isolated by that run, so read the row below as the outcome of the bundle taken as a whole; the [v3.0 card](reports/model_v3.0/README.md) documents each mechanism.
* **v2.1 trades command accuracy for rejection capability.** Top-1 on the 35 commands falls by **0.6361 pt** against v2.0 (−70 clips). The standard error of that unpaired difference is ≈0.31 pt, so this cost is at the edge of significance — larger than the noise floor, unlike the v2.0 gain.
* **Decomposition of the 70 lost clips**: 37 are commands wrongly rejected as silence or unknown (3 + 34), and 33 are commands now confused with other commands. The **false-rejection rate on commands is 0.34 %** (37 of 11,005).
* **v2.1 is indistinguishable from v1.0** on the command task (94.1481 % vs 94.4571 %, z ≈ 0.99).
* **A `Δ Test` below ~0.4 points is not resolvable.** The standard error on a single accuracy estimate at this level is ≈0.21 points; differences smaller than that should not be read as improvements or regressions.
* **Per-version detail lives in the model cards, not here.** Each card carries its own epoch-by-epoch history, full per-class table, confusion matrix and caveats, and each card is written to stand alone — cards are not compared against each other.
* **Watch the `Params` column.** Parameters are capacity. A version that changes both the architecture and the front-end at once is not a single-variable experiment.
* **v2.2 is the one row that cannot move.** It changes how the model's *confidence* is reported, not what the model computes, so its accuracy cells are copies of the v2.1 cells by construction. Read it for the calibration error and the confidence-threshold behaviour, which the other rows do not have at all.

---

## - Model Architecture

All versions share the same residual network; only the first convolution and the classification head change with the class count. Since v3.0 the three-channel front-end no longer lives in the dataset: it is `AudioToThreeChannelMel` in `src/transforms.py`, and it runs on the accelerator. The network below is otherwise identical to the v2.x family.

| Stage | Configuration | Output shape |
| :--- | :--- | :--- |
| Input | — | `[B, 3, 64, 63]` |
| `in_conv` | `Conv2d(3→32, 3×3)`, BN, ReLU | `[B, 32, 64, 63]` |
| `layer1` | `ResidualBlock(32→64, stride 2)` | `[B, 64, 32, 32]` |
| `layer2` | `ResidualBlock(64→128, stride 2)` | `[B, 128, 16, 16]` |
| `layer3` | `ResidualBlock(128→256, stride 2)` | `[B, 256, 8, 8]` |
| Pooling | `AdaptiveAvgPool2d((1, 1))` | `[B, 256, 1, 1]` |
| Head | Flatten → `Dropout(0.3)` → `Linear(256→N)` | `[B, N]` |

Each `ResidualBlock` is two 3×3 convolutions with batch normalization and a 1×1 projection shortcut where the shape changes. `N` is 35 for v1.0 and v2.0, and 37 for v2.1, v2.2 and v3.0. v2.2 introduces no layer, no parameter and no change to the graph, and v3.0 changes neither the graph nor the parameter count — the whole v3.0 gain comes from the data pipeline built around the model.

---

## - Project Structure

```text
.
├── checkpoints/                  # Model weight releases (git-ignored; created at runtime)
│   ├── model_v1.0/               # v1.0 weights: CommandSense_v1.0.pth + CommandSense_v1.0.onnx
│   ├── ...
│   └── model_v3.0/               # v3.0 weights: CommandSense_v3.0.pth + CommandSense_v3.0.onnx
├── configs/                      # Global hyperparameters, paths and the class mapping
│   ├── calibration/              # Calibration payload: CommandSense_calibration_v3.0.json (T and tau)
│   ├── config.py                 # Training, audio, augmentation and calibration settings
│   └── labels.json               # Index -> class name mapping (37 entries)
├── data/                         # Raw datasets (git-ignored, auto-downloaded)
│   ├── SpeechCommands/           # Google Speech Commands v0.02
│   └── LibriSpeech/              # Out-of-vocabulary speech for _unknown_
├── reports/                      # Versioned evaluation cards, histories, reports and plots
│   ├── model_v1.0/
│   ├── ...
│   └── model_v3.0/         
├── src/                          # Source code modules
│   ├── calibration.py            # Temperature scaling, ECE / reliability, tau sweep
│   ├── dataset.py                # int16 RAM cache, centralized provisioning + rejection classes
│   ├── inference.py              # Dual-backend (ONNX & PyTorch) inferencer applying T and tau
│   ├── models.py                 # AudioFeatureExtractor + ResidualBlock + CommandSense
│   ├── transforms.py             # GPU front-end: Log-Mel + Delta + Delta-Delta, conditional augmentation
│   └── utils.py                  # Audio loading and preprocessing (single source of truth)
├── tests/
│   ├── test_calibration_smoke.py  # Lightweight evaluation smoke test (no dataset required)
│   └── test_v3_pipeline.py        # v3.0 int16 cache, GPU front-end and conditional augmentation tests
├── app.py                        # Interactive Gradio Web UI with a live tau slider
├── evaluate.py                   # Metrics, classification report, calibration fit and tau sweep
├── export_onnx.py                # PyTorch to ONNX exporter with strict shape validation
├── requirements.txt              # Python environment dependencies
├── train.py                      # Training pipeline with AMP and graceful termination
└── LICENSE                       # MIT LICENCE
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
Checkpoints are not committed, so place the v3.0 release assets into the
versioned weight folder and the calibration payload into `configs/calibration/`:
```text
checkpoints/model_v3.0/
├── CommandSense_v3.0.pth
└── CommandSense_v3.0.onnx

configs/calibration/
└── CommandSense_calibration_v3.0.json
```
The v3.0 release includes all three files, so it is a self-contained install:
`CommandSense_v3.0.pth` holds the trained weights, `CommandSense_v3.0.onnx` is
the same model exported for ONNX Runtime, and the calibration JSON carries the
newly fitted `T` and `tau`. `src/inference.py` and `app.py` discover the files at
start-up: they load the weights, prefer the ONNX backend when `onnxruntime` is
installed, and read `T` and `tau` from the JSON - no path ever has to be passed
on the command line. `CommandSense_v3.0.onnx` is optional (PyTorch-only setups
can drop it); the calibration JSON is versioned configuration rather than a
weight, so it lives in `configs/` next to `labels.json` and is tracked in git.

**Suggested v3.0 release note:** “CommandSense v3.0 ships fully retrained
weights — `CommandSense_v3.0.pth` and its `CommandSense_v3.0.onnx` export — at an
unchanged parameter count (1,217,349). It reaches 96.36 % validation accuracy and
95.8655 % Top-1 on the 35-command test split, driven by a memory-optimized
pipeline (raw `int16` RAM cache, GPU feature extraction and GPU-native
conditional augmentation). The release also includes
`CommandSense_calibration_v3.0.json`; place it in `configs/calibration/` to
enable calibrated confidence and threshold rejection.”

Grab the assets from the [Releases page](https://github.com/Hades-Highness/spoken-command-recognition/releases)
and drop them in place, or produce them locally with the commands below. The
inferencer only loads model files for the configured version (`MODEL_VERSION`,
`v3.0` by default). If that version's `.pth` and `.onnx` files are both missing,
initialization raises a clear error; it will not silently substitute another
version. A missing or unusable calibration file still falls back to
`T = 1.0` / `tau = 0.0`.

Unlike v2.2, v3.0 trains real weights, so its report folder carries the full
training history alongside the evaluation and calibration artifacts:
```text
reports/model_v3.0/
├── report_v3.0.txt
├── history_v3.0.json
├── training_curves_v3.0.png
├── confusion_matrix_v3.0.png
├── calibration_v3.0.png
├── coverage_vs_accuracy_v3.0.png
└── README.md
```

#### - Launch interactive Web UI
```text
python app.py
```

#### - Reproduce the published metrics
```text
python train.py       # trains v3.0; downloads Speech Commands and LibriSpeech on first run
python evaluate.py    # fits T/tau -> configs/calibration/ + reports/model_v3.0/
python export_onnx.py # optional: writes CommandSense_v3.0.onnx with T/tau in its metadata_props
```
The first training run downloads Speech Commands v0.02 (≈2.3 GB) and LibriSpeech `train-clean-100` (≈6.3 GB). `train.py` caches the audio as `int16` in RAM and builds the spectrograms on the GPU, so an epoch takes ≈25–30 s on a T4 at batch size 256; `evaluate.py` then fits the single temperature `T` and the confidence threshold `tau` on the validation split.

---

## - Rejection & Confidence

v2.1 introduced two non-command classes so the model can decline to answer; v3.0 keeps both and sharpens them:

| Class | Training source | Evaluation source | v3.0 recall |
| :--- | :--- | :--- | :---: |
| `_silence_` | 4 of the 6 `_background_noise_` recordings | the 2 held-out recordings | 100.00% (550/550) |
| `_unknown_` | LibriSpeech `train-clean-100` (OOV read speech) | LibriSpeech `dev-clean` (disjoint speakers) | 98.55% (542/550) |

Rejection is a learned class, not a confidence threshold: the model outputs `_silence_` or `_unknown_` as the argmax label. When it does so on a real command, that command is lost. On the 35-command test split v3.0 reaches 95.8655 % Top-1 (10,550 / 11,005), so 455 command clips are misclassified — either confused with another command or rejected. The `_silence_` recall is now perfect on the held-out recordings and `_unknown_` rises to 98.55 %; because those two cells are drawn from held-out clips at load time, read them as samples rather than constants.

v2.2 added the second, orthogonal half of the story: a **calibrated confidence threshold** that rejects the clips the model is unsure about — including command clips the learned classes cannot catch. v3.0 keeps it and re-fits it on the new weights. See [Calibration & Confidence Thresholding](#--calibration--confidence-thresholding-v30).

---

## - Calibration & Confidence Thresholding (v3.0)

v2.1 reported `softmax(logits)` as its confidence, and that number was systematically **too high**: the clips it labels at 90 % confidence are right far less often than 90 % of the time. **v2.2 fixed the confidence, not the accuracy** — a post-hoc calibration release that trains nothing, owns no parameters and loads the v2.1 checkpoint unchanged. **v3.0 keeps that machinery and pairs it with genuinely new weights**, re-fitting `T` and `tau` on the v3.0 checkpoint.

| | v2.1 | v2.2 | v3.0 |
| :--- | :--- | :--- | :--- |
| Weights | 1,217,349 parameters | identical, frozen (no gradient step is ever taken) | **retrained** (30 epochs), 1,217,349 parameters |
| Reported confidence | raw `softmax(logits)`, over-confident | `softmax(logits / T)`, `T = 1.8074` fitted on the validation split | `softmax(logits / T)`, **`T = 1.3116`** fitted on the validation split |
| Rejection | learned classes `_silence_` / `_unknown_`, decided by the argmax | learned classes **plus** a calibrated confidence threshold `tau* = 0.80` | learned classes **plus** a calibrated confidence threshold **`tau* = 0.89`** |
| ONNX graph | raw logits | raw logits, unchanged, with `T` and `tau` in `metadata_props` | raw logits, unchanged, with `T` and `tau` in `metadata_props` |
| Serving | `app.py` shows the argmax label | `app.py` adds the calibrated confidence, an accept/reject badge and a live `tau` slider | same UI, driven by the v3.0 payload |

**Temperature scaling** (Guo et al., 2017) divides every logit by the same positive scalar before the softmax, so the ranking of the classes — and therefore the argmax — is mathematically unchanged. In v2.2 that meant Top-1 was **identical to v2.1 by construction** (94.1481 %); in v3.0 the argmax is unchanged relative to the *v3.0* raw model, whose 35-command Top-1 is **95.8655 %** (10,550 / 11,005). Calibration still moves only the confidence values.

| Split | Clips | NLL (`T = 1` → `T = 1.3116`) | ECE (→) | MCE (→) |
| :--- | ---: | ---: | ---: | ---: |
| Validation — where `T` was fitted | 10,979 | 0.139316 → 0.132453 | 0.009217 → **0.006232** | 0.168483 → 0.138063 |
| Testing — out of sample | 12,105 | 0.132331 → 0.127490 | 0.010317 → **0.005332** | 0.192082 → 0.125716 |

The test ECE falls by **48.3 %** even though `T` was fitted on a different split, which is the practical advantage of temperature scaling: one parameter, ≈11 k clips, nothing to overfit. The v3.0 model already starts from a much better-calibrated base than v2.1 (`ECE = 0.0103` before fitting, against `0.0260`), which is expected — a stronger model with the same capacity makes fewer confident mistakes — so the scalar has less head-room to exploit and the fitted `T` is correspondingly smaller.

### - Confidence Threshold Decision

A clip is answered only when its calibrated top-1 probability reaches `tau`; otherwise it is served as `_unknown_`, the project's existing reject class. `tau` is swept over `[0, 1]` in steps of `0.01` on the validation split and selected by `min_cost` (`FRR + FAR`) under a 90 % coverage floor, which picks `tau* = 0.89` for v3.0 (up from `0.80` in v2.2).

| Metric (testing split, 12,105 clips) | Value |
| :--- | ---: |
| Coverage (clips answered) | **89.7646 %** (10,866 / 12,105) |
| Retained accuracy (answered clips only) | **99.4754 %** (10,809 / 10,866) |
| False rejection rate — correct clips discarded | 7.1551 % |
| False acceptance rate — incorrect clips served | 12.3110 % |
| Accuracy with no threshold at all | 96.1751 % of all clips |
| End-to-end yield — answered **and** correct | 89.2937 % of all clips |
| Rejected as uncertain | 10.2354 % (served as `_unknown_`) |

The threshold buys **precision, not accuracy**. At `tau*` the answers that are given are correct 99.48 % of the time instead of 96.18 %, and 406 of the split's 463 errors are suppressed before they reach a caller. The price is coverage: 833 correct clips are discarded, so the end-to-end yield of *correct and answered* clips (89.29 %) is **below** the 96.18 % the threshold-free model scores on every clip. The 57 wrong predictions that still clear `tau` are served with no warning — a threshold reduces errors, it does not remove them.

#### - Deployment

* `configs/calibration/CommandSense_calibration_v3.0.json` is the deployment payload (the fitted `T` and `tau`). It carries no weights, so it is committed with the code instead of living in the git-ignored `checkpoints/` tree.
* `src/inference.py` resolves that JSON by itself — `configs/calibration/` first, then the configured version's checkpoint folder as a legacy location — and names the file it read at start-up. If it is missing, unreadable or unusable, the inferencer falls back on `T = 1.0` / `tau = 0.0` — raw logits, no confidence rejection — and never raises. Model weights are not substituted across versions: if no model asset for the configured version is present, initialization raises an actionable error.
* The exported graph keeps emitting **raw logits** and stores the same values in its `metadata_props` under `commandsense.calibration.*`, so `T` and `tau` can be changed without re-exporting the model.
* `src/inference.py` applies `softmax(logits / T)` in Python, identically for the ONNX and PyTorch backends, and every response carries `raw_label` (the raw argmax) and `is_low_confidence`, so a threshold rejection can be told apart from the learned `_unknown_` class.
* Verified end to end on the original v2.2 calibration (the mechanics are identical at v3.0): `_background_noise_/doing_the_dishes.wav` is answered `_silence_` with confidence `0.9994` on both backends, and a command clip scored `0.9613` is served at `tau = 0.80` but becomes `_unknown_` with `raw_label = backward` at `tau = 0.99`.

The reliability diagram, the coverage/accuracy curve, the full `tau` sweep and the before/after counts live in the [v3.0 model card](reports/model_v3.0/README.md); the original calibration pass is documented in the [v2.2 model card](reports/model_v2.2/README.md).

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

## Licence

This project is released under the [MIT License](LICENSE) — © 2026 Hades-Highness.

The datasets keep their own terms (both CC BY 4.0). Model checkpoints published as release assets are covered by this project's MIT licence, but any model retrained or redistributed using the datasets must also respect their licences.

---

## Acknowledgements

This project builds upon foundational work in the speech recognition community and leverages several open-source tools:

- **Datasets**: [Google Speech Commands v0.02](https://arxiv.org/abs/1804.03209) (Warden, 2018) and [LibriSpeech](https://www.openslr.org/12/) (Panayotov et al., 2015) for training and OOD evaluation.
- **Frameworks**: [PyTorch](https://pytorch.org/) and [Torchaudio](https://pytorch.org/audio/) for the deep learning pipeline and GPU-native audio processing.
- **Techniques**: The conditional data augmentation strategy is inspired by [SpecAugment](https://arxiv.org/abs/1904.08779) (Park et al., 2019) and standard KWS practices for time/frequency masking.
- **Inspiration**: This repository is the direct multi-class evolution of my earlier [Spoken Digit Recognition](https://github.com/Hades-Highness/spoken-digit-recognition) project, extending its baseline architecture to handle rejection classes and robust training at scale.