# CommandSense - Spoken Command Recognition

<p align="center">
  <a href="https://github.com/Hades-Highness/spoken-command-recognition/releases/tag/v2.0.0">
    <img src="https://img.shields.io/github/v/release/Hades-Highness/spoken-command-recognition?color=7c3aed&label=Latest-Release&style=for-the-badge" alt="Latest Release">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ONNX%20Runtime-Supported-005ced?style=for-the-badge" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

An end-to-end deep learning pipeline for spoken keyword spotting and audio command recognition across **35 distinct classes**, featuring a 16 kHz high-resolution three-channel Log-Mel spectrogram front-end and a dual-backend deployment architecture reaching **94.77% validation accuracy** and **94.78% test accuracy**.

*The multi-class keyword evolution of [spoken-digit-recognition](https://github.com/Hades-Highness/spoken-digit-recognition).*

---

## - Note on Model Versions & Source Code

The current repository source code and the interactive web interface **(`app.py`)** are optimized **exclusively** for **CommandSense v2.0** (35 keyword classes, 3-channel 16 kHz spectrograms, achieving 94.77% validation accuracy).

Model artifacts **(`.pth` and `.onnx`)** for official releases are available in the **GitHub Releases** section to keep the source repository lightweight and version-controlled. The v1.0.0 checkpoint is still downloadable from its own release, and its results remain in the repository under `reports/model_v1.0/`.

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

## - Version History & Project Evolution

### ★ Version 1.0 — Speech Commands Benchmark (35 Classes)
* **Dataset Scope**: Google Speech Commands Dataset v0.02 (35 distinct spoken keyword classes).
* **Single-Channel Front-End**: 16 kHz Log-Mel spectrogram, input tensor `[Batch, 1, 64, 63]`.
* **Pipeline Upgrades**:
  1. **Dual Inference Engine**: PyTorch (`.pth`) and ONNX Runtime (`.onnx`) inference backends.
  2. **In-RAM Dataset Caching**: the whole spectrogram dataset is cached in memory during training to remove disk I/O bottlenecks.
  3. **Interactive UI**: Gradio web application supporting live microphone capture and `.wav` file uploads.
* **Outcome**: **94.66% validation accuracy** and **94.4571% test accuracy**.
* **Model Card & Details**: [View Full v1.0 Report & Curves](reports/model_v1.0/README.md).


### ★ Version 2.0 — Three-Channel Spectral Front-End
* **Dataset Scope**: Google Speech Commands Dataset v0.02 (35 distinct spoken keyword classes).
* **Target Classes**: `backward`, `bed`, `bird`, `cat`, `dog`, `down`, `eight`, `five`, `follow`, `forward`, `four`, `go`, `happy`, `house`, `learn`, `left`, `marvin`, `nine`, `no`, `off`, `on`, `one`, `right`, `seven`, `sheila`, `six`, `stop`, `three`, `tree`, `two`, `up`, `visual`, `wow`, `yes`, `zero`.
* **Single-Variable Change**: The input representation was replaced with a three-channel spectrogram (Log-Mel + $\Delta$ + $\Delta^2$) so the network sees spectral dynamics as well as static energy. The network, the hyperparameters and the splits are identical to v1.0.0, which makes the v1 → v2 comparison attributable to the features alone.
* **Additional Work**:
  1. **Test-Set Evaluation**: `evaluate.py` now reports per-class precision / recall / F1 and the confusion matrix on the official `testing_list.txt` split (11,005 speaker-disjoint clips).
  2. **Reproducibility Guard-Rails**: global seed (42), committed class mapping (`configs/labels.json`), and versioned output folders for checkpoints and reports.
  3. **Unified Preprocessing**: a single feature path shared by training, evaluation and inference, with a fallback decoder for browser-recorded audio (WebM/Opus).
  4. **Hardened ONNX Backend**: the exported graph is validated at export time **and** at load time (input rank, channel count, frame count), so a stale export fails loudly instead of predicting silently.
* **Outcome**: **94.77% validation accuracy** and **94.7842% test accuracy** (macro-average F1 0.9436).
* **Model Card & Details**: [View Full v2.0 Report & Curves](reports/model_v2.0/README.md).

---

## - Performance & Progression Summary

All figures below come from the committed artifacts in `reports/`. Both versions were trained for 20 epochs with the same network, optimizer and splits; the only difference is the input representation.

| Version | Params | Best Val Acc | Test Acc | Test Macro F1 | Training Curves |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **v1.0** | 1,216,259 | 94.66% | 94.4571% | 0.9406 | ![v1.0 curves](reports/model_v1.0/training_curves_v1.0.png) |
| **v2.0** | 1,216,835 | **94.77%** | **94.7842%** | **0.9436** | ![v2.0 curves](reports/model_v2.0/training_curves_v2.0.png) |

* Val: official `validation_list.txt` (9,981 clips). Test: official `testing_list.txt` (11,005 clips). Both splits are speaker-disjoint by construction. Test accuracies are quoted to four decimals because the v1 → v2 difference is smaller than the rounding step.

<details>
<summary>Per-run detail</summary>

| Metric | v1.0 | v2.0 |
| :--- | :---: | :---: |
| Best validation accuracy | 94.66% (epoch 16) | 94.77% (epoch 17) |
| Correct validation clips | 9448 / 9981 | 9459 / 9981 |
| Validation loss at best epoch | 0.2177 | 0.2156 |
| Validation accuracy at epoch 1 | 75.24% | 84.17% |
| Test accuracy | 94.4571% (10395 / 11005) | 94.7842% (10431 / 11005) |
| Test macro-avg F1 | 0.9406 | 0.9436 |
| Final (epoch 20) training accuracy | 99.10% | 99.12% |
| Final (epoch 20) validation accuracy | 94.22% | 93.86% |
| Train − val accuracy gap (epoch 20) | 4.88 pts | 5.27 pts |

Histories: [`history_v1.0.json`](reports/model_v1.0/history_v1.0.json) · [`history_v2.0.json`](reports/model_v2.0/history_v2.0.json).
Test reports: [`report_v1.0.txt`](reports/model_v1.0/report_v1.0.txt) · [`report_v2.0.txt`](reports/model_v2.0/report_v2.0.txt).

</details>

### How large is the improvement

On the test split, v2.0 corrects **36 more clips out of 11,005**, i.e. **+0.3271 points** (94.4571% → 94.7842%); on validation it corrects 11 more clips out of 9,981, i.e. **+0.1102 points**.

The standard error of a single accuracy estimate at this level on 11,005 clips is ≈0.21 points, and the standard error of the *unpaired difference* between the two versions is ≈0.30 points. The observed +0.33-point test gap is therefore larger than the noise on each estimate but smaller than the noise on their difference. Since both models were evaluated on the same clips, the correct test is a paired one (McNemar) over per-clip predictions, which were not retained. **The honest reading is that v2.0 is a small, directionally consistent improvement, not a decisive one.**

What the run does show clearly is where the remaining headroom is: both versions reach ~99.1% training accuracy while validation sits ~5 points lower, and both converge to the same ~94.8% plateau — even though v2.0 starts 8.93 points higher at epoch 1 (84.17% vs 75.24%). The model is not short of input representation; it is fitting the training set and stopping. The next gain has to come from regularization and augmentation, which is what v3.0.0 targets in the [Roadmap](#roadmap).

---

## - Model Architecture

Both versions share the same residual network; only the first convolution differs.

| Stage | Configuration | Output shape |
| :--- | :--- | :--- |
| Input | — | `[B, 3, 64, 63]` |
| `in_conv` | `Conv2d(3→32, 3×3)`, BN, ReLU | `[B, 32, 64, 63]` |
| `layer1` | `ResidualBlock(32→64, stride 2)` | `[B, 64, 32, 32]` |
| `layer2` | `ResidualBlock(64→128, stride 2)` | `[B, 128, 16, 16]` |
| `layer3` | `ResidualBlock(128→256, stride 2)` | `[B, 256, 8, 8]` |
| Pooling | `AdaptiveAvgPool2d((1, 1))` | `[B, 256, 1, 1]` |
| Head | Flatten → `Dropout(0.3)` → `Linear(256→35)` | `[B, 35]` |

Each `ResidualBlock` is two 3×3 convolutions with batch normalization and a 1×1 projection shortcut where the shape changes. Parameter count: **1,216,835** for v2.0 versus **1,216,259** for v1.0 — the difference is the 576 extra weights of the first convolution.

---

## - Project Structure

```text
.
├── checkpoints/                  # Model weight releases
│   ├── model_v1.0/
│   └── model_v2.0/               
├── configs/                      # Global hyperparameters, paths and the class mapping
│   ├── config.py
│   └── labels.json               # Index -> class name mapping 
├── data/                         # Raw Speech Commands dataset 
├── reports/                      # Versioned evaluation cards, histories, reports and plots
│   ├── model_v1.0/
│   │   ├── README.md             # Detailed Model Card for v1.0
│   │   ├── confusion_matrix_v1.0.png
│   │   ├── history_v1.0.json
│   │   ├── report_v1.0.txt        # Test-set classification report
│   │   └── training_curves_v1.0.png
│   └── model_v2.0/
│       ├── README.md             # Detailed Model Card for v2.0
│       ├── confusion_matrix_v2.0.png
│       ├── history_v2.0.json
│       ├── report_v2.0.txt        # Test-set classification report
│       └── training_curves_v2.0.png
├── src/                          # Source code modules
│   ├── dataset.py                # Speech Commands reader, official splits & RAM caching
│   ├── inference.py              # Dual-backend (ONNX & PyTorch) inferencer
│   ├── models.py                 # AudioFeatureExtractor + ResidualBlock + CommandSense
│   └── utils.py                  # Audio loading and preprocessing (single source of truth)
├── app.py                        # Interactive Gradio Web UI
├── evaluate.py                   # Classification report & confusion matrix generator
├── export_onnx.py                # PyTorch to ONNX exporter with strict shape validation
├── requirements.txt              # Python environment dependencies
├── train.py                      # Training pipeline with AMP and graceful termination
└── LICENCE                       # MIT
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
checkpoints/model_v2.0/CommandSense_v2.0.pth
checkpoints/model_v2.0/CommandSense_v2.0.onnx 
```

#### - Launch interactive Web UI
```text
python app.py
```

#### - Reproduce the published metrics
```text
python train.py       # trains v2.0, writes reports/model_v2.0/ and checkpoints/model_v2.0/
python evaluate.py    # writes reports/model_v2.0/report_v2.0.txt + confusion matrix
```

---

## - Rejection & Confidence

The model currently returns one of the 35 command labels for any input, including silence or noise, because it has no dedicated rejection class and no confidence threshold. Adding `_unknown_` / `_silence_` classes and a calibrated rejection threshold is version 2.1.0 and 2.2.0 in the roadmap; until then, treat the reported accuracy as applying to in-vocabulary, clean, 1-second recordings.

---

## - Datasets & Citation

| Dataset | Content | Source rate | Licence |
| :--- | :--- | :---: | :--- |
| [Google Speech Commands v0.02](https://www.tensorflow.org/datasets/catalog/speech_commands) | 105,829 clips, 35 command words + background noise, 2,618 speakers | 16 kHz | CC BY 4.0 |

Loaded through `torchaudio.datasets.SPEECHCOMMANDS`. The **training / validation / testing** split comes from the dataset's own `validation_list.txt` (9,981 clips) and `testing_list.txt` (11,005 clips), which are speaker-disjoint by construction — this is why the validation and test numbers in this repository agree so closely, and why they are not inflated by speaker identity.

The `subset` argument only accepts `training`, `validation` and `testing`; the `_unknown_`, `_silence_` and `_background_noise_` folders must therefore be indexed manually (planned for v2.1.0).

```bib
@article{warden2018speech,
  title   = {Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition},
  author  = {Warden, Pete},
  journal = {arXiv preprint arXiv:1804.03209},
  year    = {2018},
  url     = {https://arxiv.org/abs/1804.03209}
}
```

The dataset is released under **CC BY 4.0**, so attribution is required when the data is redistributed or used in published work.

---

## - Licence

This project is released under the **MIT Licence**. The Speech Commands dataset keeps its own terms (CC BY 4.0). Model checkpoints published as release assets are covered by this project's MIT licence, but any model retrained or redistributed using the dataset must also respect the dataset's licence.
