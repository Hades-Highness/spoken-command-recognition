# CommandSense - Spoken Command Recognition

<p align="center">
  <a href="https://github.com/Hades-Highness/spoken-command-recognition/releases/tag/v1.0.0">
    <img src="https://img.shields.io/github/v/release/Hades-Highness/spoken-command-recognition?color=7c3aed&label=Latest-Release&style=for-the-badge" alt="Latest Release">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ONNX%20Runtime-Supported-005ced?style=for-the-badge" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

An end-to-end deep learning pipeline for spoken keyword spotting and audio command recognition across **35 distinct classes**, featuring a 16 kHz high-resolution Log-Mel spectrogram front-end and a dual-backend deployment architecture reaching **~94.66% validation accuracy**.

*The multi-class keyword evolution of [spoken-digit-recognition](https://github.com/Hades-Highness/spoken-digit-recognition).*

---

## - Audio Technical Specifications (v1.0 Final)

* **Sampling Rate ($f_s$)**: 16,000 Hz (mono)
* **Normalized Duration**: 1.0 second (16,000 samples)
* **Spectral Representation**: Single-Channel Log-Mel Spectrogram
  * `n_mels`: 64
  * `n_fft`: 1024 (~64 ms window)
  * `hop_length`: 256 (~16 ms step)
  * **Normalization**: Per-Sample Log-Mel Scaling ($\log(S + 10^{-6})$)
  * **Input Tensor Shape**: `[Batch, 1, 64, 63]`

---

## - Note on Model Versions & Source Code

The current repository source code and the interactive web interface **(`app.py`)** are optimized **exclusively** for **CommandSense v1.0** (35 keyword classes, 16 kHz Log-Mel Spectrograms, achieving 94.66% validation accuracy).

Model artifacts **(`.pth` and `.onnx`)** for official releases are available in the **GitHub Releases** section to keep the source repository lightweight and version-controlled.

---

## - Version History & Project Evolution

### ★ Version 1.0 — Speech Commands Benchmark (35 Classes)
* **Dataset Scope**: Google Speech Commands Dataset v0.02 (35 distinct spoken keyword classes).
* **Target Classes**: `backward`, `bed`, `bird`, `cat`, `dog`, `down`, `eight`, `five`, `follow`, `forward`, `four`, `go`, `happy`, `house`, `learn`, `left`, `marvin`, `nine`, `no`, `off`, `on`, `one`, `right`, `seven`, `sheila`, `six`, `stop`, `three`, `tree`, `two`, `up`, `visual`, `wow`, `yes`, `zero`.
* **Pipeline Upgrades**:
  1. **Dual Inference Engine**: Implemented seamless execution switching between PyTorch (`.pth`) and ONNX Runtime (`.onnx`) inside `src/inference.py`.
  2. **In-RAM Dataset Caching**: Cached entire spectrogram dataset in memory during training to eliminate disk I/O bottlenecks.
  3. **Interactive UI**: Multi-backend Gradio web application supporting live microphone capture and `.wav` file uploads.
* **Outcome**: Achieved **94.66% validation accuracy** with stable convergence and low validation loss.
* **Model Card & Details**: [View Full v1.0 Report & Curves](reports/model_v1.0/README.md).

---

## - Performance & Progression Summary

| Version | Dataset Composition | Target Classes | Best Val Acc | Backends | Model Card / Curves |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **v1.0** | 100% Speech Commands v0.02 | 35 Classes | **94.66%** | `.pth` / `.onnx` | [📄 Model Card v1.0](reports/model_v1.0/README.md) |

---

## - Project Structure

```text
.
├── checkpoints/              # Model weight releases (CommandSense_v1.0.pth, CommandSense_v1.0.onnx)
│   └── model_v1.0/
├── configs/                  # Global hyperparameter configurations and system paths
│   └── config.py
├── data/                     # Raw Speech Commands dataset archives and WAV files
├── reports/                  # Versioned evaluation cards, history metrics, and plots
│   └── model_v1.0/
│       ├── README.md         # Detailed Model Card for v1.0
│       ├── confusion_matrix.png
│       ├── history.json
│       └── training_curves.png
├── src/                      # Source code modules
│   ├── dataset.py            # Dataset reader, Log-Mel extraction & RAM caching
│   ├── inference.py          # Dual-backend (ONNX & PyTorch) inferencer wrapper
│   ├── models.py             # CommandSense CNN Neural Network Architecture
│   └── utils.py              # Audio preprocessing utilities
├── app.py                    # Interactive Gradio Web UI
├── evaluate.py               # Classification report & confusion matrix generator
├── export_onnx.py            # PyTorch model to ONNX exporter
├── requirements.txt          # Python environment dependencies
└── train.py                  # Training pipeline with AMP and graceful termination