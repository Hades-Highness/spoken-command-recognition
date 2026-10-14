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

An end-to-end deep learning pipeline for spoken keyword spotting and audio command recognition, featuring a 16 kHz high-resolution three-channel Log-Mel spectrogram front-end, a dual-backend deployment architecture, and explicit rejection of silence and out-of-vocabulary speech. **v3.0** moves the whole signal front-end onto the accelerator, and **v3.1** provides a heavily optimized INT8 artifact for edge and web deployment.

_The multi-class keyword evolution of spoken-digit-recognition._

---

## - What's New

**CommandSense v3.1** is a deployment release. It retrains nothing and keeps the v3.0 weights frozen, serving as the **INT8 sibling of the v3.0 ONNX graph** via post-training static quantization.

- **Static INT8 Quantization**: Exports the v3.0 graph to QDQ format (per-channel int8 weights, int8 activations), calibrated on 500 validation clips.
- **74.1% Size Reduction**: Artifact size drops from 4.64 MB to 1.20 MB, making it highly suitable for ONNX Runtime Web and edge deployment.
- **High Fidelity**: Maintains 99.55% top-1 agreement with the float32 graph on the testing split. Discrepancies are strictly confined to low-margin ties.
- **Deployment Focus**: CPU latency is not improved (1.68x slower on CPU provider); the value proposition is purely artifact size and web compatibility.

---

## - Version Progression

Every version is trained with the same recipe (residual CNN, AdamW, batch size 256) and evaluated on the same official speaker-disjoint splits. 

> **Note:** For detailed per-class metrics, calibration analysis, confusion matrices, and version-specific caveats, refer to the linked Model Card for each release.

| Version | Key Change | Params | Test Acc (35 cmd) | Model Card |
| :-- | :-- | :-: | :-: | :-- |
| **v1.0** | Baseline: single-channel Log-Mel front-end | 1,216,259 | 94.4571% | [v1.0 card](reports/model_v1.0/README.md) |
| **v2.0** | Three-channel front-end: Log-Mel + Δ + Δ2 | 1,216,835 | 94.7842% | [v2.0 card](reports/model_v2.0/README.md) |
| **v2.1** | Rejection classes: `_silence_` + `_unknown_` | 1,217,349 | 94.1481% | [v2.1 card](reports/model_v2.1/README.md) |
| **v2.2** | Calibration release: temperature scaling + threshold | 1,217,349 | 94.1481% | [v2.2 card](reports/model_v2.2/README.md) |
| **v3.0** | Memory-optimized pipeline + conditional GPU augmentation | 1,217,349 | **95.8655%** | [v3.0 card](reports/model_v3.0/README.md) |
| **v3.1** | Deployment release: static INT8 (QDQ) quantization of v3.0 | 1,217,349 | 95.8655%* | [v3.1 card](reports/model_v3.1/README.md) |

*\*v3.1 accuracy cells are copies of v3.0 by construction, as no new weights are trained. Read the v3.1 card for quantization fidelity metrics.*

---

## - Audio Technical Specifications

- **Sampling Rate**: 16,000 Hz (mono)
- **Normalized Duration**: 1.0 second (16,000 samples)
- **Spectral Representation**: Three-Channel Log-Mel Spectrogram
  - `n_mels`: 64 | `n_fft` / `win_length`: 1024 | `hop_length`: 256
  - **Frames**: 63 | **Channels**: 3 (Log-Mel, Delta, Delta-Delta)
  - **Input Tensor Shape**: `[Batch, 3, 64, 63]`

---

## - Model Architecture

All versions share the same residual network. Since v3.0, the three-channel front-end runs on the accelerator via `AudioToThreeChannelMel` in `src/transforms.py`.

| Stage | Configuration | Output shape |
| :-- | :-- | :-- |
| Input | — | `[B, 3, 64, 63]` |
| `in_conv` | `Conv2d(3→32, 3×3)`, BN, ReLU | `[B, 32, 64, 63]` |
| `layer1` | `ResidualBlock(32→64, stride 2)` | `[B, 64, 32, 32]` |
| `layer2` | `ResidualBlock(64→128, stride 2)` | `[B, 128, 16, 16]` |
| `layer3` | `ResidualBlock(128→256, stride 2)` | `[B, 256, 8, 8]` |
| Pooling | `AdaptiveAvgPool2d((1, 1))` | `[B, 256, 1, 1]` |
| Head | Flatten → `Dropout(0.3)` → `Linear(256→N)` | `[B, N]` |

*(N is 37 for v2.1, v2.2, v3.0, and v3.1)*

---

## - Project Structure

```text
.
├── checkpoints/                  # Model weight releases (git-ignored)
│   ├── model_v3.0/               # CommandSense_v3.0.pth, CommandSense_v3.0.onnx
│   └── model_v3.1/               # CommandSense_v3.1_int8.onnx
├── configs/                      # Global hyperparameters and class mapping
│   ├── calibration/              # Calibration payloads (T and tau)
│   ├── config.py                 
│   └── labels.json               
├── data/                         # Raw datasets (git-ignored, auto-downloaded)
├── reports/                      # Versioned evaluation cards and plots
│   ├── model_v3.0/               # v3.0 training curves, report, confusion matrix
│   └── model_v3.1/               # v3.1 quantization report and model card
├── src/                          # Source code modules (dataset, transforms, models, etc.)
├── tests/                        # Pipeline and quantization smoke tests
├── app.py                        # Interactive Gradio Web UI
├── evaluate.py                   # Metrics, calibration fit and tau sweep
├── export_onnx.py                # PyTorch to ONNX exporter
├── quantize_int8.py              # v3.1 static INT8 quantization pipeline
├── requirements.txt              
└── train.py                      # Training pipeline with AMP
```

---

## - Quick Start

```bash
git clone https://github.com/Hades-Highness/spoken-command-recognition.git
cd spoken-command-recognition
pip install -r requirements.txt
```

Place the release assets into the versioned folders:
```text
checkpoints/model_v3.0/CommandSense_v3.0.pth
checkpoints/model_v3.1/CommandSense_v3.1_int8.onnx
configs/calibration/CommandSense_calibration_v3.1.json
```

```bash
python train.py       # Trains v3.0 (downloads datasets on first run)
python evaluate.py    # Fits T/tau and generates reports
python quantize_int8.py # Generates the v3.1 INT8 artifact
python app.py         # Launches the interactive Web UI
```

---

## - Rejection & Confidence

The model explicitly learns to reject via two non-command classes (`_silence_` from background noise, `_unknown_` from LibriSpeech OOV speech). Additionally, v2.2 introduced a **calibrated confidence threshold** (`tau`), refined in v3.0 (`T = 1.3116`, `tau = 0.89`). 

A clip is answered only when its calibrated top-1 probability reaches `tau`; otherwise, it is served as `_unknown_`. This buys precision (99.48% retained accuracy on answered clips) at the cost of coverage (89.76% of clips answered). See the v3.0 Model Card for the full calibration analysis.

---

## - Datasets & Citation

- **Google Speech Commands v0.02**: 105,829 clips, 35 command words + background noise (CC BY 4.0).
- **LibriSpeech**: `train-clean-100` and `dev-clean` used to build the disjoint `_unknown_` class (CC BY 4.0).

```bibtex
@article{warden2018speech,
  title={Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition},
  author={Warden, Pete},
  journal={arXiv preprint arXiv:1804.03209},
  year={2018}
}
@inproceedings{panayotov2015librispeech,
  title={LibriSpeech: An ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5206--5210},
  year={2015}
}
```

---

## - Licence

This project is released under the **MIT Licence**. The datasets keep their own terms (both CC BY 4.0). Model checkpoints published as release assets are covered by this project's MIT licence, but any model retrained or redistributed using the datasets must also respect their licences.