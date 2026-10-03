# CommandSense v1.0 - Model Card

<p align="center">
  <img src="https://img.shields.io/badge/Model%20Version-v1.0.0-7c3aed?style=for-the-badge" alt="Model Version">
  <img src="https://img.shields.io/badge/Val%20Accuracy-94.66%25-green?style=for-the-badge" alt="Validation Accuracy">
  <img src="https://img.shields.io/badge/Test%20Accuracy-94.46%25-green?style=for-the-badge" alt="Test Accuracy">
  <img src="https://img.shields.io/badge/Classes-35%20Keywords-blue?style=for-the-badge" alt="35 Classes">
</p>

---

## ★ Model Overview

* **Model Name**: CommandSense v1.0
* **Architecture**: Residual CNN (3 residual stages) with a single-channel Log-Mel front-end
* **Task**: Multi-class Keyword Spotting (KWS) & Spoken Command Recognition
* **Target Classes (35)**: `backward`, `bed`, `bird`, `cat`, `dog`, `down`, `eight`, `five`, `follow`, `forward`, `four`, `go`, `happy`, `house`, `learn`, `left`, `marvin`, `nine`, `no`, `off`, `on`, `one`, `right`, `seven`, `sheila`, `six`, `stop`, `three`, `tree`, `two`, `up`, `visual`, `wow`, `yes`, `zero`.
* **Parameters**: 1,216,259 (≈4.87 MB in float32)
* **Checkpoint**: `CommandSense_v1.0.pth` (PyTorch) and `CommandSense_v1.0.onnx` (ONNX Runtime, opset 17), attached to the [v1.0.0 release](https://github.com/Hades-Highness/spoken-command-recognition/releases/tag/v1.0.0)
* **Scope**: 35-class keyword spotting on Google Speech Commands v0.02, trained with the project's standard recipe (residual CNN, AdamW, 20 epochs, official splits) over a single-channel spectral front-end.

---

## ★ Technical Specifications & Preprocessing

* **Audio Sampling Rate ($f_s$)**: 16,000 Hz (Mono)
* **Clip Duration**: 1.0 second (16,000 samples, zero-padded or truncated)
* **Spectral Representation**: Single-Channel Log-Mel Spectrogram
  * `n_mels`: 64
  * `n_fft` / `win_length`: 1024 (~64 ms window)
  * `hop_length`: 256 (~16 ms step)
  * **Frames**: 63
  * **Channels**: 1 — Log-Mel
  * **Scaling**: Logarithmic energy normalization, $\log(S + 10^{-6})$
* **Input Tensor Shape**: `[Batch, 1, 64, 63]`

The front-end computes one 64-band Log-Mel channel per frame, so the network receives static spectral energy without explicit derivative channels.

---

## ★ Training Configuration

| Setting | Value |
| :--- | :--- |
| Epochs | 20 |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Schedule | None |
| Loss | `CrossEntropyLoss` |
| Batch size | 256 |
| Precision | Mixed precision (AMP) |
| Seed | 42 (Python, NumPy, PyTorch, CUDA) |
| Feature caching | Full split cached in RAM as single-channel spectrograms |
| Checkpoint selection | Best validation accuracy |
| `Ctrl+C` handling | Writes the partial history and curves |

---

## ★ Training Logs & Epoch Summary

The complete per-epoch loss and accuracy metrics are stored in [`history_v1.0.json`](history_v1.0.json).

| Parameter / Metric | Value |
| :--- | :--- |
| **Best Epoch** | Epoch 16 |
| **Best Validation Accuracy** | **94.66%** |
| **Validation Loss (Best Epoch)** | `0.2177` |
| **Validation Accuracy (Epoch 1)** | `75.24%` |
| **Final Training Accuracy (Epoch 20)** | `99.10%` |
| **Final Training Loss (Epoch 20)** | `0.0303` |
| **Final Validation Accuracy (Epoch 20)** | `94.22%` |
| **Final Validation Loss (Epoch 20)** | `0.2563` |
| **Train − Val Gap (Epoch 20)** | `4.88 pts` |

### - Training & Validation Curves

![Training Curves](training_curves_v1.0.png)

Validation accuracy rises from 75.24% at epoch 1 to 94.66% at epoch 16, then oscillates inside the 93.2–94.7% band for the remaining epochs. Training accuracy continues to 99.10%, so the train/val gap widens over the run rather than closing.

---

## ★ Test Set Results

Evaluated on the official `testing_list.txt` split: **11,005 clips**, speaker-disjoint from training and validation.

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **94.46%** (10395 / 11005, i.e. 94.4571%) |
| **Macro-Average Precision** | `0.9465` |
| **Macro-Average Recall** | `0.9365` |
| **Macro-Average F1** | `0.9406` |
| **Weighted-Average F1** | `0.9445` |
| **Standard Error on Accuracy** | ≈0.22 pt (95% CI ≈ ±0.43 pt) |

Full report: [`report_v1.0.txt`](report_v1.0.txt).

### - Confusion Matrix

![Confusion Matrix](confusion_matrix_v1.0.png)

### - Per-Class Results

| Command | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | :---: |
| backward | 0.9877 | 0.9697 | 0.9786 | 165 |
| bed | 0.9261 | 0.9082 | 0.9171 | 207 |
| bird | 1.0000 | 0.8811 | 0.9368 | 185 |
| cat | 0.9095 | 0.9330 | 0.9211 | 194 |
| dog | 0.9757 | 0.9136 | 0.9437 | 220 |
| down | 0.9612 | 0.9163 | 0.9382 | 406 |
| eight | 0.9240 | 0.9828 | 0.9525 | 408 |
| five | 0.9751 | 0.9663 | 0.9707 | 445 |
| follow | 0.9536 | 0.8372 | 0.8916 | 172 |
| forward | 0.9241 | 0.8645 | 0.8933 | 155 |
| four | 0.9466 | 0.9300 | 0.9382 | 400 |
| go | 0.9187 | 0.9279 | 0.9233 | 402 |
| happy | 0.9646 | 0.9409 | 0.9526 | 203 |
| house | 0.9113 | 0.9686 | 0.9391 | 191 |
| learn | 0.9429 | 0.8199 | 0.8771 | 161 |
| left | 0.9218 | 0.9733 | 0.9469 | 412 |
| marvin | 0.9442 | 0.9538 | 0.9490 | 195 |
| nine | 0.9467 | 0.9583 | 0.9525 | 408 |
| no | 0.8700 | 0.9753 | 0.9197 | 405 |
| off | 0.9891 | 0.9005 | 0.9427 | 402 |
| on | 0.9199 | 0.9571 | 0.9381 | 396 |
| one | 0.9662 | 0.9323 | 0.9490 | 399 |
| right | 0.8924 | 0.9848 | 0.9364 | 396 |
| seven | 0.9974 | 0.9581 | 0.9774 | 406 |
| sheila | 0.8918 | 0.9717 | 0.9300 | 212 |
| six | 0.9796 | 0.9772 | 0.9784 | 394 |
| stop | 0.9692 | 0.9951 | 0.9820 | 411 |
| three | 0.9273 | 0.9136 | 0.9204 | 405 |
| tree | 0.9066 | 0.8549 | 0.8800 | 193 |
| two | 0.9181 | 0.9788 | 0.9475 | 424 |
| up | 0.9154 | 0.9671 | 0.9405 | 425 |
| visual | 0.9936 | 0.9455 | 0.9689 | 165 |
| wow | 0.9844 | 0.9175 | 0.9497 | 206 |
| yes | 0.9951 | 0.9737 | 0.9843 | 419 |
| zero | 0.9774 | 0.9306 | 0.9534 | 418 |
| **accuracy** | | | **0.9446** | **11005** |
| macro avg | 0.9465 | 0.9365 | 0.9406 | 11005 |
| weighted avg | 0.9461 | 0.9446 | 0.9445 | 11005 |

* **Best classes by F1**: `yes` (0.9843), `stop` (0.9820), `backward` (0.9786), `six` (0.9784), `seven` (0.9774)
* **Weakest classes by F1**: `learn` (0.8771), `tree` (0.8800), `follow` (0.8916), `forward` (0.8933), `bed` (0.9171)
* **Worst recall**: `learn` (0.8199), `follow` (0.8372), `tree` (0.8549), `forward` (0.8645), `bird` (0.8811)
* **Worst precision**: `no` (0.8700), `sheila` (0.8918), `right` (0.8924), `tree` (0.9066), `cat` (0.9095)

The errors concentrate on acoustically similar words rather than spreading at random. `learn`, `follow` and `tree` are the least recalled, and `follow` / `forward` sit on the same opening syllable. `no` is the least precise class (0.8700) while `bird`, `right` and `no` themselves reach high recall at lower precision, which is the signature of a class absorbing neighbours rather than of diffuse error. Note that `bird` has perfect precision (1.0000) at 0.8811 recall — it is never over-predicted, only missed.

---

## ★ Checkpoints & Artifacts

Model weights are version-controlled and hosted separately via [GitHub Releases](https://github.com/Hades-Highness/spoken-command-recognition/releases/tag/v1.0.0). To use them locally, place the downloaded files into:

```text
checkpoints/model_v1.0/
```

* `CommandSense_v1.0.pth` (PyTorch Weights, ≈4.87 MB)
* `CommandSense_v1.0.onnx` (ONNX Runtime Model, opset 17)

| Repository artifact | Content |
| :--- | :--- |
| `reports/model_v1.0/history_v1.0.json` | 20 epochs of train/val loss and accuracy |
| `reports/model_v1.0/training_curves_v1.0.png` | Loss and accuracy curves, 300 dpi |
| `reports/model_v1.0/confusion_matrix_v1.0.png` | 35×35 test-set confusion matrix, 300 dpi |
| `reports/model_v1.0/report_v1.0.txt` | Per-class precision / recall / F1 on the test split |
| `configs/labels.json` | Index → class-name mapping (35 entries) |

---

## ★ Caveats & Known Issues

* **Single run.** The seed is fixed at 42, but this version was trained once, so run-to-run variance is unmeasured and any difference below ~0.4 pt cannot be interpreted.
* **No repeated-run or significance testing.** The figures above come from one training run and one evaluation pass.
* **No rejection capability.** There is no `_unknown_` or `_silence_` class and no confidence threshold, so the model returns one of the 35 labels for any input, including silence or noise.
* **Fixed 1-second window.** Longer clips are truncated from the start and shorter ones zero-padded; there is no voice-activity detection, so a command spoken late in a recording is cut off.
* **Vocabulary is the 35 words only.** Out-of-vocabulary words are forced into the nearest known class.
* **Checkpoint channel requirement.** This checkpoint was trained with a single-channel front-end, so it must be loaded into a `CommandSense` instantiated with **1 input channel** and 35 classes. Loading it into a differently shaped network raises a state-dict mismatch rather than failing silently.
* **RAM footprint.** Training caches the full split as single-channel float32 features: ≈1.4 GB for the training split (84,843 clips × 1 × 64 × 63).
* **Trained on clean, 16 kHz, single-word, single-speaker English recordings.** Performance on noisy, far-field or conversational audio is not characterised by the numbers above.
