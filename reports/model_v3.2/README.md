# CommandSense v3.2 - Model Card

<p align="center">
  <img src="https://img.shields.io/badge/Model%20Version-v3.2.0-7c3aed?style=for-the-badge" alt="Model Version">
  <img src="https://img.shields.io/badge/PyTorch-Suported-EE4C2C?style=for-the-badge" alt="PyTorch">
  <img src="https://img.shields.io/badge/INT8%20ONNX-SUPPORTED-green?style=for-the-badge" alt="INT8 ONNX">
  <img src="https://img.shields.io/badge/FP32%20ONNX-SUPORTED-005ced?style=for-the-badge" alt="FP32 ONNX">
  <img src="https://img.shields.io/badge/Classes-37-blue?style=for-the-badge" alt="37 Classes">
</p>
---

## ★ Model Overview

- **Model Name**: CommandSense v3.2
- **Architecture**: Residual CNN (3 residual stages) with a three-channel Log-Mel front-end
- **Base weights**: v3.0, reused as-is. **v3.2 trains nothing and owns no parameters of its own** (1,217,349 parameters, ≈4.87 MB in float32)
- **Base quantization**: v3.1 INT8 QDQ graph, reused as-is. No quantization-aware training or graph rewrite is performed.
- **Task**: Multi-class Keyword Spotting (KWS) & Spoken Command Recognition, with explicit rejection of silence, out-of-vocabulary speech **and** low-confidence predictions
- **Target Classes (37)**: `_silence_`, `_unknown_`, and the 35 command words `backward`, `bed`, `bird`, `cat`, `dog`, `down`, `eight`, `five`, `follow`, `forward`, `four`, `go`, `happy`, `house`, `learn`, `left`, `marvin`, `nine`, `no`, `off`, `on`, `one`, `right`, `seven`, `sheila`, `six`, `stop`, `three`, `tree`, `two`, `up`, `visual`, `wow`, `yes`, `zero`.
- **Artifacts**: the v3.2 release provides a unified application bundle: `CommandSense_v3.2.pth` (v3.0 weights), `CommandSense_v3.2_int8.onnx` (v3.1 INT8 graph), and `CommandSense_calibration_v3.2.json` (v3.0 calibration). All three artifacts are unchanged from their source releases.
- **Scope**: application and integration release. v3.2 ships a unified Gradio interface that serves the v3.0/v3.1 lineage through a single shared runtime, exposing both continuous microphone keyword spotting and single-clip analysis from one process. It answers the question _"how do I use this model in a real-time application?"_, not _"is the model more accurate?"_.

---

## ★ What v3.2 Changes

|  | v3.1 | v3.2 |
| :-- | :-- | :-- |
| Weights | v3.0 checkpoint, frozen | v3.0 checkpoint, frozen (same file, v3.2 filename) |
| INT8 graph | v3.1 QDQ artifact, frozen | v3.1 QDQ artifact, frozen (same file, v3.2 filename) |
| Calibration | `T = 1.311633`, `tau = 0.89` | `T = 1.311633`, `tau = 0.89` (same values) |
| Training | None | None |
| Quantization | None | None |
| Application | Inference engine only | Unified Gradio interface with shared runtime, real-time microphone spotting, and single-clip analysis |

**v3.2 is an application release, not a model release.** The decision function is identical to v3.0 and v3.1. Any difference between a v3.1 and v3.2 prediction on the same input would indicate a serving defect, not a model improvement.

---

## ★ Inherited Predictive Metrics

The following values come from the v3.0 evaluation and the v3.1 quantization report. They were not re-measured for v3.2.

| Metric | FP32 | INT8 | Notes |
| :-- | :-: | :-: | :-- |
| 35-command test accuracy | 95.8655% | 95.8655% | Inherited from v3.0; no new weights |
| Top-1 accuracy, full testing split | 96.1917% | 96.2247% | 12,105 clips, 37-class decisions |
| Top-1 agreement, INT8 vs FP32 | - | 99.55% | 54 flips in 12,105 clips |
| Coverage at `tau = 0.89` | 89.76% | 89.98% | Clips answered rather than rejected |
| Retained accuracy on answered clips | 99.50% | 99.50% | Precision at the operating point |
| Recall `_silence_` | 100.0000% | 100.0000% | 550/550 clips |
| Recall `_unknown_` | 98.5455% | 98.5455% | 542/550 clips |

**The rejection clips are drawn at load time.** The 1,100 `_silence_` / `_unknown_` evaluation clips are not stored on disk — `src/dataset.py` slices them out of the held-out recordings at load time using the global RNG. The two rejection-class recalls are therefore samples rather than constants: re-running the evaluation with a different seed moves those cells, while the 35 command columns do not move. This caveat is inherited from the v3.0 and v3.1 cards.

---

## ★ Real-time Performance

v3.2 was benchmarked on a reference six-core CPU host (Intel Core i5-9400, 2.90 GHz, 16 GB RAM, Windows 11) to establish that the unified application layer can serve real-time keyword spotting within a 500 ms hop budget.

| Stage | Mean | Median | p95 | Maximum |
| :-- | :-: | :-: | :-: | :-: |
| `FeatureFrontend` (waveform to `[1, 3, 64, 63]`) | 17.15 ms | 16.91 ms | 34.04 ms | 257.09 ms |
| ONNX inference, softmax and argmax | 4.25 ms | 3.58 ms | 8.16 ms | 39.52 ms |
| **Complete real-time window** | **21.40 ms** | **21.07 ms** | **38.84 ms** | **260.97 ms** |

The hop is 500 ms. The median window uses **4.21%** of the hop and the mean uses **4.28%** of it. The single 261 ms maximum is an operating-system scheduling outlier; the median and p95 describe normal operation.

**Feature extraction dominates the path on this host.** The INT8 graph itself accounts for roughly one fifth of the mean window time. This is consistent with the v3.1 finding that ONNX Runtime's CPU QDQ kernels are not faster than float32 for these small activations. The application benefit of INT8 is artifact size (74.1% reduction), not CPU speed.

### - Cold start and warm-up

| Metric | Median | Minimum | Maximum |
| :-- | :-: | :-: | :-: |
| Runtime construction | 230.97 ms | 189.81 ms | 495.39 ms |
| `warmup_ms` (first feature extraction + first inference) | 83.42 ms | 30.68 ms | 142.64 ms |

The first construction in a process is slower because it initialises native libraries and allocates thread pools; later constructions reuse those process-level resources. `warmup_ms` excludes Python imports and Gradio start-up. Warm-up is performed before the microphone loop starts, so the first real window does not absorb one-time kernel selection and allocation costs.

### - Thread safety

A deterministic feature tensor was classified once to establish a baseline. Eight threads then performed 25 locked predictions each, for 200 calls total.

| Observation | Result |
| :-- | :-: |
| Calls completed | 200 |
| Label mismatches against baseline | 0 |
| Confidence mismatches against baseline | 0 |
| Exceptions | 0 |
| Elapsed time | 1,294.4 ms |
| Serialized throughput | 155 calls/s |

The test establishes correctness under concurrent access, not parallel scalability. Because the inference lock serializes calls, throughput does not increase with thread count. The real-time path needs two windows per second, and the single-clip path is user initiated, so the measured serialized capacity is far above the application's demand.

---

## ★ Checkpoints & Artifacts

Model weights and quantized graphs are version-controlled and hosted separately via GitHub Releases. To use them locally, place the downloaded files into:

```
checkpoints/model_v3.2/
```

- `CommandSense_v3.2.pth` (PyTorch Weights, ≈4.87 MB) — **identical to `CommandSense_v3.0.pth`**
- `CommandSense_v3.2_int8.onnx` (ONNX Runtime Model, INT8 QDQ format) — **identical to `CommandSense_v3.1_int8.onnx`**
- `CommandSense_calibration_v3.2.json` (Calibration payload, `T = 1.311633`, `tau = 0.89`) — **identical to `CommandSense_calibration_v3.0.json`**

The `.pth` and `.onnx` files are distributed under v3.2 filenames so users do not need to fetch v3.0 and v3.1 separately. They contain the same frozen model and graph; the `.pth` files have identical SHA-256 hashes, and the ONNX graphs emit the same raw logits. The calibration payload is versioned configuration, not a weight, so it is committed with the code at `configs/calibration/CommandSense_calibration_v3.2.json` rather than with the git-ignored checkpoints.

| Repository artifact | Content |
| :-- | :-- |
| `configs/calibration/CommandSense_calibration_v3.2.json` | Inherited `T` and `tau` from v3.0, plus calibration diagnostics |
| `reports/model_v3.2/README.md` | This application-layer model card |

---

## ★ Caveats

- **No accuracy improvement.** v3.2 inherits v3.0 accuracy and v3.1 quantization fidelity. It should not be cited as a new accuracy result. The decision function is unchanged; any difference between a v3.1 and v3.2 prediction on the same input would indicate a serving defect.
- **INT8 CPU latency.** The INT8 graph is 1.68x slower than float32 on the CPU provider in the v3.1 benchmark. The application benefit is artifact size (74.1% reduction), not CPU speed.
- **Fixed 1.0 s window.** Utterances longer than one second are truncated; shorter clips are zero-padded. A command that straddles a window boundary can be split across two inferences.
- **Fixed operating point.** `tau = 0.89` was fitted on the validation split of this dataset. Coverage may change under different microphones, room acoustics, languages or speaking rates.
- **Cooldown semantics.** Repeating the same command within 1.5 s is intentionally suppressed. This is a UX choice, not a model property.
- **Single-host latency evidence.** All timings come from one six-core CPU host with a synthetic input. Mobile, WebAssembly, GPU and browser measurements are not included.
- **No end-to-end browser benchmark.** The 21 ms figure covers server-side feature extraction and inference, not upload, WebM decoding, network transport or DOM rendering.
- **Microphone contention.** The application does not arbitrate device ownership across sessions or processes. Exclusive devices can fail to open.
- **Overlapping speech.** The model does not separate overlapping talkers. Each 1.0 s window is converted to a single three-channel spectrogram and mapped to one label by a softmax classifier. When two people speak at once, their energies add in the mel domain before inference. The network has no separation mask, speaker embedding, diarization stage or multi-label head. Overlapping speech is not a safe operating condition.
- **Scope.** This is a 35-command keyword spotter with OOV rejection, not a general ASR system. It does not transcribe free speech.
- **This card describes v3.2 only.** It is written to be read on its own and makes no claim about any other release: no other version appears in it, and no metric is stated as a delta against one. Comparisons across versions live in the repository's root `README.md`, where every version sits in a single table.
