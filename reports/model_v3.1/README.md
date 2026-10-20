# CommandSense v3.1 - Model Card

<p align="center">
  <img src="https://img.shields.io/badge/Model%20Version-v3.1.0-7c3aed?style=for-the-badge" alt="Model Version">
  <img src="https://img.shields.io/badge/PyTorch-Supported-EE4C2C?style=for-the-badge" alt="PyTorch">
  <img src="https://img.shields.io/badge/INT8%20ONNX-SUPPORTED-green?style=for-the-badge" alt="INT8 ONNX">
  <img src="https://img.shields.io/badge/FP32%20ONNX-SUPPORTED-005ced?style=for-the-badge" alt="FP32 ONNX">
  <img src="https://img.shields.io/badge/Classes-37-blue?style=for-the-badge" alt="37 Classes">
</p>

---

## ★ Model Overview

* **Model Name**: CommandSense v3.1
* **What it is**: a **deployment release, not a new model**. v3.1 takes the frozen v3.0 weights, re-exports them to ONNX and produces the **INT8 sibling of that graph**. It trains nothing, adds no layer, no parameter and no hyperparameter, and no gradient step is taken anywhere in the release.
* **Architecture**: unchanged from v3.0 - a residual CNN with three residual stages (`in_conv` + `layer1..3` + `AdaptiveAvgPool2d` + `Dropout(0.3)` + `Linear`) behind the three-channel Log-Mel front-end, **1,217,349 parameters**. The exported float32 graph reports **1,215,973 initializers** because the exporter folds batch normalization, so the two counts are not the same quantity; the graph occupies 4.643 MB against 4.675 MB for the `.pth` checkpoint the weights came from.
* **Weights**: `CommandSense_v3.0.pth`, opened read-only. No fine-tuning and no quantization-aware training: this is post-training quantization.
* **Front-end**: `AudioToThreeChannelMel` (`src/transforms.py`), the same `[3, 64, 63]` tensor that training and serving use; the calibration pass feeds it clip by clip instead of reimplementing it.
* **Task**: unchanged - 37 classes (35 command words, `_silence_`, `_unknown_`) with temperature scaling and confidence-threshold rejection applied on top of the raw logits.
* **Artifacts** (build outputs under the git-ignored `checkpoints/model_v3.1/`):
  * `CommandSense_v3.1_fp32.onnx` - the re-exported float32 reference the quantizer needs and the baseline every number below is measured against,
  * `CommandSense_v3.1_int8.onnx` - the shipped graph: QDQ, per-channel int8 weights, int8 activations, with the operating point embedded in its `metadata_props`,
  * `configs/calibration/CommandSense_calibration_v3.1.json` - the v3.1 calibration payload, versioned configuration and tracked in git,
  * `reports/model_v3.1/quantization_report.txt` - the run this card summarises.
* **Scope**: this card documents the quantization of the v3.0 weights and the operating point measured on it. It reports v3.1 in isolation and states no delta against any other release. No accuracy *improvement* is claimed: the INT8 graph's accuracy is the v3.0 accuracy plus the 0.45 % of decisions that quantization moves.

---

## ★ Quantization Recipe

v3.1 is **post-training static quantization** with ONNX Runtime, in the QDQ format:

| Setting | Value |
| :--- | :--- |
| Method | ONNX Runtime post-training static quantization (`quantize_static`) |
| Format | QDQ (`QuantizeLinear` / `DequantizeLinear` pairs), the form ONNX Runtime Web executes natively |
| Weight type | `QInt8` |
| Activation type | `QInt8` |
| Per-channel | `True` (weights) |
| Calibration method | `MinMax` |
| Calibration split | `validation`, 500 of 10,979 clips, RNG seed `42` |
| Observed input range | `[-13.8155, 11.0512]` |
| Exported opset | 17, dynamic batch axis |

* **The calibration clips go through the deployed front-end.** Each clip is decoded, resampled and turned into `[3, 64, 63]` features by `AudioToThreeChannelMel`, then handed to the quantizer one clip at a time, so the recorded ranges describe the tensor the graph will actually receive.
* **`testing` never calibrates anything.** It is the measurement split: it fed no weight, no activation range and no `T`/`tau`, and none of its numbers is written back into the artifact.
* **One split is used twice, and that is stated rather than hidden.** `validation` supplies both the activation ranges and the `T`/`tau` that `evaluate.py` fitted for v3.0. That reuse is why the v3.0 operating point is re-measured on `testing` instead of being copied on faith.
* **The run is reproducible.** `set_seed(42)` fixes the split draw - including the synthetic `_silence_` / `_unknown_` clips the dataset slices on the fly - before calibration, so rebuilding from the same checkpoint reproduces the same ranges and the same calibration decision.
* **Nothing downstream of the graph is changed.** The graph keeps emitting raw logits and `T`/`tau` stay a runtime concern of `src/inference.py`, so the threshold can still be changed without a re-export.

---

## ★ Fidelity (INT8 against float32, whole `testing` split)

Both graphs score all 12,105 clips of `testing` (11,005 commands + 1,100 rejection clips) through the same features.

| Metric | Value |
| :--- | :--- |
| Top-1 agreement | **99.55 %** (54 of 12,105 clips keep the float32 label) |
| Mean abs logit deviation | 0.193094 |
| Max abs logit deviation | 14.466805 (informational, see below) |
| Mean abs softmax deviation | 0.000201 |
| Max abs softmax deviation | 0.172401 |

| Per-class agreement | Value | Clips | Flips |
| :--- | ---: | ---: | ---: |
| 35 commands (combined) | 99.52 % | 11,005 | 53 |
| `_silence_` | 100.00 % | 550 | 0 |
| `_unknown_` | 99.82 % | 550 | 1 |

28 of the 37 classes carry at least one flip and 9 carry none. The flips are spread rather than concentrated: `zero` (5), `forward` (4), `go`, `on`, `one`, `three`, `tree` (3 each), `bird`, `cat`, `dog`, `down`, `five`, `sheila`, `up` (2 each), and 13 classes with a single flip (`_unknown_`, `bed`, `eight`, `four`, `happy`, `left`, `no`, `off`, `right`, `six`, `stop`, `two`, `yes`). No command class loses more than 5 of its clips, and the two rejection classes stay intact.

**Why the logit deviation is not a tolerance.** This checkpoint's logits span roughly +/-40, so a single int8 step is already ~0.1-0.4 logit units and the observed max deviation stays above 0.05 for every configuration tried - MinMax, Entropy and Percentile calibration, per-channel on and off, 50 against 500 calibration clips, a float32 head, and dynamic quantization alike. Any near-zero bound therefore sits below the quantizer's resolution and is unreachable. The claim this release can support is the decision-space one, and `tests/test_v3_1_quantization.py` asserts exactly those bounds, with its agreement floor derived from the measurement above instead of picked as a round number.

**The flips sit on near ties.** The clips that change label are ones the float32 graph itself decides by a negligible top-1/top-2 softmax margin, and the temperature/threshold stage downstream serves or rejects them on confidence rather than on that margin.

---

## ★ Operating Point

v3.0 ships `T = 1.311633` and `tau = 0.89`. Quantization is expected to move the decision boundary slightly, so the pair is re-measured on the whole `testing` split before it is reused:

| Metric (whole `testing` split) | float32 graph | INT8 graph | Delta |
| :--- | ---: | ---: | ---: |
| coverage | 89.76 % | 89.98 % | +0.21 pt |
| retained accuracy within coverage | 99.50 % | 99.50 % | -0.01 pt |
| top-1 accuracy (all clips) | 96.1917 % | 96.2247 % | +0.03 pt |
| FRR (commands wrongly rejected) | 7.15 % | 6.96 % | -0.18 pt |
| FAR (rejection classes wrongly served) | 11.71 % | 12.04 % | +0.32 pt |
| answered clips | 10,866 | 10,892 | +26 |

The coverage gap is **0.21 pt against a 0.5 pt tolerance**, so the v3.0 pair transfers and **v3.1 ships the same `T` and `tau`**. Re-fitting the temperature on the INT8 logits would give `T = 1.320859` with the same `tau = 0.89`, i.e. the same decision within noise; the re-fit branch exists in `quantize_int8.py` and was not taken. The remaining cells move by well under a point and in both directions, and none of them is an accuracy gain.

### ★ Calibration quality

NLL, ECE and MCE before (`T = 1`) and after temperature scaling, 15 bins:

| Split | Graph | NLL before | NLL after | ECE before | ECE after | MCE before | MCE after |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `validation` (10,979) | int8 | 0.139424 | 0.132338 | 0.009438 | 0.005422 | 0.164855 | 0.091434 |
| `validation` (10,979) | float32 | 0.139316 | 0.132453 | 0.009217 | 0.006232 | 0.168483 | 0.138064 |
| `testing` (12,105) | int8 | 0.133002 | 0.127715 | 0.010026 | 0.004124 | 0.163825 | 0.070392 |
| `testing` (12,105) | float32 | 0.132755 | 0.127788 | 0.010113 | 0.005558 | 0.192082 | 0.125716 |

Temperature scaling improves every column on both graphs. On the held-out split the INT8 graph ends up marginally better calibrated than float32 (ECE 0.004124 against 0.005558), which is quantization smoothing the over-confident tail rather than an argument that INT8 is preferable.

### ★ Calibration artifact

* `configs/calibration/CommandSense_calibration_v3.1.json` is the payload `src/inference.py` resolves for `v3.1` through `calibration.resolve_calibration_file("v3.1")`, a per-version lookup, so v3.0 keeps serving its own file.
* The payload records `threshold_criterion = "min_cost"`, `n_bins = 15`, `model_version = "v3.1"` and its own provenance; `calibration_samples = 10,979` refers to the split the `T`/`tau` pass walked, while the activation ranges came from the 500-clip subset named at the top.
* The same payload is stamped into the graph as **17 `commandsense.calibration.*` entries** (`model_name`, `model_version`, `created_utc`, `temperature`, `confidence_threshold`, `reject_label`, `n_bins`, `ece_before`, `ece_after`, `nll_before`, `nll_after`, `nll_before_train`, `threshold_criterion`, `calibration_split`, `calibration_samples`, `source_checkpoint`, `notes`). Reading them back yields `T = 1.311633`, `tau = 0.89`, so a single-file runtime can recover the operating point from the graph alone, and `tests/test_v3_1_quantization.py` proves the graph and the JSON agree.

---

## ★ Size and Latency

| Artifact | Size | Compression |
| :--- | ---: | :--- |
| `CommandSense_v3.1_fp32.onnx` | 4.643 MB | reference |
| `CommandSense_v3.1_int8.onnx` | **1.202 MB** | **74.1 % smaller (3.86x)** |

**That is the whole win.** Latency was measured on the same host and is reported as it came out:

| Graph | Median | Min | Max | Against float32 |
| :--- | ---: | ---: | ---: | :--- |
| float32 | 1.614 ms/clip | 1.518 ms | 1.713 ms | reference |
| INT8 | 2.718 ms/clip | 2.541 ms | 3.455 ms | 1.68x slower |

Measurement: 5 rounds of 200 clips, one clip per call, `CPUExecutionProvider`, median of the rounds. Host: Intel64 Family 6 Model 158 Stepping 10, 6 logical cores, Windows 11, Python 3.13.1, `onnxruntime` 1.30.0.

ONNX Runtime's CPU QDQ path pays a quantize/dequantize pair per activation and has no vectorized int8 kernel for activations this small (`3 x 64 x 63`), so the INT8 graph is slower here rather than faster. **v3.1 makes no speed claim anywhere**; the intended consumer is the v4.0 ONNX Runtime Web client, whose WASM SIMD / WebNN kernels consume the QDQ graph natively. `tests/test_v3_1_quantization.py` re-measures the ratio wherever it runs and prints it instead of asserting one.

---

## ★ Deployment

### - Loading the Artifacts

| What | Resolution |
| :--- | :--- |
| INT8 graph | `checkpoints/model_v3.1/CommandSense_v3.1_int8.onnx` |
| float32 reference | `checkpoints/model_v3.1/CommandSense_v3.1_fp32.onnx` |
| `T` / `tau` | `configs/calibration/CommandSense_calibration_v3.1.json`, with the graph's `commandsense.calibration.*` metadata as the self-describing copy |

* The repository default is still `MODEL_VERSION = "v3.0"`: `src/inference.py` and `app.py` keep serving the v3.0 float32 bundle, and the INT8 build is a separate artifact rather than a drop-in replacement on that path. `src/inference.py` resolves `CommandSense_<version>.onnx`, so pointing a deployment at the INT8 graph means shipping it under that name for its version.
* The graph stays **raw-logit**. `src/inference.py` applies `softmax(logits / T)` and `tau` in Python for both backends, so changing the threshold never requires a re-export and both backends report the same numbers.
* `tests/test_v3_1_quantization.py` is the contract check: it re-derives the shape and input name, the agreement floor and the per-class agreement, the mean softmax drift, the metadata round trip and the operating-point transfer, and prints the flips, their float32 margins, the artifact size and the latency ratio.

### - Files in this folder

| File | Contents |
| :--- | :--- |
| `quantization_report.txt` | The full run: provenance, artifact size, operating point, fidelity, per-class flips, calibration quality, latency and notes |

Reproduce it with:
```text
python quantize_int8.py                 # builds both graphs, the JSON payload and this report
python tests/test_v3_1_quantization.py  # re-measures the artifact and prints the same metrics
```

`quantize_int8.py` seeds the RNG (`SEED = 42`) before calibration, so a rebuild from the same checkpoint reproduces the same activation ranges, the same `T`/`tau` decision and the same report.

---

## ★ Caveats

* **The win is size, and only size.** No latency improvement is claimed: on the CPU provider the INT8 graph is 1.68x slower (2.718 ms/clip against 1.614 ms/clip), and the report states that on its own line.
* **No accuracy claim.** No weight was trained or fine-tuned for v3.1. Where a version table shows v3.1 with v3.0's accuracy cells, those cells are copies by construction; the only v3.1-specific numbers are the fidelity and operating-point measurements on this card.
* **The size reduction is not free in logit space.** The max abs logit deviation is 14.47 on a checkpoint whose logits span roughly +/-40, which is why this card characterises the copy in decision space (99.55 % agreement, 0.0002 mean softmax drift) and prints the logit deviation for inspection rather than asserting a bound on it.
* **The operating point is v3.0's, transferred.** It is reused because the coverage gap (0.21 pt) stayed inside the 0.5 pt tolerance. If a future rebuild of the artifact moves coverage past that tolerance, `quantize_int8.py` re-fits `T`/`tau` on the INT8 logits and the shipped payload changes.
* **The rejection clips are drawn at load time.** Both splits contain synthetic `_silence_` / `_unknown_` clips that `src/dataset.py` slices out of the held-out recordings, so the two rejection rows of the fidelity table are samples rather than constants: a different seed moves them while the 35 command columns stay put.
* **The counts here are the measurement split's.** 12,105 clips is the whole `testing` split (11,005 commands + 1,100 rejection clips), which is why the 35-command cells on this card differ from the command-only figures the repository's version table reports.
* **The artifacts are build outputs.** `checkpoints/` is git-ignored, so neither ONNX graph is in the repository and both must be rebuilt with `python quantize_int8.py`; the calibration JSON is versioned configuration and is tracked.
* **This card describes v3.1 only.** It is written to be read on its own and states no delta against another release; comparisons across versions live in the repository's root `README.md`.