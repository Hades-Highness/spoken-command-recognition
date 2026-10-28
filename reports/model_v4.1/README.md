# CommandSense v4.1 - Model Card

---

## Model Overview

- **Model name:** CommandSense v4.1
- **Task:** Spoken keyword spotting with explicit silence and out-of-vocabulary rejection
- **Architecture:** Lightweight Keyword Spotting Network with a compact residual CNN and an audio feature front-end embedded in the exported model
- **Deployment format:** INT8 ONNX model executed by ONNX Runtime Web with the WebAssembly execution provider
- **Model size:** Approximately 1.4 MB
- **Audio input:** 16 kHz, mono, Float32 waveform
- **Input window:** 1.0 second (16,000 samples)
- **Output classes:** 37 total: 35 commands, `_silence_`, and `_unknown_`
- **Inference modes:** Listen Mode evaluates overlapping one-second windows every 0.5 seconds; Analyze Mode performs a single-shot analysis of a recorded or uploaded sample
- **Calibration:** Temperature scaling and confidence threshold are adjustable in the web application's Settings

The web application captures and processes audio in the browser. Audio is not sent to a server for inference.

## Technical Specifications

| Property | Specification |
| :-- | :-- |
| Architecture | Lightweight Keyword Spotting Network (residual CNN) |
| Trainable parameters | 1,217,349 |
| Model format | ONNX |
| Quantization | INT8 |
| Runtime | ONNX Runtime Web |
| Execution provider | WebAssembly (WASM) |
| Artifact size | Approximately 1.4 MB |
| Sample rate | 16,000 Hz |
| Channels | 1 (mono) |
| Sample type | Float32 |
| Model input shape | `[1, 16000]` |
| Listen Mode window / hop | 1.0 second / 0.5 second |
| Output classes | 37 |

### Class set

The model predicts `_silence_`, `_unknown_`, and these 35 commands:

`backward`, `bed`, `bird`, `cat`, `dog`, `down`, `eight`, `five`, `follow`, `forward`, `four`, `go`, `happy`, `house`, `learn`, `left`, `marvin`, `nine`, `no`, `off`, `on`, `one`, `right`, `seven`, `sheila`, `six`, `stop`, `three`, `tree`, `two`, `up`, `visual`, `wow`, `yes`, `zero`.

## Calibration

The calibration report records a fitted temperature of `T = 1.252585` and a confidence threshold of `tau = 0.92`. Both parameters can be adjusted in the web application's Settings. Temperature scaling calibrates confidence scores; the threshold controls whether a prediction is accepted or rejected as `_unknown_`.

The calibration values are stored in `configs/calibration/CommandSense_calibration_v4.1.json`. The exported ONNX model produces logits, allowing the web runtime to apply the selected calibration and threshold without re-exporting the model.

## Evaluation Results

The v4.1 report evaluates the model on the speaker-disjoint testing split. Temperature scaling does not change the predicted class, but it affects confidence and threshold-based acceptance.

| Metric | Result |
| :-- | --: |
| Testing split, 37-class top-1 accuracy | 95.9025% |
| Testing split, 35-command top-1 accuracy | 95.6111% |
| Coverage at `tau = 0.92` | 88.9467% |
| Retained accuracy on answered samples | 99.5728% |
| End-to-end yield | 88.5667% |

These are the results in `reports/model_v4.1/report_v4.1.txt`. They describe the reported evaluation split and do not guarantee performance under every recording condition.

## Intended Use and Limitations

CommandSense v4.1 is intended for local, browser-based detection of a fixed vocabulary of 35 spoken commands. It is not a general-purpose automatic speech recognition system and does not transcribe continuous speech. Detection quality depends on microphone quality, speaker, pronunciation, background noise and recording conditions. Threshold-based rejection trades coverage for confidence and should be configured for the intended application.

The Listen Mode sliding-window scan evaluates two overlapping windows per second. Analyze Mode instead preprocesses one recording and creates a single model input. The calibration and evaluation metrics are based on one-second evaluation clips; longer recordings may present more opportunities for a false detection.

## Artifacts

| Artifact | Location |
| :-- | :-- |
| Browser deployment model | `web/public/models/CommandSense_v4.1.onnx` |
| Calibration configuration | `configs/calibration/CommandSense_calibration_v4.1.json` |
| Evaluation report | `reports/model_v4.1/report_v4.1.txt` |
| Web application documentation | `web/README.md` |

The ONNX model is distributed separately from the source repository. Place it at the path above to run the web application.
