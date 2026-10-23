import type * as OrtTypes from "onnxruntime-web";

/**
 * CommandSense v4.1 inference engine for the browser.
 *
 * The v4.1 export bakes the whole front-end into the graph: the session takes
 * raw mono audio shaped [batch, 16000] in [-1, 1] and returns 37 raw logits.
 * Temperature scaling, softmax and the confidence threshold are applied here,
 * exactly like src/inference.py does on the Python side. Keeping them outside
 * the graph means tau can be tuned without re-exporting the model.
 */

/** Native sample rate of the training data and of the ONNX graph input. */
export const SAMPLE_RATE = 16000;

/** Samples per inference window: 1.0 s at 16 kHz. */
export const WINDOW_SAMPLES = SAMPLE_RATE;

/** Samples between consecutive windows: 0.5 s (50 percent overlap). */
export const HOP_SAMPLES = WINDOW_SAMPLES / 2;

/** Release artifact served from the public folder. */
export const MODEL_URL = "/models/CommandSense_v4.1.onnx";

/** Fitted on the validation split; see configs/calibration/CommandSense_calibration_v4.1.json. */
export const TEMPERATURE = 1.252585;

/** Fitted operating point; probabilities below this value are rejected. */
export const CONFIDENCE_THRESHOLD = 0.92;

export type CalibrationSettings = {
  temperature: number;
  confidenceThreshold: number;
};

export const SILENCE_LABEL = "_silence_";
export const UNKNOWN_LABEL = "_unknown_";

/**
 * Class order of the trained model, copied from configs/labels.json.
 * Index 0 is silence, index 1 is unknown, the remaining 35 entries are the
 * commands the UI is allowed to display.
 */
export const COMMAND_LABELS: readonly string[] = [
  "_silence_",
  "_unknown_",
  "backward",
  "bed",
  "bird",
  "cat",
  "dog",
  "down",
  "eight",
  "five",
  "follow",
  "forward",
  "four",
  "go",
  "happy",
  "house",
  "learn",
  "left",
  "marvin",
  "nine",
  "no",
  "off",
  "on",
  "one",
  "right",
  "seven",
  "sheila",
  "six",
  "stop",
  "three",
  "tree",
  "two",
  "up",
  "visual",
  "wow",
  "yes",
  "zero",
];

/** Labels that describe "no command spoken" rather than a word. */
export const REJECTION_LABELS: ReadonlySet<string> = new Set([
  SILENCE_LABEL,
  UNKNOWN_LABEL,
]);

export type InferenceResult = {
  /** argmax label, including silence and unknown. */
  command: string;
  /** Calibrated probability of the winning class, in [0, 1]. */
  confidence: number;
  /** True when the winning probability is below CONFIDENCE_THRESHOLD. */
  isRejected: boolean;
};

type OrtModule = typeof import("onnxruntime-web");

let ortModulePromise: Promise<OrtModule> | null = null;

/**
 * Loads onnxruntime-web on demand so the runtime never enters the server
 * bundle, and points it at the WASM assets that scripts/copy-ort-runtime.mjs
 * writes to public/ort.
 */
async function loadOrtRuntime(): Promise<OrtModule> {
  if (!ortModulePromise) {
    ortModulePromise = import("onnxruntime-web/wasm")
      .then((module) => {
        const ort = module as unknown as OrtModule;
        const isMobile = 
  /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
  window.innerWidth < 768;

        ort.env.wasm.wasmPaths = "/ort/";
        ort.env.logLevel = "error";

        if (isMobile) {
          ort.env.wasm.numThreads = 1;
          ort.env.wasm.proxy = false;
          console.log("[ONNX] Mobile browser detected. Using single-threaded WASM for stability.");
        } else {
          console.log("[ONNX] Desktop browser detected. Using multithreaded WASM.");
        }
        return ort;
      })
      .catch((error: unknown) => {
        ortModulePromise = null;
        throw error;
      });
  }

  return ortModulePromise;
}

/**
 * Owns the analyze ONNX InfererenceSession and serializes window classification
 * so overlapping windows cannot run at the same time.
 */
export class CommandSenseInference {
  private session: OrtTypes.InferenceSession | null = null;

  private sessionPromise: Promise<OrtTypes.InferenceSession> | null = null;

  private inputName = "raw_audio";

  private outputName = "logits";

  private queue: Promise<unknown> = Promise.resolve();

  get isReady(): boolean {
    return this.session !== null;
  }

  /** Creates the session if needed. Safe to call from multiple components. */
  async load(): Promise<void> {
    await this.ensureSession();
  }

  /**
   * Classifies one 1.0 s window and returns the calibrated decision.
   *
   * @param audioData Exactly 16000 mono float samples in [-1, 1] at 16 kHz.
   */
  runInference(
    audioData: Float32Array,
    calibration: CalibrationSettings = {
      temperature: TEMPERATURE,
      confidenceThreshold: CONFIDENCE_THRESHOLD,
    },
  ): Promise<InferenceResult> {
    if (audioData.length !== WINDOW_SAMPLES) {
      return Promise.reject(
        new Error(
          `runInference expects exactly ${WINDOW_SAMPLES} samples, received ${audioData.length}.`,
        ),
      );
    }

    // Chain on the previous call so the session sees one window at a time.
    const task = this.queue.then(() => this.execute(audioData, calibration));
    this.queue = task.then(
      () => undefined,
      () => undefined,
    );
    return task;
  }

  /** Releases the WASM session. The next load() recreates it. */
  async dispose(): Promise<void> {
    const session = this.session;
    this.session = null;
    this.sessionPromise = null;

    if (session) {
      await session.release();
    }
  }

  private async execute(
    audioData: Float32Array,
    calibration: CalibrationSettings,
  ): Promise<InferenceResult> {
    const ort = await loadOrtRuntime();
    const session = await this.ensureSession();

    const input = new ort.Tensor("float32", audioData, [1, WINDOW_SAMPLES]);
    const outputs = await session.run({ [this.inputName]: input });
    const output = outputs[this.outputName] ?? Object.values(outputs)[0];

    if (!output) {
      throw new Error("The CommandSense model returned no output tensor.");
    }

    const logits = output.data;
    if (!(logits instanceof Float32Array)) {
      throw new Error(
        "The CommandSense model returned logits that are not float32.",
      );
    }
    if (logits.length !== COMMAND_LABELS.length) {
      throw new Error(
        `The model returned ${logits.length} logits but ${COMMAND_LABELS.length} labels are mapped. ` +
          "The label order and the release artifact do not match.",
      );
    }

    return applyCalibration(logits, calibration);
  }

  private ensureSession(): Promise<OrtTypes.InferenceSession> {
    if (this.session) {
      return Promise.resolve(this.session);
    }

    if (!this.sessionPromise) {
      this.sessionPromise = this.createSession().catch((error: unknown) => {
        this.sessionPromise = null;
        throw error;
      });
    }

    return this.sessionPromise;
  }

  private async createSession(): Promise<OrtTypes.InferenceSession> {
    const ort = await loadOrtRuntime();

    // Fetch explicitly so a missing artifact surfaces as a readable 404 instead
    // of an opaque ONNX Runtime deserialization error.
    const response = await fetch(MODEL_URL);
    if (!response.ok) {
      throw new Error(
        `Unable to load ${MODEL_URL} (HTTP ${response.status}). ` +
          "Place the CommandSense_v4.1.onnx release artifact in web/public/models/.",
      );
    }

    const model = new Uint8Array(await response.arrayBuffer());
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    this.inputName = session.inputNames[0] ?? "raw_audio";
    this.outputName = session.outputNames[0] ?? "logits";
    this.session = session;

    // Warm up the WASM kernels so the first real window is not slower than the
    // rest of the stream.
    await session.run({
      [this.inputName]: new ort.Tensor(
        "float32",
        new Float32Array(WINDOW_SAMPLES),
        [1, WINDOW_SAMPLES],
      ),
    });

    return session;
  }
}

/**
 * Temperature scaling followed by a numerically stable softmax, then the tau
 * gate. Exported so the math can be unit tested without a model.
 */
export function applyCalibration(
  logits: Float32Array,
  {
    temperature = TEMPERATURE,
    confidenceThreshold = CONFIDENCE_THRESHOLD,
  }: CalibrationSettings = {
    temperature: TEMPERATURE,
    confidenceThreshold: CONFIDENCE_THRESHOLD,
  },
): InferenceResult {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new Error("Temperature must be a finite number greater than zero.");
  }
  if (
    !Number.isFinite(confidenceThreshold) ||
    confidenceThreshold < 0 ||
    confidenceThreshold > 1
  ) {
    throw new Error("Confidence threshold must be between 0 and 1.");
  }

  const classCount = logits.length;
  const scaled = new Float32Array(classCount);

  let maxScaled = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < classCount; index += 1) {
    const value = logits[index] / temperature;
    scaled[index] = value;
    if (value > maxScaled) {
      maxScaled = value;
    }
  }

  let total = 0;
  for (let index = 0; index < classCount; index += 1) {
    const value = Math.exp(scaled[index] - maxScaled);
    scaled[index] = value;
    total += value;
  }

  let bestIndex = 0;
  let bestProbability = 0;
  for (let index = 0; index < classCount; index += 1) {
    const probability = total > 0 ? scaled[index] / total : 0;
    if (probability > bestProbability) {
      bestProbability = probability;
      bestIndex = index;
    }
  }

  return {
    command: COMMAND_LABELS[bestIndex] ?? UNKNOWN_LABEL,
    confidence: bestProbability,
    isRejected: bestProbability < confidenceThreshold,
  };
}

/** Process-wide analyzeton: the model is loaded once per browser tab. */
export const onnxInference = new CommandSenseInference();

export function loadModel(): Promise<void> {
  return onnxInference.load();
}

export function runInference(
  audioData: Float32Array,
  calibration?: CalibrationSettings,
): Promise<InferenceResult> {
  return onnxInference.runInference(audioData, calibration);
}
