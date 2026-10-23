import { SAMPLE_RATE, WINDOW_SAMPLES } from "./onnxInference";

const SILENCE_FLOOR = 0.003;
const SPEECH_PADDING_SAMPLES = Math.round(SAMPLE_RATE * 0.08);
const RMS_BLOCK_SAMPLES = Math.round(SAMPLE_RATE * 0.01);

/** Decodes a browser recording and resamples it to the model's mono rate. */
export async function decodeRecording(blob: Blob): Promise<Float32Array> {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer());
    const outputLength = Math.ceil(decoded.duration * SAMPLE_RATE);
    const offlineContext = new OfflineAudioContext(
      1,
      outputLength,
      SAMPLE_RATE,
    );
    const source = offlineContext.createBufferSource();
    source.buffer = decoded;
    source.connect(offlineContext.destination);
    source.start();

    return (await offlineContext.startRendering()).getChannelData(0).slice();
  } finally {
    await context.close();
  }
}

/**
 * Removes leading/trailing silence and centers the spoken part in the model's
 * fixed one-second input window.
 */
export function prepareInferenceWindow(audio: Float32Array): Float32Array {
  let peak = 0;
  for (const sample of audio) {
    peak = Math.max(peak, Math.abs(sample));
  }
  if (peak < SILENCE_FLOOR) {
    throw new Error("No speech was detected. Record the word again.");
  }

  const threshold = Math.max(SILENCE_FLOOR, peak * 0.12);
  let firstSpeechBlock = -1;
  let lastSpeechBlock = -1;

  for (let start = 0; start < audio.length; start += RMS_BLOCK_SAMPLES) {
    const end = Math.min(start + RMS_BLOCK_SAMPLES, audio.length);
    let squaredTotal = 0;
    for (let index = start; index < end; index += 1) {
      squaredTotal += audio[index] * audio[index];
    }
    const rms = Math.sqrt(squaredTotal / (end - start));
    if (rms >= threshold) {
      if (firstSpeechBlock === -1) {
        firstSpeechBlock = start;
      }
      lastSpeechBlock = end;
    }
  }

  if (firstSpeechBlock === -1) {
    throw new Error("No speech was detected. Record the word again.");
  }

  const speechStart = Math.max(0, firstSpeechBlock - SPEECH_PADDING_SAMPLES);
  const speechEnd = Math.min(
    audio.length,
    lastSpeechBlock + SPEECH_PADDING_SAMPLES,
  );
  const speechLength = speechEnd - speechStart;
  const window = new Float32Array(WINDOW_SAMPLES);

  if (speechLength >= WINDOW_SAMPLES) {
    const start = speechStart + Math.floor((speechLength - WINDOW_SAMPLES) / 2);
    window.set(audio.subarray(start, start + WINDOW_SAMPLES));
  } else {
    const destinationStart = Math.floor((WINDOW_SAMPLES - speechLength) / 2);
    window.set(
      audio.subarray(speechStart, speechEnd),
      destinationStart,
    );
  }

  return window;
}

/** Downsamples audio into min/max sample pairs for an SVG waveform trace. */
export function getWaveformEnvelope(
  audio: Float32Array,
  pointCount = 800,
): Array<{ min: number; max: number }> {
  const envelope: Array<{ min: number; max: number }> = [];
  for (let point = 0; point < pointCount; point += 1) {
    const start = Math.floor((point * audio.length) / pointCount);
    const end = Math.max(
      start + 1,
      Math.floor(((point + 1) * audio.length) / pointCount),
    );
    let min = 0;
    let max = 0;
    for (let index = start; index < Math.min(end, audio.length); index += 1) {
      min = Math.min(min, audio[index]);
      max = Math.max(max, audio[index]);
    }
    envelope.push({ min, max });
  }
  return envelope;
}
