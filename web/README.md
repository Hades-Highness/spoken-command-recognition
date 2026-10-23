This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## CommandSense web client

The app runs the exported CommandSense v4.1 graph entirely in the browser with
ONNX Runtime Web (WASM execution provider) and the Web Audio API. No audio
leaves the machine.

### Before the first run

1. Place the release artifact at `public/models/CommandSense_v4.1.onnx`.
   Model weights are not committed; the loader expects exactly this file name.
2. Install dependencies with `npm install`. The `postinstall` script copies the
   ONNX Runtime WASM files from `node_modules` into `public/ort`, which is where
   `ort.env.wasm.wasmPaths` points. `predev` and `prebuild` refresh them if the
   `onnxruntime-web` version changes.

If the model is missing, the UI reports the HTTP 404 instead of failing
silently.

### Pipeline

- `src/lib/onnxInference.ts` loads the session once per tab, feeds 16000-sample
  float windows shaped `[1, 16000]`, then applies temperature scaling
  (`T = 1.252585`), softmax and the `tau = 0.92` rejection gate. Label order is
  mirrored from `configs/labels.json`.
- `src/lib/audioProcessor.ts` captures the microphone at the hardware rate,
  low-passes it with a 129-tap windowed-sinc FIR inside an `AudioWorklet` and
  decimates to 16 kHz by linear interpolation. The main thread accumulates
  16000-sample windows and classifies one every 8000 samples (0.5 s hop, 50
  percent overlap).
- `src/components/ListenMode.tsx` owns the session state; the visualizer is
  presentational and only receives the detection history.
- `src/components/AnalyzeMode.tsx` records one short microphone sample,
  accepts uploaded audio files, displays its live and recorded audio waveform,
  and supports playback and re-recording.
  Single-shot analysis resamples audio to 16 kHz, removes leading and trailing
  silence, centers the spoken segment in the model's one-second input window,
  and runs the same calibrated inference as listen mode. Microphone selection
  and the temperature (`T`) and confidence threshold (`tau`) can be adjusted in
  Settings. Audio is processed locally in the browser.

### Detections

Silence and unknown predictions are dropped, as are probabilities below
`tau`. Because consecutive windows overlap, a repeated word is ignored for one
second so a single utterance produces a single history entry.

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
