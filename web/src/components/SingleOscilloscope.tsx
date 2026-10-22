"use client";

const WAVEFORM_AMPLITUDES = [
  0.04, 0.18, 0.42, 0.26, 0.6, 0.34, 0.14, 0.5, 0.72, 0.4, 0.22, 0.48, 0.66,
  0.3, 0.12, 0.38, 0.58, 0.24, 0.08, 0.3, 0.52, 0.78, 0.36, 0.16, 0.44, 0.62,
  0.28, 0.1, 0.34, 0.56, 0.74, 0.4, 0.18, 0.46, 0.64, 0.3, 0.12, 0.26, 0.48,
  0.2, 0.06,
];

const GRID_COLUMNS = [100, 200, 300, 400, 500, 600, 700];
const GRID_ROWS = [60, 120, 180];

function buildWaveformPath(amplitudes: readonly number[]) {
  const width = 800;
  const height = 240;
  const centerY = height / 2;
  const maxAmplitude = height * 0.4;
  const step = width / (amplitudes.length - 1);

  return amplitudes
    .map((amplitude, index) => {
      const x = index * step;
      const direction = index % 2 === 0 ? -1 : 1;
      const y = centerY + direction * amplitude * maxAmplitude;

      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

type SingleOscilloscopeProps = {
  isCapturing: boolean;
  onToggleCapture: () => void;
};

export default function SingleOscilloscope({
  isCapturing,
  onToggleCapture,
}: SingleOscilloscopeProps) {
  const waveformPath = buildWaveformPath(WAVEFORM_AMPLITUDES);

  return (
    <section
      aria-label="Single-shot waveform analysis"
      className="w-full rounded-3xl border border-white/5 bg-white/[0.02] p-5 sm:p-8"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-[11px] font-medium uppercase tracking-[0.22em] text-text-muted">
            Single shot
          </p>
          <h2 className="mt-2 text-lg font-semibold tracking-tight text-text-main">
            Waveform analysis
          </h2>
        </div>

        <span
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${
            isCapturing
              ? "border-primary-hover/60 text-primary-hover"
              : "border-white/10 text-text-muted"
          }`}
        >
          <span
            aria-hidden="true"
            className={`h-1.5 w-1.5 rounded-full ${
              isCapturing
                ? "animate-pulse-slow bg-primary-hover"
                : "bg-text-muted/60"
            }`}
          />
          {isCapturing ? "Capturing" : "Idle"}
        </span>
      </div>

      <div className="relative mt-6 overflow-hidden rounded-2xl border border-white/5 bg-background/60">
        <svg
          viewBox="0 0 800 240"
          preserveAspectRatio="none"
          role="img"
          aria-label="Placeholder waveform awaiting capture"
          className="h-56 w-full sm:h-64"
        >
          <defs>
            <linearGradient
              id="oscilloscope-stroke"
              x1="0"
              y1="0"
              x2="1"
              y2="0"
            >
              <stop offset="0%" stopColor="#c20009" stopOpacity="0.35" />
              <stop offset="50%" stopColor="#f92b35" stopOpacity="1" />
              <stop offset="100%" stopColor="#c20009" stopOpacity="0.35" />
            </linearGradient>
          </defs>

          {GRID_COLUMNS.map((x) => (
            <line
              key={`column-${x}`}
              x1={x}
              y1="0"
              x2={x}
              y2="240"
              stroke="#ffffff"
              strokeOpacity="0.05"
            />
          ))}

          {GRID_ROWS.map((y) => (
            <line
              key={`row-${y}`}
              x1="0"
              y1={y}
              x2="800"
              y2={y}
              stroke="#ffffff"
              strokeOpacity="0.05"
            />
          ))}

          <line
            x1="0"
            y1="120"
            x2="800"
            y2="120"
            stroke="#ffffff"
            strokeOpacity="0.1"
            strokeDasharray="4 8"
          />

          <path
            d={waveformPath}
            fill="none"
            stroke="url(#oscilloscope-stroke)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`transition-opacity duration-500 ${
              isCapturing ? "animate-pulse-slow opacity-100" : "opacity-60"
            }`}
          />
        </svg>

        <span className="pointer-events-none absolute left-4 top-3 text-[10px] uppercase tracking-[0.2em] text-text-muted">
          16 kHz mono
        </span>
        <span className="pointer-events-none absolute bottom-3 right-4 text-[10px] uppercase tracking-[0.2em] text-text-muted">
          1.00 s window
        </span>
      </div>

      <dl className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="rounded-xl border border-white/5 bg-white/[0.02] px-4 py-3">
          <dt className="text-[10px] font-medium uppercase tracking-[0.18em] text-text-muted">
            Command
          </dt>
          <dd className="mt-1 text-sm font-medium text-text-main">
            {isCapturing ? "Analyzing" : "Awaiting capture"}
          </dd>
        </div>

        <div className="rounded-xl border border-white/5 bg-white/[0.02] px-4 py-3">
          <dt className="text-[10px] font-medium uppercase tracking-[0.18em] text-text-muted">
            Confidence
          </dt>
          <dd className="mt-1 font-mono text-sm tabular-nums text-text-main">
            {isCapturing ? "Pending" : "—"}
          </dd>
        </div>

        <div className="rounded-xl border border-white/5 bg-white/[0.02] px-4 py-3">
          <dt className="text-[10px] font-medium uppercase tracking-[0.18em] text-text-muted">
            Window
          </dt>
          <dd className="mt-1 font-mono text-sm tabular-nums text-text-main">
            1.00 s
          </dd>
        </div>
      </dl>

      <div className="mt-6 flex justify-end">
        <button
          type="button"
          onClick={onToggleCapture}
          aria-pressed={isCapturing}
          className="inline-flex items-center justify-center rounded-full bg-primary px-5 py-2.5 text-sm font-medium text-text-main transition-colors hover:bg-primary-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover focus-visible:ring-offset-2 focus-visible:ring-offset-background"
        >
          {isCapturing ? "Stop capture" : "Start capture"}
        </button>
      </div>
    </section>
  );
}
