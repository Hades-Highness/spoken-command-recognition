"use client";

import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";

export type MicrophoneOption = {
  deviceId: string;
  label: string;
};

type SettingsModalProps = {
  open: boolean;
  microphones: MicrophoneOption[];
  selectedMicrophoneId: string;
  onMicrophoneChange: (deviceId: string) => void;
  onClearHistory: () => void;
  onClose: () => void;
};

export default function SettingsModal({
  open,
  microphones,
  selectedMicrophoneId,
  onMicrophoneChange,
  onClearHistory,
  onClose,
}: SettingsModalProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    const previouslyFocused =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);
    closeButtonRef.current?.focus();

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          key="settings-modal"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 flex items-end justify-center p-4 sm:items-center"
        >
          <button
            type="button"
            aria-label="Close settings"
            onClick={onClose}
            className="absolute inset-0 h-full w-full cursor-default bg-black/70 backdrop-blur-sm"
          />

          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
            initial={{ opacity: 0, y: 24, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.98 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="relative w-full max-w-md overflow-hidden rounded-2xl border border-white/10 bg-[#111111] shadow-2xl"
          >
            <div className="flex items-start justify-between gap-4 border-b border-white/5 px-5 py-4">
              <div>
                <h2
                  id="settings-title"
                  className="text-base font-semibold tracking-tight text-text-main"
                >
                  Settings
                </h2>
                <p className="mt-1 text-xs text-text-muted">
                  Microphone input and local detection history.
                </p>
              </div>

              <button
                type="button"
                ref={closeButtonRef}
                onClick={onClose}
                aria-label="Close settings"
                className="flex h-9 w-9 items-center justify-center rounded-full border border-white/10 text-text-muted transition-colors hover:border-primary/60 hover:text-text-main focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover"
              >
                <svg
                  aria-hidden="true"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  className="h-4 w-4"
                >
                  <path d="M6 6l12 12" />
                  <path d="M18 6L6 18" />
                </svg>
              </button>
            </div>

            <div className="flex flex-col gap-6 px-5 py-5">
              <div>
                <label
                  htmlFor="microphone-select"
                  className="text-[11px] font-medium uppercase tracking-[0.18em] text-text-muted"
                >
                  Microphone
                </label>

                <div className="relative mt-2">
                  <select
                    id="microphone-select"
                    value={selectedMicrophoneId}
                    onChange={(event) =>
                      onMicrophoneChange(event.target.value)
                    }
                    className="w-full appearance-none rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2.5 pr-9 text-sm text-text-main transition-colors hover:border-white/20 focus:border-primary/60 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover"
                  >
                    {microphones.length > 0 ? (
                      microphones.map((microphone) => (
                        <option
                          key={microphone.deviceId}
                          value={microphone.deviceId}
                        >
                          {microphone.label}
                        </option>
                      ))
                    ) : (
                      <option value="default">System default microphone</option>
                    )}
                  </select>

                  <svg
                    aria-hidden="true"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted"
                  >
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </div>

                <p className="mt-2 text-xs leading-5 text-text-muted">
                  Browser microphone permission is requested when a session
                  starts. The device list stays empty until access is granted.
                </p>
              </div>

              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
                <p className="text-sm font-medium text-text-main">History</p>
                <p className="mt-1 text-xs leading-5 text-text-muted">
                  Remove every detection stored in this browser session. The
                  action cannot be undone.
                </p>

                <button
                  type="button"
                  onClick={onClearHistory}
                  className="mt-3 inline-flex items-center justify-center rounded-full border border-primary/40 px-4 py-2 text-xs font-medium text-primary-hover transition-colors hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover"
                >
                  Clear history
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between gap-4 border-t border-white/5 px-5 py-4">
              <span className="text-[11px] text-text-muted">
                CommandSense v4.1
              </span>

              <button
                type="button"
                onClick={onClose}
                className="inline-flex items-center justify-center rounded-full bg-primary px-5 py-2 text-sm font-medium text-text-main transition-colors hover:bg-primary-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover focus-visible:ring-offset-2 focus-visible:ring-offset-[#111111]"
              >
                Done
              </button>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
