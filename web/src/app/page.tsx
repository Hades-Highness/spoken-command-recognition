"use client";

import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion, MotionConfig } from "framer-motion";
import ContinuousMode, {
  type ContinuousModeHandle,
} from "@/components/ContinuousMode";
import Header from "@/components/Header";
import ModeToggle, { type RecognitionMode } from "@/components/ModeToggle";
import SettingsModal, {
  type MicrophoneOption,
} from "@/components/SettingsModal";
import SingleOscilloscope from "@/components/SingleOscilloscope";

const MICROPHONES: MicrophoneOption[] = [
  { deviceId: "default", label: "System default microphone" },
];

export default function Home() {
  const [mode, setMode] = useState<RecognitionMode>("continuous");
  const [isCapturing, setIsCapturing] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [microphoneId, setMicrophoneId] = useState(MICROPHONES[0].deviceId);
  const continuousRef = useRef<ContinuousModeHandle>(null);

  const handleModeChange = (nextMode: RecognitionMode) => {
    setMode(nextMode);
    setIsCapturing(false);
  };

  const handleToggleCapture = () => {
    setIsCapturing((current) => !current);
  };

  const handleClearHistory = useCallback(() => {
    continuousRef.current?.clearHistory();
  }, []);

  const handleOpenSettings = useCallback(() => {
    setSettingsOpen(true);
  }, []);

  const handleCloseSettings = useCallback(() => {
    setSettingsOpen(false);
  }, []);

  return (
    <MotionConfig reducedMotion="user">
      <div className="flex flex-1 flex-col">
        <Header onOpenSettings={handleOpenSettings} />

        <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 pb-12 pt-10 sm:px-6 sm:pt-12 lg:px-8">
          <h1 className="sr-only">CommandSense spoken command recognition</h1>

          <div className="flex justify-center">
            <ModeToggle mode={mode} onModeChange={handleModeChange} />
          </div>

          <div className="mt-8 flex flex-1 flex-col">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={mode}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
                className="w-full"
              >
                {mode === "continuous" ? (
                  <ContinuousMode
                    ref={continuousRef}
                    microphoneId={microphoneId}
                  />
                ) : (
                  <SingleOscilloscope
                    isCapturing={isCapturing}
                    onToggleCapture={handleToggleCapture}
                  />
                )}
              </motion.div>
            </AnimatePresence>
          </div>
        </main>

        <SettingsModal
          open={settingsOpen}
          microphones={MICROPHONES}
          selectedMicrophoneId={microphoneId}
          onMicrophoneChange={setMicrophoneId}
          onClearHistory={handleClearHistory}
          onClose={handleCloseSettings}
        />
      </div>
    </MotionConfig>
  );
}

