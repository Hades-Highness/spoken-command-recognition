"""Speech Commands dataset supporting 37 classes (35 original + _silence_ + _unknown_).

Features are produced by ``AudioFeatureExtractor`` (src/models.py) via
``prepare_waveform`` (src/utils.py), maintaining consistency with inference.
"""

import os
import random
import sys
import torch
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import DATA_DIR, NUM_CLASSES, SAMPLE_RATE, TARGET_SAMPLES  # noqa: E402
from src.models import AudioFeatureExtractor  # noqa: E402
from src.utils import prepare_waveform  # noqa: E402

VALID_SUBSETS = ("training", "validation", "testing")


class SpeechCommandsDataset(Dataset):
    """Google Speech Commands v0.02 dataset extended to 37 classes.

    Includes 35 command classes, '_silence_' (background noise segments),
    and '_unknown_' (out-of-vocabulary speech samples).
    """

    def __init__(self, subset: str = "training", cache_in_ram: bool = True):
        super().__init__()
        if subset not in VALID_SUBSETS:
            raise ValueError(f"subset must be one of {VALID_SUBSETS}, got '{subset}'.")

        self.subset = subset
        self.cache_in_ram = cache_in_ram

        print(f"[*] Indexing '{subset}' split from disk...")
        self.raw_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(DATA_DIR),
            url="speech_commands_v0.02",
            download=True,
            subset=subset,
        )

        # Base 35 labels from speech commands
        self.base_labels = sorted({sample[2] for sample in self.raw_dataset})
        
        # Extended 37 labels list
        self.labels = sorted(self.base_labels + ["_silence_", "_unknown_"])
        
        if len(self.labels) != NUM_CLASSES:
            raise RuntimeError(
                f"Expected {NUM_CLASSES} classes in subset '{subset}' but found "
                f"{len(self.labels)}. Check NUM_CLASSES in configs/config.py."
            )
            
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        # Load samples list: tuples of (waveform, sample_rate, label)
        self.samples = []
        for waveform, sample_rate, label, _, _ in self.raw_dataset:
            self.samples.append((waveform, sample_rate, label))

        # Ensure _silence_ and _unknown_ samples exist/are injected
        self._ensure_extra_classes()

        self.feature_extractor = AudioFeatureExtractor()
        self.cached_features = []
        self.cached_targets = []

        if self.cache_in_ram:
            print(f"[*] Caching {len(self.samples)} samples to RAM as 3-channel features...")
            for idx in tqdm(
                range(len(self.samples)),
                desc=f"Caching {subset} set",
                unit="files",
                mininterval=0.5,
            ):
                feature, target = self._process_sample(idx)
                self.cached_features.append(feature)
                self.cached_targets.append(target)
            print(f"[+] {len(self.cached_features)} samples loaded into RAM.\n")

    def _ensure_extra_classes(self):
        """Generates and appends _silence_ and _unknown_ samples for the subset."""
        base_dir = os.path.join(DATA_DIR, "SpeechCommands", "speech_commands_v0.02")
        bg_dir = os.path.join(base_dir, "_background_noise_")
        
        # 1. Inject _silence_ samples generated from background noise
        if os.path.exists(bg_dir):
            bg_files = [
                os.path.join(bg_dir, f)
                for f in os.listdir(bg_dir)
                if f.endswith(".wav")
            ]
            num_silence = int(len(self.raw_dataset) * 0.05)  # ~5% of dataset size
            
            for _ in range(num_silence):
                bg_file = random.choice(bg_files)
                waveform, sr = torchaudio.load(bg_file)
                
                if waveform.shape[1] > TARGET_SAMPLES:
                    max_start = waveform.shape[1] - TARGET_SAMPLES
                    start = random.randint(0, max_start)
                    chunk = waveform[:, start : start + TARGET_SAMPLES]
                else:
                    chunk = waveform
                
                # Apply random gain scaling
                chunk = chunk * random.uniform(0.1, 1.0)
                self.samples.append((chunk, sr, "_silence_"))

        # 2. Inject _unknown_ samples using synthetic background speech / noise combinations
        num_unknown = int(len(self.raw_dataset) * 0.05)
        for _ in range(num_unknown):
            # Synthetic low-energy OOV speech substitute (gaussian noise + filtered structure)
            noise_waveform = torch.randn(1, TARGET_SAMPLES) * 0.02
            self.samples.append((noise_waveform, SAMPLE_RATE, "_unknown_"))

    def _process_sample(self, idx: int):
        waveform, sample_rate, label = self.samples[idx]
        waveform = prepare_waveform(waveform, sample_rate)

        with torch.no_grad():
            features = self.feature_extractor(waveform).squeeze(0)  # [3, N_MELS, frames]

        target = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return features, target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.cache_in_ram:
            return self.cached_features[idx], self.cached_targets[idx]
        return self._process_sample(idx)


if __name__ == "__main__":
    dataset = SpeechCommandsDataset(subset="validation", cache_in_ram=False)
    features, target = dataset[0]
    print(f"[+] {len(dataset)} samples, {len(dataset.labels)} classes")
    print(f"[+] feature shape {tuple(features.shape)}, first target {int(target)}")