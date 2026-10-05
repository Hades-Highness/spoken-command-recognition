import gc
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
    and '_unknown_' (out-of-vocabulary speech samples from LibriSpeech).
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

        # Extra samples buffer for _silence_ and _unknown_ classes only
        self.extra_samples = []
        self._generate_extra_samples()

        self.num_raw = len(self.raw_dataset)
        self.num_total = self.num_raw + len(self.extra_samples)

        self.feature_extractor = AudioFeatureExtractor()
        self.cached_features = []
        self.cached_targets = []

        if self.cache_in_ram:
            print(f"[*] Caching {self.num_total} samples to RAM as 3-channel features...")
            for idx in tqdm(
                range(self.num_total),
                desc=f"Caching {subset} set",
                unit="files",
                mininterval=0.5,
            ):
                feature, target = self._process_sample(idx)
                self.cached_features.append(feature)
                self.cached_targets.append(target)

            # Instantly release extra raw waveforms from RAM after feature extraction
            del self.extra_samples
            self.extra_samples = []
            gc.collect()

            print(f"[+] {len(self.cached_features)} samples loaded into RAM.\n")

    def _generate_extra_samples(self):
        """Generates _silence_ and _unknown_ samples using real audio sources."""
        base_dir = os.path.join(DATA_DIR, "SpeechCommands", "speech_commands_v0.02")
        bg_dir = os.path.join(base_dir, "_background_noise_")
        
        num_extra_target = int(len(self.raw_dataset) * 0.05)  # ~5% of split size

        # 1. Inject _silence_ samples (split noise sources to prevent train/test leakage)
        if os.path.exists(bg_dir):
            bg_files = sorted([
                os.path.join(bg_dir, f)
                for f in os.listdir(bg_dir)
                if f.endswith(".wav")
            ])
            
            # Partition background noise files between splits
            if self.subset == "training":
                selected_bg = bg_files[:4]
            else:
                selected_bg = bg_files[4:]
                
            if selected_bg:
                for _ in range(num_extra_target):
                    bg_file = random.choice(selected_bg)
                    waveform, sr = torchaudio.load(bg_file)
                    
                    if waveform.shape[1] > TARGET_SAMPLES:
                        max_start = waveform.shape[1] - TARGET_SAMPLES
                        start = random.randint(0, max_start)
                        chunk = waveform[:, start : start + TARGET_SAMPLES]
                    else:
                        chunk = waveform
                    
                    chunk = chunk * random.uniform(0.1, 1.0)
                    self.extra_samples.append((chunk, sr, "_silence_"))
        else:
            print(f"[!] Warning: Background noise directory {bg_dir} not found.")

        # 2. Inject _unknown_ samples from LibriSpeech (real OOV speech)
        ls_url = "dev-clean" if self.subset in ("validation", "testing") else "train-clean-100"
        try:
            librispeech_ds = torchaudio.datasets.LIBRISPEECH(
                root=str(DATA_DIR),
                url=ls_url,
                download=True,
            )
            
            indices = list(range(len(librispeech_ds)))
            random.shuffle(indices)
            
            collected = 0
            for idx in indices:
                if collected >= num_extra_target:
                    break
                waveform, sr, _, _, _, _ = librispeech_ds[idx]
                
                # Crop 1-second segment from speech file
                if waveform.shape[1] >= TARGET_SAMPLES:
                    max_start = waveform.shape[1] - TARGET_SAMPLES
                    start = random.randint(0, max_start)
                    chunk = waveform[:, start : start + TARGET_SAMPLES]
                else:
                    chunk = torch.nn.functional.pad(
                        waveform, (0, TARGET_SAMPLES - waveform.shape[1])
                    )
                
                self.extra_samples.append((chunk, sr, "_unknown_"))
                collected += 1

        except Exception as e:
            print(f"[!] Warning: Failed to load LibriSpeech dataset ({e}). Falling back to noise mix.")
            for _ in range(num_extra_target):
                noise = torch.randn(1, TARGET_SAMPLES) * 0.02
                self.extra_samples.append((noise, SAMPLE_RATE, "_unknown_"))

    def _get_raw_sample(self, idx: int):
        """Fetches raw audio dynamically on demand to avoid RAM saturation."""
        if idx < self.num_raw:
            waveform, sample_rate, label, _, _ = self.raw_dataset[idx]
            return waveform, sample_rate, label
        else:
            return self.extra_samples[idx - self.num_raw]

    def _process_sample(self, idx: int):
        waveform, sample_rate, label = self._get_raw_sample(idx)
        waveform = prepare_waveform(waveform, sample_rate)

        with torch.no_grad():
            features = self.feature_extractor(waveform).squeeze(0)  # [3, N_MELS, frames]

        target = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return features, target

    def __len__(self) -> int:
        return self.num_total

    def __getitem__(self, idx: int):
        if self.cache_in_ram:
            return self.cached_features[idx], self.cached_targets[idx]
        return self._process_sample(idx)


if __name__ == "__main__":
    dataset = SpeechCommandsDataset(subset="validation", cache_in_ram=False)
    features, target = dataset[0]
    print(f"[+] {len(dataset)} samples, {len(dataset.labels)} classes")
    print(f"[+] feature shape {tuple(features.shape)}, first target {int(target)}")