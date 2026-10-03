"""Speech Commands dataset with the official splits and in-RAM feature caching.

Features are produced by ``AudioFeatureExtractor`` (src/models.py) via
``prepare_waveform`` (src/utils.py), i.e. the exact same code path used at
inference time.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import DATA_DIR, NUM_CLASSES  # noqa: E402
from src.models import AudioFeatureExtractor  # noqa: E402
from src.utils import prepare_waveform  # noqa: E402  (also installs the load fallback)

VALID_SUBSETS = ("training", "validation", "testing")


class SpeechCommandsDataset(Dataset):
    """Google Speech Commands v0.02, 35 classes, 3-channel features.

    ``subset`` maps to the dataset's own speaker-disjoint split files:
    ``validation`` and ``testing`` are read from ``validation_list.txt`` and
    ``testing_list.txt``, and ``training`` is everything else.
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
        print(f"[+] Found {len(self.raw_dataset)} samples for '{subset}'.")

        self.labels = sorted({sample[2] for sample in self.raw_dataset})
        if len(self.labels) != NUM_CLASSES:
            raise RuntimeError(
                f"Expected {NUM_CLASSES} classes in subset '{subset}' but found "
                f"{len(self.labels)}. Check NUM_CLASSES in configs/config.py and that "
                "the dataset download completed."
            )
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        self.feature_extractor = AudioFeatureExtractor()

        self.cached_features = []
        self.cached_targets = []

        if self.cache_in_ram:
            print(f"[*] Caching {len(self.raw_dataset)} samples to RAM as 3-channel features...")
            for idx in tqdm(
                range(len(self.raw_dataset)),
                desc=f"Caching {subset} set",
                unit="files",
                mininterval=0.5,
            ):
                feature, target = self._process_sample(idx)
                self.cached_features.append(feature)
                self.cached_targets.append(target)
            print(f"[+] {len(self.cached_features)} samples loaded into RAM.\n")

    def _process_sample(self, idx: int):
        waveform, sample_rate, label, _, _ = self.raw_dataset[idx]
        waveform = prepare_waveform(waveform, sample_rate)

        with torch.no_grad():
            features = self.feature_extractor(waveform).squeeze(0)  # [3, N_MELS, frames]

        target = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return features, target

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int):
        if self.cache_in_ram:
            return self.cached_features[idx], self.cached_targets[idx]
        return self._process_sample(idx)


if __name__ == "__main__":
    dataset = SpeechCommandsDataset(subset="validation", cache_in_ram=False)
    features, target = dataset[0]
    print(f"[+] {len(dataset)} samples, {len(dataset.labels)} classes")
    print(f"[+] feature shape {tuple(features.shape)}, first target {int(target)}")
