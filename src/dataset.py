import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import (
    SAMPLE_RATE, TARGET_SAMPLES, N_FFT, HOP_LENGTH, N_MELS, DATA_DIR
)

class SpeechCommandsDataset(Dataset):
    def __init__(self, subset="training", cache_in_ram=True):
        super().__init__()
        self.subset = subset
        self.cache_in_ram = cache_in_ram
        
        print(f"[*] Indexing '{subset}' files from disk (scanning directory)...")
        # Automatic dataset downloading & path resolution via torchaudio
        self.raw_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=DATA_DIR,
            url="speech_commands_v0.02",
            download=True,
            subset=subset
        )
        print(f"[+] Found {len(self.raw_dataset)} samples for '{subset}'.")
        
        # Extract the target classes
        self.labels = sorted(list(set(sample[2] for sample in self.raw_dataset)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # 1. Log-Mel Spectrogram Transformation
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        
        # 2. Delta & Delta-Delta Transformations for 3-Channel Spectrograms (v2.0.0)
        self.compute_deltas = T.ComputeDeltas()
        
        self.cached_features = []
        self.cached_targets = []
        
        if self.cache_in_ram:
            print(f"[*] Caching {len(self.raw_dataset)} 3-channel samples to RAM...")
            for idx in tqdm(
                range(len(self.raw_dataset)), 
                desc=f"Caching {subset} set (3-channels)", 
                unit="files", 
                mininterval=0.1
            ):
                feat, target = self._process_sample(idx)
                self.cached_features.append(feat)
                self.cached_targets.append(target)
            print(f"[+] {len(self.cached_features)} samples loaded into RAM.\n")

    def _process_sample(self, idx):
        waveform, sample_rate, label, _, _ = self.raw_dataset[idx]
        
        # Resample to 16kHz if necessary
        if sample_rate != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or crop to target 1 second length (16,000 samples)
        num_samples = waveform.shape[1]
        if num_samples < TARGET_SAMPLES:
            padding = TARGET_SAMPLES - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif num_samples > TARGET_SAMPLES:
            waveform = waveform[:, :TARGET_SAMPLES]
            
        # Compute Log-Mel Spectrogram (Channel 1)
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-6)
        
        # Compute Delta (Channel 2) and Delta-Delta (Channel 3)
        delta_spec = self.compute_deltas(log_mel_spec)
        delta2_spec = self.compute_deltas(delta_spec)
        
        # Concatenate along channel dimension -> Shape: (3, N_MELS, T)
        three_channel_spec = torch.cat([log_mel_spec, delta_spec, delta2_spec], dim=0)
        
        target_idx = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return three_channel_spec, target_idx

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        if self.cache_in_ram:
            return self.cached_features[idx], self.cached_targets[idx]
        return self._process_sample(idx)

if __name__ == "__main__":
    print("[*] Testing SpeechCommandsDataset initialization & triggering download...")
    dataset = SpeechCommandsDataset(subset="validation", cache_in_ram=False)
    print(f"[+] Dataset successfully loaded! Total samples: {len(dataset)}")