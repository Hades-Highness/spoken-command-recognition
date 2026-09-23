import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from configs.config import (
    DATA_DIR, SAMPLE_RATE, TARGET_SAMPLES, 
    N_MELS, N_FFT, HOP_LENGTH
)

class SpeechCommandsDataset(Dataset):
    def __init__(self, subset="training", cache_in_ram=True):
        super().__init__()
        self.subset = subset
        self.cache_in_ram = cache_in_ram
        
        # Téléchargement et extraction automatique dans ./data
        self.raw_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=DATA_DIR,
            url="speech_commands_v0.02",
            download=True,
            subset=subset
        )
        
        # Extraction des 35 classes
        self.labels = sorted(list(set(sample[2] for sample in self.raw_dataset)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # Transformation Log-Mel Spectrogram
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        
        self.cached_features = []
        self.cached_targets = []
        
        if self.cache_in_ram:
            print(f"[*] Chargement et mise en cache du dataset ({subset}) en RAM...")
            for idx in tqdm(range(len(self.raw_dataset))):
                feat, target = self._process_sample(idx)
                self.cached_features.append(feat)
                self.cached_targets.append(target)
            print(f"[+] {len(self.cached_features)} échantillons chargés en RAM.")

    def _process_sample(self, idx):
        waveform, sample_rate, label, _, _ = self.raw_dataset[idx]
        
        if sample_rate != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        num_samples = waveform.shape[1]
        if num_samples < TARGET_SAMPLES:
            padding = TARGET_SAMPLES - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif num_samples > TARGET_SAMPLES:
            waveform = waveform[:, :TARGET_SAMPLES]
            
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-6)
        
        target_idx = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return log_mel_spec, target_idx

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        if self.cache_in_ram:
            return self.cached_features[idx], self.cached_targets[idx]
        return self._process_sample(idx)