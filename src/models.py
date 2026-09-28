import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import NUM_CLASSES

class ResidualBlock(nn.Module):
    """
    Bloc résiduel classique adapté aux spectrogrammes 2D (Temps x Fréquence).
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut si changement de dimension ou de stride
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CommandSense(nn.Module):
    """
    Architecture CNN Résiduelle légère pour la classification de commandes vocales (35 classes).
    Entrée : [Batch, 1, N_MELS (64), FRAMES (63)]
    Sortie : Logits [Batch, 35]
    """
    def __init__(self, num_classes=NUM_CLASSES, in_channels=3):
        super().__init__()
        
        # Couche d'entrée / Préparation des features
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Blocs de convolutions résiduelles
        self.layer1 = ResidualBlock(32, 64, stride=2)   # Shape -> [64, 32, 32]
        self.layer2 = ResidualBlock(64, 128, stride=2)  # Shape -> [128, 16, 16]
        self.layer3 = ResidualBlock(128, 256, stride=2) # Shape -> [256, 8, 8]
        
        # Pooling global & Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


if __name__ == "__main__":
    # Test d'intégration rapide : vérification des shapes et du nombre de paramètres
    model = CommandSense(num_classes=35)
    
    # Simulation d'un batch de spectrogrammes Log-Mel [Batch=4, Channels=1, Mel_bins=64, Frames=63]
    dummy_input = torch.randn(4, 1, 64, 63)
    dummy_output = model(dummy_input)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("--- Verification Architecture CommandSense ---")
    print(f"Shape de l'entrée  : {dummy_input.shape}")
    print(f"Shape de la sortie : {dummy_output.shape} (Attendu: [4, 35])")
    print(f"Nombre total de paramètres entraînabiles : {num_params:,}")