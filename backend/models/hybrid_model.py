import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from models.fft_model import FFTDetector


class HybridDetector(nn.Module):
    CNN_DIM: int = 2048

    def __init__(self, image_size: int = 256, num_bands: int = 4):
        super().__init__()
        fft_dim = 64 * num_bands

        backbone    = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.cnn_branch = backbone

        self.fft_branch = FFTDetector(image_size=image_size, num_bands=num_bands)

        self.fusion = nn.Sequential(
            nn.Linear(self.CNN_DIM + fft_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_branch(x)
        fft_feat = self.fft_branch.forward_features(x)
        combined = torch.cat([cnn_feat, fft_feat], dim=1)
        return self.fusion(combined)