import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .fft_model import FFTDetector


# Late fusion variant that projects the CNN branch down to the FFT branch width.
# HybridDetector equalises the scale of the two branches but not their width, so
# 2048 of the 2304 fusion inputs still come from the CNN side. This tests whether
# dimensional parity matters on top of scale parity, so the normalisation stays.
class HybridProjDetector(nn.Module):
    CNN_DIM: int = 2048

    def __init__(self, image_size: int = 256, num_bands: int = 4):
        super().__init__()
        fft_dim = 64 * num_bands

        # ResNet-50 with identity fc to produce 2048-dim spatial embeddings
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.cnn_branch = backbone

        self.fft_branch = FFTDetector(image_size=image_size, num_bands=num_bands)

        self.cnn_proj = nn.Linear(self.CNN_DIM, fft_dim)

        # MLP that classifies from the concatenated CNN and FFT embeddings
        self.fusion = nn.Sequential(
            nn.Linear(fft_dim + fft_dim, 512),
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
        # the branches are 2048-dim and 256-dim with different natural scales, so
        # unnormalised concatenation lets the CNN side dominate the fusion input
        cnn_feat = F.normalize(cnn_feat, dim=1)
        fft_feat = F.normalize(fft_feat, dim=1)
        cnn_feat = self.cnn_proj(cnn_feat)
        combined = torch.cat([cnn_feat, fft_feat], dim=1)
        return self.fusion(combined)
