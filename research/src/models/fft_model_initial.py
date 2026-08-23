import torch
import torch.nn as nn


class FFTDetectorInitial(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def fft_transform(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft))
        mean = mag.mean(dim=(-2, -1), keepdim=True)
        std = mag.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (mag - mean) / std

    def forward(self, x):
        x = self.fft_transform(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x