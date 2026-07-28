import torch
import torch.nn as nn


# Frequency-domain detector that learns to weight different spectral bands
class FFTDetector(nn.Module):
    def __init__(self, image_size: int = 256, num_bands: int = 4):
        super().__init__()
        self.image_size = image_size
        self.num_bands = num_bands

        # Learnable softmax weights that control each band's contribution
        self.band_logits = nn.Parameter(torch.zeros(num_bands))
        self.register_buffer("band_masks", self._build_band_masks(image_size, image_size))

        hann_1d = torch.hann_window(image_size)
        hann_2d = torch.outer(hann_1d, hann_1d).unsqueeze(0).unsqueeze(0)
        self.register_buffer("hann_window", hann_2d)

        # Separate shallow CNN per band to extract spatial features from each frequency ring
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout2d(0.15),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            for _ in range(num_bands)
        ])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * num_bands, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    # Partition the FFT spectrum into concentric annular frequency bands
    def _build_band_masks(self, h: int, w: int) -> torch.Tensor:
        center_h, center_w = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        dist = torch.sqrt(((y - center_h) ** 2 + (x - center_w) ** 2).float())
        max_dist = torch.sqrt(torch.tensor(center_h ** 2 + center_w ** 2, dtype=torch.float32))
        min_radius = 8.0
        usable = max_dist - min_radius
        masks = []
        for i in range(self.num_bands):
            low = min_radius + (i / self.num_bands) * usable
            high = min_radius + ((i + 1) / self.num_bands) * usable
            masks.append(((dist >= low) & (dist < high)).float())
        return torch.stack(masks, dim=0).unsqueeze(1)

    # Convert image to a normalised log-magnitude FFT spectrum
    def fft_transform(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=1, keepdim=True)
        x = x * self.hann_window
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft))
        mean = mag.mean(dim=(-2, -1), keepdim=True)
        std = mag.std(dim=(-2, -1), keepdim=True) + 1e-4
        return (mag - mean) / std

    # Apply band masks, scale by learned weights, and extract per-band features
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        fft_mag = self.fft_transform(x)
        band_weights = torch.softmax(self.band_logits, dim=0)
        band_features = []
        for i in range(self.num_bands):
            band = fft_mag * self.band_masks[i]
            band = band * band_weights[i]
            features = self.feature_extractors[i](band)
            band_features.append(features)
        combined = torch.cat(band_features, dim=1)
        return combined.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
