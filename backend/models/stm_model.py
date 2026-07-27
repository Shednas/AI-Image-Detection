from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import joblib

from skimage.feature import hog, local_binary_pattern
from scipy.ndimage import gaussian_filter
from scipy.fft import dctn
from scipy.stats import kurtosis as _kurtosis, skew as _skew


class STMFeatureExtractor:
    def __init__(self, image_size: int = 256):
        self.image_size = image_size

    def extract(self, image: np.ndarray) -> np.ndarray:
        img  = image.astype(np.float32) / 255.0
        gray = (0.2989 * img[:, :, 0]
                + 0.5870 * img[:, :, 1]
                + 0.1140 * img[:, :, 2])

        return np.concatenate([
            self._hog_features(gray),
            self._lbp_features(gray),
            self._dct_features(gray),
            self._colour_features(img),
            self._noise_features(img),
        ]).astype(np.float32)

    def _hog_features(self, gray: np.ndarray) -> np.ndarray:
        return hog(
            gray,
            orientations=9,
            pixels_per_cell=(32, 32),
            cells_per_block=(2, 2),
            feature_vector=True,
        )

    def _lbp_features(self, gray: np.ndarray) -> np.ndarray:
        gray_u8 = (gray * 255).astype(np.uint8)
        feats = []
        for P, R in [(8, 1), (16, 2)]:
            lbp  = local_binary_pattern(gray_u8, P=P, R=R, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2))
            feats.append(hist / (hist.sum() + 1e-8))
        return np.concatenate(feats)

    def _dct_features(self, gray: np.ndarray) -> np.ndarray:
        small = gray[::4, ::4]
        dct   = np.abs(dctn(small, norm="ortho"))
        zones = [
            dct[:8,   :8  ].ravel(),
            dct[8:32, 8:32].ravel(),
            dct[32:,  32: ].ravel(),
        ]
        feats = []
        for zone in zones:
            feats.extend([zone.mean(), zone.std()])
        return np.array(feats, dtype=np.float32)

    def _colour_features(self, img: np.ndarray) -> np.ndarray:
        feats = []
        for ch in range(3):
            channel = img[:, :, ch].ravel()
            feats.extend([
                float(channel.mean()),
                float(channel.std()),
                float(_skew(channel)),
                float(_kurtosis(channel)),
            ])
        return np.array(feats, dtype=np.float32)

    def _noise_features(self, img: np.ndarray) -> np.ndarray:
        feats = []
        for ch in range(3):
            channel = img[:, :, ch]
            noise   = channel - gaussian_filter(channel, sigma=2.0)
            flat    = noise.ravel()
            feats.extend([
                float(flat.mean()),
                float(flat.std()),
                float(_kurtosis(flat)),
                float(self._shannon_entropy(flat)),
            ])
        return np.array(feats, dtype=np.float32)

    @staticmethod
    def _shannon_entropy(values: np.ndarray, bins: int = 64) -> float:
        hist, _ = np.histogram(values, bins=bins)
        p = hist / (hist.sum() + 1e-8)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p + 1e-12)))


class STMDetector(nn.Module):
    _MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    _STD:  tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(self, lgbm_checkpoint: str, image_size: int = 256):
        super().__init__()
        self.feature_extractor = STMFeatureExtractor(image_size=image_size)
        self.lgbm_model        = joblib.load(lgbm_checkpoint)
        self._image_size       = image_size

        mean = torch.tensor(self._MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(self._STD ).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std",  std)

    def _to_uint8_hwc(self, x: torch.Tensor) -> np.ndarray:
        raw = (x.cpu() * self._std.cpu() + self._mean.cpu()).clamp(0.0, 1.0)
        return (raw * 255).byte().numpy().transpose(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        images_np = self._to_uint8_hwc(x)
        features  = np.stack([self.feature_extractor.extract(img) for img in images_np])
        proba     = self.lgbm_model.predict_proba(features)[:, 1]
        proba     = np.clip(proba, 1e-7, 1.0 - 1e-7)
        logits    = np.log(proba / (1.0 - proba))
        return torch.tensor(logits, dtype=torch.float32).unsqueeze(1).to(x.device)