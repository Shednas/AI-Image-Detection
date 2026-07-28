import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.evaluation.metrics import collect_predictions, compute_metrics

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

UNSEEN_DIRS = {
    "chameleon": "data/unseen/chameleon",
    "mnw": "data/unseen/MNW",
}


def _eval_transform(image_size: int = 256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class _ImageList(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _find_images(directory: str | Path) -> list[Path]:
    return [p for p in Path(directory).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]


# Sample a balanced real/fake split from the Chameleon dataset.
# Labels follow the training mapping, {ai_generated: 0, real: 1}, so the sigmoid
# output is P(real). Reversing them here inverts every metric below.
def _chameleon_samples(base_dir: str, n_samples: int, seed: int) -> tuple[list, bool]:
    rng = random.Random(seed)
    fakes = _find_images(Path(base_dir) / "fake")
    reals = _find_images(Path(base_dir) / "real")
    half = n_samples // 2
    fakes = rng.sample(fakes, min(half, len(fakes)))
    reals = rng.sample(reals, min(half, len(reals)))
    samples = [(str(p), 0) for p in fakes] + [(str(p), 1) for p in reals]
    return samples, True


# Sample AI-only MNW images, which have no real counterpart
def _mnw_samples(base_dir: str, n_samples: int, seed: int) -> tuple[list, bool]:
    rng = random.Random(seed)
    images = _find_images(base_dir)
    images = rng.sample(images, min(n_samples, len(images)))
    return [(str(p), 0) for p in images], False


# Evaluate a trained model on an unseen dataset and save results to JSON
def run_unseen_eval(
    model: nn.Module,
    stage: int,
    model_type: str,
    dataset: str,
    n_samples: int,
    image_size: int = 256,
    batch_size: int = 16,
    seed: int = 42,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    base_dir = UNSEEN_DIRS[dataset.lower()]

    if dataset.lower() == "chameleon":
        samples, has_real = _chameleon_samples(base_dir, n_samples, seed)
    elif dataset.lower() == "mnw":
        samples, has_real = _mnw_samples(base_dir, n_samples, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from: {list(UNSEEN_DIRS)}")

    loader = DataLoader(
        _ImageList(samples, _eval_transform(image_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\nUnseen eval | {model_type} stage {stage} | dataset: {dataset} | n={len(samples)}")

    labels, probs = collect_predictions(model, loader, device)

    if has_real:
        metrics = compute_metrics(labels, probs)
    else:
        # AI-only set, so detection rate replaces the classification metrics.
        # probs is P(real), so an image is detected when it falls below the
        # threshold, not above it.
        detected = (probs < 0.5).astype(float)
        metrics = {
            "detection_rate": float(detected.mean()),
            "n_samples": int(len(detected)),
            "note": "all AI-generated, no real images, so only detection rate is reported",
        }

    results_dir = Path(f"results/{model_type}/stage_{stage}")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"unseen_{dataset.lower()}.json"
    with open(out_path, "w") as f:
        json.dump({"stage": stage, "model": model_type, "dataset": dataset, **metrics}, f, indent=2)

    print(f"Saved → {out_path}")
    print(metrics)
