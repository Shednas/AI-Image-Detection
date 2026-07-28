import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from src.data.dataset_loader import get_loaders
from src.evaluation.metrics import evaluate_logits


STAGE_DATA_DIRS: dict[int, list[str]] = {
    1: ["data/processed/stage_1"],
    2: ["data/processed/stage_1", "data/processed/stage_2"],
    3: ["data/processed/stage_1", "data/processed/stage_2", "data/processed/stage_3"],
}


@dataclass
class TrainConfig:
    model_type: str
    model_label: str
    checkpoint_filename: str
    lr: float
    default_epochs: int
    batch_size: int = 16
    image_size: int = 256
    patience: int = 3


# Full training loop with mixed-precision support, early stopping, and test evaluation
def train(model: nn.Module, config: TrainConfig, stage: int, epochs: int) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    use_amp = use_cuda
    scaler = GradScaler("cuda") if use_amp else None

    data_dirs = STAGE_DATA_DIRS[stage]
    train_loader, val_loader, test_loader, _ = get_loaders(
        data_dir=data_dirs,
        batch_size=config.batch_size,
        image_size=config.image_size,
    )

    print(f"Stage {stage} | {config.model_label} | device: {device}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # Halve LR when validation F1 stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    checkpoint_dir = Path(f"checkpoints/{config.model_type}/stage_{stage}")
    results_dir = Path(f"results/{config.model_type}/stage_{stage}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / config.checkpoint_filename

    best_val_f1 = -1.0
    patience_counter = 0
    training_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            optimizer.zero_grad()

            amp_ctx = autocast("cuda") if use_amp else contextlib.nullcontext()
            with amp_ctx:
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_logits(model, val_loader, device)
        val_f1 = val_metrics["f1"]

        scheduler.step(val_f1)

        training_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_f1": round(val_f1, 4),
            "val_auc": round(val_metrics["roc_auc"], 4),
        })

        print(
            f"  epoch {epoch:>3}/{epochs} | "
            f"loss {avg_train_loss:.4f} | "
            f"val_f1 {val_f1:.4f} | val_auc {val_metrics['roc_auc']:.4f}"
        )

        # Save checkpoint on improvement; stop early if patience is exhausted
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"saved (val_f1 {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_metrics = evaluate_logits(model, test_loader, device)

    with open(results_dir / "training_history.json", "w") as f:
        json.dump({"stage": stage, "model": config.model_label, "epochs": training_history}, f, indent=2)

    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n{config.model_label} stage {stage} test: {test_metrics}")
