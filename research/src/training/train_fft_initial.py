import os
import json
import argparse
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from src.data.dataset_loader import get_loaders
from src.models.fft_model_initial import FFTDetectorInitial
from src.evaluation.metrics import evaluate_logits


def train(stage: int = 1):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    if stage == 1:
        data_dirs = ["data/processed/stage_1"]
    elif stage == 2:
        data_dirs = ["data/processed/stage_1", "data/processed/stage_2"]
    elif stage == 3:
        data_dirs = ["data/processed/stage_1", "data/processed/stage_2", "data/processed/stage_3"]

    print(f"\nStage {stage} Training (FFT Initial Baseline)")

    train_loader, val_loader, test_loader, classes = get_loaders(
        data_dir=data_dirs,
        batch_size=16,
        image_size=256
    )

    model = FFTDetectorInitial().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda") if use_cuda else None

    best_val_f1 = 0.0
    os.makedirs("checkpoints/fft", exist_ok=True)
    os.makedirs(f"results/fft/stage_{stage}", exist_ok=True)

    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_cuda:
                with autocast("cuda"):
                    logits = model(images).squeeze(1)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        val_metrics = evaluate_logits(model, val_loader, device)
        print(f"Stage {stage} | Epoch {epoch} | Loss: {running_loss/len(train_loader):.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), "checkpoints/fft/best_fft_initial.pt")

    model.load_state_dict(torch.load("checkpoints/fft/best_fft_initial.pt", map_location=device, weights_only=True))
    test_metrics = evaluate_logits(model, test_loader, device)

    with open(f"results/fft/stage_{stage}/test_metrics_initial.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nFFT Initial (Stage {stage}) Test Metrics:", test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FFT Initial detector")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    train(stage=args.stage)