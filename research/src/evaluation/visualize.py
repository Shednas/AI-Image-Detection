import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch
import joblib
from PIL import Image

from src.data.dataset_loader import get_loaders
from src.models.cnn_model import CNNDetector
from src.models.fft_model import FFTDetector
from src.models.hybrid_model import HybridDetector
from src.models.stm_model import STMFeatureExtractor
from src.evaluation.metrics import collect_predictions
from src.training.trainer import STAGE_DATA_DIRS


# Denormalise tensors back to uint8, extract STM features, and run LightGBM inference
def collect_stm_predictions(model, extractor, loader, image_size=256):
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images_np = images.cpu().numpy()

        for i in range(len(images_np)):
            img_array = ((images_np[i].transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
            features = extractor.extract(img_array)
            prob = model.predict_proba([features])[0, 1]
            all_probs.append(prob)
            all_labels.append(labels[i].item())

    return np.array(all_labels), np.array(all_probs)


# Plot training loss and validation metrics across epochs
def plot_training_curves(stage, model_type):
    results_dir = Path(f"results/{model_type}/stage_{stage}")
    history_file = results_dir / "training_history.json"

    if not history_file.exists():
        print(f"Training history not found: {history_file}")
        return

    with open(history_file, "r") as f:
        history = json.load(f)

    if model_type == "stm":  # no per-epoch data
        return

    epochs = [e["epoch"] for e in history["epochs"]]
    train_loss = [e["train_loss"] for e in history["epochs"]]
    val_f1 = [e["val_f1"] for e in history["epochs"]]
    val_auc = [e["val_auc"] for e in history["epochs"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_type.upper()} Training Loss - Stage {stage}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_f1, marker="o", label="Val F1", linewidth=2, color="green")
    axes[1].plot(epochs, val_auc, marker="s", label="Val AUC", linewidth=2, color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{model_type.upper()} Validation Metrics - Stage {stage}")
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output_path = results_dir / "plots" / "training_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# Render confusion matrix with per-cell counts
def plot_confusion_matrix(confusion_mat, stage, model_type, classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_mat, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(confusion_mat.shape[1]),
        yticks=np.arange(confusion_mat.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            color = "white" if confusion_mat[i, j] > confusion_mat.max() / 2 else "black"
            ax.text(j, i, str(confusion_mat[i, j]), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    plt.title(f"{model_type.upper()} Confusion Matrix - Stage {stage}")
    output_path = Path(f"results/{model_type}/stage_{stage}/plots/confusion_matrix.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# Plot ROC curve with AUC score
def plot_roc_curve(y_true, y_pred_prob, stage, model_type):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_type.upper()} ROC Curve - Stage {stage}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    output_path = Path(f"results/{model_type}/stage_{stage}/plots/roc_curve.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# Histogram of model confidence scores split by true class
def plot_confidence_distribution(predictions, labels, stage, model_type):
    real_scores = predictions[labels == 0]
    ai_scores = predictions[labels == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(real_scores, bins=20, alpha=0.6, label="Real Images", color="blue", edgecolor="black")
    plt.hist(ai_scores, bins=20, alpha=0.6, label="AI Images", color="red", edgecolor="black")
    plt.xlabel("Model Confidence P(AI)")
    plt.ylabel("Count")
    plt.title(f"{model_type.upper()} Confidence Distribution - Stage {stage}")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    output_path = Path(f"results/{model_type}/stage_{stage}/plots/confidence_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# Load the requested model checkpoint and generate all four diagnostic plots
def generate_visualizations(stage, model_type):
    print(f"\nGenerating visualizations for {model_type.upper()} Stage {stage}...")

    if stage not in STAGE_DATA_DIRS:
        raise ValueError(f"Invalid stage: {stage}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dirs = STAGE_DATA_DIRS[stage]

    _, _, test_loader, classes = get_loaders(data_dirs, batch_size=16, image_size=256)

    checkpoint_map = {
        "cnn": f"checkpoints/cnn/stage_{stage}/best_cnn.pt",
        "fft": f"checkpoints/fft/stage_{stage}/best_fft.pt",
        "hybrid": f"checkpoints/hybrid/stage_{stage}/best_hybrid.pt",
        "stm": f"checkpoints/stm/stage_{stage}/stm_model.joblib",
    }
    checkpoint_path = checkpoint_map[model_type]

    if not Path(checkpoint_path).exists():
        print(f"Model not found: {checkpoint_path}")
        return

    if model_type == "cnn":
        model = CNNDetector().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        all_labels, probs = collect_predictions(model, test_loader, device)
    elif model_type == "fft":
        model = FFTDetector(image_size=256, num_bands=4).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        all_labels, probs = collect_predictions(model, test_loader, device)
    elif model_type == "hybrid":
        model = HybridDetector(image_size=256, num_bands=4).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        all_labels, probs = collect_predictions(model, test_loader, device)
    elif model_type == "stm":
        model = joblib.load(checkpoint_path)
        extractor = STMFeatureExtractor(image_size=256)
        all_labels, probs = collect_stm_predictions(model, extractor, test_loader, image_size=256)

    preds = (probs > 0.5).astype(int)
    conf_mat = confusion_matrix(all_labels, preds)

    plot_training_curves(stage, model_type)
    plot_confusion_matrix(conf_mat, stage, model_type, classes)
    plot_roc_curve(all_labels, probs, stage, model_type)
    plot_confidence_distribution(probs, all_labels, stage, model_type)

    print(f"All visualizations saved to results/{model_type}/stage_{stage}/plots/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations for trained models")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "fft", "hybrid", "stm"])
    args = parser.parse_args()
    generate_visualizations(args.stage, args.model)
