import warnings
import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Run model inference over a DataLoader and return true labels + sigmoid probabilities
def collect_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


# Compute standard classification metrics from true labels and predicted probabilities
def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


# Convenience wrapper: collect predictions then compute metrics
def evaluate_logits(model, loader, device):
    labels, probs = collect_predictions(model, loader, device)
    return compute_metrics(labels, probs)
