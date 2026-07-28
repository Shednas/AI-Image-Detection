from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from PIL import Image

from src.evaluation.metrics import compute_metrics
from src.models.stm_model import STMFeatureExtractor
from src.training.trainer import STAGE_DATA_DIRS


LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "goss",
    "num_leaves": 127,
    "max_depth": 7,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "top_rate": 0.2,
    "other_rate": 0.1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "random_state": 42,
}

_IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


# Gather image paths and integer class labels from a dataset split directory
def _collect_paths_and_labels(stage_dirs: list[str], split: str):
    first_split = Path(stage_dirs[0]) / split
    classes = sorted(d.name for d in first_split.iterdir() if d.is_dir())

    all_paths: list[Path] = []
    all_labels: list[int] = []

    for stage_dir in stage_dirs:
        split_dir = Path(stage_dir) / split
        for label_idx, class_name in enumerate(classes):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            for ext in _IMG_EXTS:
                for p in sorted(class_dir.glob(ext)):
                    all_paths.append(p)
                    all_labels.append(label_idx)

    return all_paths, all_labels, classes


# Extract handcrafted features for a split, using disk cache to avoid re-extraction
def extract_features(stage_dirs, split, extractor, image_size, cache_path):
    if cache_path.exists():
        print(f"  [cache] {cache_path}")
        data = np.load(cache_path, allow_pickle=True).item()
        return data["features"], data["labels"]

    paths, labels, classes = _collect_paths_and_labels(stage_dirs, split)
    n = len(paths)
    print(f"  Extracting {split}: {n} images, classes={classes}")

    features_list = []
    t0 = time.perf_counter()

    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        feat = extractor.extract(np.array(img, dtype=np.uint8))
        features_list.append(feat)

        if (i + 1) % 500 == 0 or (i + 1) == n:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"    {i+1:>6}/{n}  ({rate:.0f} img/s)")

    features = np.stack(features_list).astype(np.float32)
    labels_np = np.array(labels, dtype=np.int32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, {"features": features, "labels": labels_np})
    print(f"  Cached -> {cache_path}  shape={features.shape}")

    return features, labels_np


# Train LightGBM on cached STM features with early stopping on val loss
def train(stage: int) -> None:
    image_size = 256
    extractor = STMFeatureExtractor(image_size=image_size)
    stage_dirs = STAGE_DATA_DIRS[stage]
    cache_root = Path(f"checkpoints/stm/stage_{stage}/feature_cache")
    ckpt_dir = Path(f"checkpoints/stm/stage_{stage}")
    results_dir = Path(f"results/stm/stage_{stage}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = extract_features(stage_dirs, "train", extractor, image_size, cache_root / "train.npy")
    X_val, y_val = extract_features(stage_dirs, "validation", extractor, image_size, cache_root / "val.npy")
    X_test, y_test = extract_features(stage_dirs, "test", extractor, image_size, cache_root / "test.npy")

    print(f"\nTrain: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = clf.predict_proba(X)[:, 1]
        metrics = compute_metrics(y.astype(int), proba)
        print(f"  {split_name}: f1 {metrics['f1']:.4f} | auc {metrics['roc_auc']:.4f}")

    test_proba = clf.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test.astype(int), test_proba)

    with open(results_dir / "test_metrics.json", "w") as fh:
        json.dump(test_metrics, fh, indent=2)

    with open(results_dir / "training_history.json", "w") as fh:
        json.dump({
            "stage": stage,
            "model": "stm",
            "best_iteration": clf.best_iteration_,
            "n_features": int(X_train.shape[1]),
            "lgbm_params": LGBM_PARAMS,
        }, fh, indent=2)

    importance = clf.feature_importances_
    top_idx = np.argsort(importance)[::-1][:20]
    with open(results_dir / "feature_importance.json", "w") as fh:
        json.dump({f"feature_{int(i)}": float(importance[i]) for i in top_idx}, fh, indent=2)

    ckpt_path = ckpt_dir / "stm_model.joblib"
    joblib.dump(clf, ckpt_path)
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"STM stage {stage} test: {test_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train STM detector")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()
    train(args.stage)
