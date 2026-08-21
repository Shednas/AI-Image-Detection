import argparse
from src.models.hybrid_norm_model import HybridNormDetector
from src.training.trainer import TrainConfig, train

# Training configuration for the Hybrid CNN+FFT detector with L2 normalised branches
HYBRID_NORM_CONFIG = TrainConfig(
    model_type="hybrid_norm",
    model_label="hybrid_norm",
    checkpoint_filename="best_hybrid_norm.pt",
    lr=5e-5,
    default_epochs=15,
    patience=5,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid CNN+FFT detector with normalised branches")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else HYBRID_NORM_CONFIG.default_epochs
    model = HybridNormDetector(image_size=256, num_bands=4)
    train(model, HYBRID_NORM_CONFIG, stage=args.stage, epochs=epochs)
