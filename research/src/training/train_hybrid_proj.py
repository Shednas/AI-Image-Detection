import argparse
from src.models.hybrid_proj_model import HybridProjDetector
from src.training.trainer import TrainConfig, train

# Training configuration for the Hybrid CNN+FFT detector with a projected CNN branch
HYBRID_PROJ_CONFIG = TrainConfig(
    model_type="hybrid_proj",
    model_label="hybrid_proj",
    checkpoint_filename="best_hybrid_proj.pt",
    lr=5e-5,
    default_epochs=15,
    patience=5,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid CNN+FFT detector with CNN projection")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else HYBRID_PROJ_CONFIG.default_epochs
    model = HybridProjDetector(image_size=256, num_bands=4)
    train(model, HYBRID_PROJ_CONFIG, stage=args.stage, epochs=epochs)
