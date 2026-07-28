import argparse
from src.models.cnn_model import CNNDetector
from src.training.trainer import TrainConfig, train

# Training configuration for the CNN detector
CNN_CONFIG = TrainConfig(
    model_type="cnn",
    model_label="cnn",
    checkpoint_filename="best_cnn.pt",
    lr=1e-4,
    default_epochs=10,
    patience=4,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN detector")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else CNN_CONFIG.default_epochs
    model = CNNDetector()
    train(model, CNN_CONFIG, stage=args.stage, epochs=epochs)
