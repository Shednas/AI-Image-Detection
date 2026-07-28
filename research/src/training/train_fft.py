import argparse
from src.models.fft_model import FFTDetector
from src.training.trainer import TrainConfig, train

# Training configuration for the FFT detector
FFT_CONFIG = TrainConfig(
    model_type="fft",
    model_label="fft_improved",
    checkpoint_filename="best_fft.pt",
    lr=2e-4,
    default_epochs=18,
    patience=6,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FFT detector")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else FFT_CONFIG.default_epochs
    model = FFTDetector(image_size=256, num_bands=4)
    train(model, FFT_CONFIG, stage=args.stage, epochs=epochs)
