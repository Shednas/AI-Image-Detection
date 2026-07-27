import io
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.cnn_model import CNNDetector
from models.fft_model import FFTDetector
from models.hybrid_model import HybridDetector
from models.stm_model import STMDetector


TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MODEL_DISPLAY_NAMES = {
    "cnn": "Spatial_CNN",
    "fft": "Frequency_FFT",
    "hybrid": "Hybrid_Fusion",
    "stm": "Handcrafted_STM",
}

WEIGHTS_DIR = Path(__file__).parent / "models" / "weights"


class InferencePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = None
        self.fft = None
        self.hybrid = None
        self.stm = None

    # warm up all four models at startup so first request has no cold-start delay
    def load_models(self, weights_dir: Path = WEIGHTS_DIR) -> None:
        print(f"Loading models on {self.device}...")

        self.cnn = CNNDetector().to(self.device)
        self.cnn.load_state_dict(
            torch.load(weights_dir / "best_cnn.pt", map_location=self.device, weights_only=True)
        )
        self.cnn.eval()

        self.fft = FFTDetector(image_size=256, num_bands=4).to(self.device)
        self.fft.load_state_dict(
            torch.load(weights_dir / "best_fft.pt", map_location=self.device, weights_only=True)
        )
        self.fft.eval()

        self.hybrid = HybridDetector(image_size=256, num_bands=4).to(self.device)
        self.hybrid.load_state_dict(
            torch.load(weights_dir / "best_hybrid.pt", map_location=self.device, weights_only=True)
        )
        self.hybrid.eval()

        joblib_files = list(weights_dir.glob("*.joblib"))
        if not joblib_files:
            raise FileNotFoundError(f"No .joblib file found in {weights_dir}")
        self.stm = STMDetector(lgbm_checkpoint=str(joblib_files[0]))
        self.stm.eval()

        print("All models ready.")

    # load() decodes the whole image, verify() only reads the header. A truncated
    # JPEG passes verify() and then crashes inside preprocess with a 500.
    def validate_image(self, image_bytes: bytes) -> bool:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()
            return True
        except Exception:
            return False

    # normalise to ImageNet stats to match training distribution
    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)
        return tensor

    # dispatch to the selected model and measure wall-clock latency
    def predict(self, image_tensor: torch.Tensor, model_name: str) -> dict:
        model_map = {
            "cnn": self.cnn,
            "fft": self.fft,
            "hybrid": self.hybrid,
            "stm": self.stm,
        }
        model = model_map.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model: '{model_name}'. Choose from: cnn, fft, hybrid, stm")

        t0 = time.perf_counter()
        with torch.no_grad():
            logit = model(image_tensor).squeeze()
            p_real = torch.sigmoid(logit).item()
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # named p_real rather than probability so a stale caller fails loudly
        # instead of silently reading P(real) where it wanted P(AI)
        return {
            "model_name": MODEL_DISPLAY_NAMES[model_name],
            "p_real": round(p_real, 4),          # training mapping is {ai: 0, real: 1}
            "p_ai": round(1.0 - p_real, 4),      # what the UI displays
            "verdict": "AUTHENTIC" if p_real >= 0.5 else "AI_GENERATED",
            "latency_ms": latency_ms,
        }

    # convenience wrapper used in evaluation/comparison workflows
    def predict_all(self, image_tensor: torch.Tensor) -> dict:
        return {key: self.predict(image_tensor, key) for key in ("cnn", "fft", "hybrid", "stm")}
