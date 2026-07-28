import io
import logging
import threading
import time
from enum import Enum
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.cnn_model import CNNDetector
from models.fft_model import FFTDetector
from models.hybrid_model import HybridDetector
from models.stm_model import STMDetector

logger = logging.getLogger("ai_detection.pipeline")


# an enum so FastAPI rejects an unknown model with 422, rather than predict
# raising and surfacing as a 500
class ModelName(str, Enum):
    cnn = "cnn"
    fft = "fft"
    hybrid = "hybrid"
    stm = "stm"


# the name was valid but its weights did not load: service state, not a caller
# mistake
class ModelUnavailable(RuntimeError):
    pass


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
        # never returned to the browser: the text carries local paths
        self.load_errors: dict[str, str] = {}
        # one model instance is shared by every request, and Grad-CAM hangs hooks
        # on a layer that other requests also pass through. Every pass is taken
        # under this lock, whether it registers hooks or not.
        self.model_lock = threading.RLock()

    def _load_torch(self, model, filename: str, weights_dir: Path):
        model = model.to(self.device)
        model.load_state_dict(
            torch.load(weights_dir / filename, map_location=self.device, weights_only=True)
        )
        model.eval()
        return model

    # glob rather than a fixed name: train_stm.py writes a run-specific filename
    def _load_stm(self, weights_dir: Path):
        joblib_files = list(weights_dir.glob("*.joblib"))
        if not joblib_files:
            raise FileNotFoundError(f"No .joblib file found in {weights_dir}")
        model = STMDetector(lgbm_checkpoint=str(joblib_files[0]))
        model.eval()
        return model

    # loaded independently so a bad checkpoint costs one model, not the server
    def load_models(self, weights_dir: Path = WEIGHTS_DIR) -> None:
        logger.info("Loading models on %s", self.device)
        self.load_errors = {}

        builders = {
            "cnn": lambda: self._load_torch(CNNDetector(), "best_cnn.pt", weights_dir),
            "fft": lambda: self._load_torch(
                FFTDetector(image_size=256, num_bands=4), "best_fft.pt", weights_dir
            ),
            "hybrid": lambda: self._load_torch(
                HybridDetector(image_size=256, num_bands=4), "best_hybrid.pt", weights_dir
            ),
            "stm": lambda: self._load_stm(weights_dir),
        }

        for name, build in builders.items():
            try:
                setattr(self, name, build())
            except Exception as e:
                logger.exception("Could not load the %s model", name)
                setattr(self, name, None)
                self.load_errors[name] = f"{type(e).__name__}: {e}"

        ready = [name for name in builders if getattr(self, name) is not None]
        if len(ready) == len(builders):
            logger.info("All models ready.")
        else:
            logger.error(
                "Started with %d of %d models: %s unavailable",
                len(ready), len(builders), ", ".join(sorted(self.load_errors)),
            )

    def is_loaded(self, model_name: str) -> bool:
        return getattr(self, model_name, None) is not None

    def model_status(self) -> dict:
        return {name.value: self.is_loaded(name.value) for name in ModelName}

    # load() decodes the whole image, verify() only reads the header. A truncated
    # JPEG passes verify() and then crashes inside preprocess with a 500.
    def validate_image(self, image_bytes: bytes) -> bool:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()
            return True
        except Exception:
            return False

    # ImageNet stats to match the training distribution
    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image_tensor: torch.Tensor, model_name: str) -> dict:
        model_map = {
            "cnn": self.cnn,
            "fft": self.fft,
            "hybrid": self.hybrid,
            "stm": self.stm,
        }
        if model_name not in model_map:
            raise ValueError(f"Unknown model: '{model_name}'. Choose from: cnn, fft, hybrid, stm")

        # separate from the check above, which would otherwise report an unloaded
        # model as a typo
        model = model_map[model_name]
        if model is None:
            raise ModelUnavailable(f"The {model_name} model is not loaded.")

        t0 = time.perf_counter()
        with self.model_lock, torch.no_grad():
            logit = model(image_tensor).squeeze()
            p_real = torch.sigmoid(logit).item()
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # named p_real rather than probability so a stale caller fails loudly
        # instead of silently reading P(real) where it wanted P(AI)
        return {
            "model_name": MODEL_DISPLAY_NAMES[model_name],
            "p_real": round(p_real, 4), # training mapping is {ai: 0, real: 1}
            "p_ai": round(1.0 - p_real, 4), # what the UI displays
            "verdict": "AUTHENTIC" if p_real >= 0.5 else "AI_GENERATED",
            "latency_ms": latency_ms,
        }

    # convenience wrapper used in evaluation/comparison workflows
    def predict_all(self, image_tensor: torch.Tensor) -> dict:
        return {key: self.predict(image_tensor, key) for key in ("cnn", "fft", "hybrid", "stm")}
