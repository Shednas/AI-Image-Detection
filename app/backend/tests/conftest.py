# Shared fixtures. Importing main constructs a DatabaseManager, so every fixture
# here replaces it and the pipeline with stubs; the suite needs no database and no
# model weights. Tests that do need PostgreSQL take postgres_url and skip without it.

import io
import os
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# main constructs a DatabaseManager at import, before any fixture can replace it,
# and that raises if DATABASE_URL is unset. Pointing it at a dead address keeps
# the import working on a machine with no .env, and guarantees no test can reach
# the real database by accident. load_dotenv does not override an existing value,
# so this wins over backend/.env.
os.environ.setdefault("DATABASE_URL", "postgresql://unused:unused@127.0.0.1:1/unused")

SESSION_ID = "11111111-1111-1111-1111-111111111111"


# A valid PNG, for tests that need real image bytes rather than a filename.
def png_bytes(size=(32, 32), colour=(120, 120, 120)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


# Build a zip in memory from (name, bytes) pairs.
def make_zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


class FakeSessionTracker:
    def resolve_session(self, session_id):
        return SESSION_ID


# Records what the endpoints tried to write, and can be told to fail, since a failed
# write should still return the verdict with a warning rather than lose it.
class FakeDb:
    def __init__(self, fail_save_batch=False, fail_requests=False):
        self.fail_save_batch = fail_save_batch
        self.fail_requests = fail_requests
        self.requests = []
        self.outputs = []
        self.batches = []
        self.status_updates = []
        self.up = True

    def ping(self):
        return self.up

    def save_batch(self, batch_id, session_id, total_files):
        if self.fail_save_batch:
            raise RuntimeError("database unavailable")
        self.batches.append((batch_id, session_id, total_files))

    def update_batch_status(self, batch_id, status, processed, skipped):
        self.status_updates.append((status, processed, skipped))

    def save_inference_request(self, data):
        if self.fail_requests:
            raise RuntimeError("database unavailable")
        self.requests.append(data)

    def save_model_output(self, data):
        if self.fail_requests:
            raise RuntimeError("database unavailable")
        self.outputs.append(data)

    def get_history(self, search=None, category=None):
        return []


# Stands in for InferencePipeline without loading 189MB of weights.
class FakePipeline:
    def __init__(self, loaded=None, device="cpu"):
        self.device = device
        self.loaded = loaded or {"cnn": True, "fft": True, "hybrid": True, "stm": True}
        self.cnn = self.fft = self.hybrid = self.stm = object()

    # Called by the lifespan handler, which the TestClient runs on entry.
    def load_models(self, weights_dir=None):
        pass

    def is_loaded(self, name):
        return self.loaded.get(name, False)

    def model_status(self):
        return dict(self.loaded)

    def validate_image(self, image_bytes):
        try:
            Image.open(io.BytesIO(image_bytes)).load()
            return True
        except Exception:
            return False

    def preprocess(self, image_bytes):
        return "tensor"

    def predict(self, tensor, model_name):
        return {
            "model_name": f"fake_{model_name}",
            "p_real": 0.25,
            "p_ai": 0.75,
            "verdict": "AI_GENERATED",
            "latency_ms": 5,
        }


# Skips visualisation rendering, which is slow and not what these tests cover.
class FakeResultsHandler:
    def format_single(self, raw, image_bytes, model_name, image_tensor=None, model=None):
        return {
            "model_name": raw["model_name"],
            "verdict": raw["verdict"],
            "latency_ms": raw["latency_ms"],
            "p_real": raw["p_real"],
            "ai_pct": round(raw["p_ai"] * 100, 1),
            "visualizations": {},
        }

    def format_batch_summary(self, results):
        valid = [r for r in results if "error" not in r]
        return {"total": len(results), "valid": len(valid), "rows": results}


# main with its database, pipeline and results handler replaced by stubs.
@pytest.fixture
def app_module(monkeypatch):
    import main

    monkeypatch.setattr(main, "db", FakeDb())
    monkeypatch.setattr(main, "pipeline", FakePipeline())
    monkeypatch.setattr(main, "session_tracker", FakeSessionTracker())
    monkeypatch.setattr(main, "results_handler", FakeResultsHandler())
    return main


# TestClient returning the 500 response instead of re-raising, so the handler that
# turns an error into a response is exercised rather than bypassed.
@pytest.fixture
def client(app_module):
    from fastapi.testclient import TestClient

    with TestClient(app_module.app, raise_server_exceptions=False) as c:
        yield c


# Connection URL for tests needing a real database, or skip, so a fresh clone with no
# PostgreSQL gets a clean run rather than failures about something never installed.
@pytest.fixture
def postgres_url():
    from dotenv import dotenv_values

    # read .env directly, since os.environ deliberately holds the dead address
    # set at the top of this file
    url = dotenv_values(BACKEND / ".env").get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL is not set, skipping tests that need PostgreSQL")

    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable, skipping: {exc}")

    return url
