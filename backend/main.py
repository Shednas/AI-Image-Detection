import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from batch_handler import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, BatchProcessor
from database.database import DatabaseManager
from pipeline import InferencePipeline, ModelName
from results import ResultsHandler
from session import SessionTracker

# content_type is supplied by the client, so this only screens obvious
# mismatches. The decode in validate_image is what actually proves the bytes
# are an image.
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = MAX_FILE_SIZE_BYTES // (1024 * 1024)

db = DatabaseManager()
session_tracker = SessionTracker(db)
pipeline = InferencePipeline()
batch_processor = BatchProcessor()
results_handler = ResultsHandler()


# load models at startup to avoid cold inference on first request
@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load_models()
    yield


app = FastAPI(title="AI Image Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# validate before inference to avoid pytorch errors on bad input
@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    model_name: ModelName = Form(...),
    session_id: str | None = Form(default=None),
):
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {MAX_FILE_SIZE_MB}MB limit.",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS or file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a JPEG, PNG or WebP image.",
        )

    if not pipeline.validate_image(image_bytes):
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

    # the rest of the pipeline compares model_name against plain strings
    model_key = model_name.value
    session_id = session_tracker.resolve_session(session_id)
    request_id = str(uuid.uuid4())
    image_tensor = pipeline.preprocess(image_bytes)
    raw = pipeline.predict(image_tensor, model_key)

    model_obj = getattr(pipeline, model_key)
    formatted = results_handler.format_single(raw, image_bytes, model_key, image_tensor, model_obj)

    db.save_inference_request({
        "request_id": request_id,
        "session_id": session_id,
        "batch_id": None,
        "file_name": file.filename,
        "final_verdict": raw["verdict"],
        "consensus_score": raw["p_ai"],
    })
    db.save_model_output({
        "output_id": str(uuid.uuid4()),
        "request_id": request_id,
        "model_name": raw["model_name"],
        "predicted_probability": raw["p_ai"],
        "latency_ms": raw["latency_ms"],
    })

    # echoed back so the client can send it again and stay in one session
    formatted["session_id"] = session_id
    return formatted


# skip empty/corrupt images but process the rest
@app.post("/api/batch")
async def analyze_batch(
    file: UploadFile = File(...),
    model_name: ModelName = Form(...),
    session_id: str | None = Form(default=None),
):
    zip_bytes = await file.read()
    session_id = session_tracker.resolve_session(session_id)
    batch_id = str(uuid.uuid4())

    files = batch_processor.extract_zip(zip_bytes)
    if not files:
        raise HTTPException(status_code=400, detail="No valid images found in the zip file.")

    db.save_batch(batch_id, session_id, total_files=len(files))
    raw_results = batch_processor.process_batch(files, model_name.value, pipeline)

    # store p_ai, not probability: the analyze endpoint writes P(AI) into these
    # same two columns, so writing P(real) here would put both meanings in one
    # column with nothing to tell them apart
    for r in raw_results:
        if "error" in r:
            continue
        request_id = str(uuid.uuid4())
        db.save_inference_request({
            "request_id": request_id,
            "session_id": session_id,
            "batch_id": batch_id,
            "file_name": r["file_name"],
            "final_verdict": r["verdict"],
            "consensus_score": r["p_ai"],
        })
        db.save_model_output({
            "output_id": str(uuid.uuid4()),
            "request_id": request_id,
            "model_name": r["model_name"],
            "predicted_probability": r["p_ai"],
            "latency_ms": r["latency_ms"],
        })

    summary = results_handler.format_batch_summary(raw_results)
    summary["session_id"] = session_id
    return summary


# thin pass-through; filtering is done in the DB layer
@app.get("/api/history")
async def get_history(
    search: str = Query(default=None),
    category: str = Query(default=None),
):
    return db.get_history(search=search, category=category)


# liveness probe for deployment checks
@app.get("/api/health")
async def health():
    return {"status": "ok"}
