import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from batch_handler import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    BatchProcessor,
    ZipRejected,
)
from database.database import BatchStatus, DatabaseManager
from pipeline import InferencePipeline, ModelName
from results import ResultsHandler
from session import SessionTracker

# client-supplied, so this only screens obvious mismatches. The decode in
# validate_image is what proves the bytes are an image.
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ai_detection")

db = DatabaseManager()
session_tracker = SessionTracker(db)
pipeline = InferencePipeline()
batch_processor = BatchProcessor()
results_handler = ResultsHandler(pipeline.model_lock)


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


# keeps stack traces and internal paths out of the browser
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on the server. Please try again."},
    )


# without this, batch returns an error on every file and looks like a completed
# batch rather than a rejected request
def require_model(model_key: str) -> None:
    if not pipeline.is_loaded(model_key):
        logger.error("Rejected a request for %s, which is not loaded", model_key)
        raise HTTPException(
            status_code=503,
            detail=f"The {model_key.upper()} model is unavailable. Try another model.",
        )


# def, not async def: inference, visualisations and database writes all block,
# and on the event loop they served one request at a time
@app.post("/api/analyze")
def analyze_image(
    file: UploadFile = File(...),
    model_name: ModelName = Form(...),
    session_id: str | None = Form(default=None),
):
    # file.read() is the async API
    image_bytes = file.file.read()

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

    model_key = model_name.value
    require_model(model_key)
    session_id = session_tracker.resolve_session(session_id)
    request_id = str(uuid.uuid4())

    try:
        image_tensor = pipeline.preprocess(image_bytes)
        raw = pipeline.predict(image_tensor, model_key)
    except Exception:
        logger.exception("Inference failed for %r using %s", file.filename, model_key)
        raise HTTPException(status_code=500, detail="Could not analyse this image.")

    model_obj = getattr(pipeline, model_key)
    formatted = results_handler.format_single(raw, image_bytes, model_key, image_tensor, model_obj)

    # the row is bookkeeping, so a database failure is reported alongside the
    # verdict rather than replacing it
    warning = None
    try:
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
    except Exception:
        logger.exception("Could not save request %s for %r", request_id, file.filename)
        warning = "This result could not be saved, so it will not appear in History."

    formatted["session_id"] = session_id
    formatted["warning"] = warning
    return formatted


def mark_batch(batch_id: str, status: BatchStatus, processed: int, skipped: int) -> None:
    try:
        db.update_batch_status(batch_id, status, processed, skipped)
    except Exception:
        logger.exception("Could not mark batch %s as %s", batch_id, status.value)


@app.post("/api/batch")
def analyze_batch(
    file: UploadFile = File(...),
    model_name: ModelName = Form(...),
    session_id: str | None = Form(default=None),
):
    zip_bytes = file.file.read()
    # before the archive is opened, unlike analyze: extraction can cost the whole
    # decompression limit and none of it survives a 503
    require_model(model_name.value)
    session_id = session_tracker.resolve_session(session_id)
    batch_id = str(uuid.uuid4())

    try:
        files = batch_processor.extract_zip(zip_bytes)
    except ZipRejected as e:
        # the rejection reason is specific and safe to show, unlike a raw error
        logger.info("Rejected zip %r from batch %s: %s", file.filename, batch_id, e)
        raise HTTPException(status_code=e.status_code, detail=str(e))

    warning = None
    batch_saved = True
    try:
        db.save_batch(batch_id, session_id, total_files=len(files))
    except Exception:
        # the per-image rows carry batch_id as a foreign key, so they cannot be
        # written without this row
        logger.exception("Could not save batch %s", batch_id)
        batch_saved = False
        warning = "This batch could not be saved, so it will not appear in History."

    try:
        raw_results = batch_processor.process_batch(files, model_name.value, pipeline)
    except Exception:
        # process_batch traps per-file errors, so reaching here means the run
        # collapsed as a whole
        logger.exception("Batch %s failed during processing", batch_id)
        if batch_saved:
            mark_batch(batch_id, BatchStatus.failed, processed=0, skipped=len(files))
        raise HTTPException(status_code=500, detail="Could not process this batch.")

    # p_ai, not probability: analyze writes P(AI) into these same two columns, so
    # writing P(real) here would put both meanings in one column
    unsaved = 0
    for r in raw_results:
        if "error" in r or not batch_saved:
            continue
        request_id = str(uuid.uuid4())
        try:
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
        except Exception:
            logger.exception("Could not save request %s for %r", request_id, r["file_name"])
            unsaved += 1

    if unsaved:
        warning = f"{unsaved} of these results could not be saved to History."

    # p_real is dropped before the response: nothing reads it, and leaving the
    # opposite direction beside p_ai invites a future edit to grab the wrong one
    for r in raw_results:
        r.pop("p_real", None)

    processed = sum(1 for r in raw_results if "error" not in r)
    if batch_saved:
        mark_batch(batch_id, BatchStatus.completed, processed, len(raw_results) - processed)

    summary = results_handler.format_batch_summary(raw_results)
    summary["session_id"] = session_id
    summary["warning"] = warning
    return summary


@app.get("/api/history")
def get_history(
    search: str = Query(default=None),
    category: str = Query(default=None),
):
    return db.get_history(search=search, category=category)


# readiness, not liveness: a 200 means a request would actually be served
@app.get("/api/health")
def health():
    models = pipeline.model_status()
    database_up = db.ping()
    ready = all(models.values()) and database_up

    # load_errors stay in the log; they carry local paths
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "degraded",
            "device": str(pipeline.device),
            "models": models,
            "database": "up" if database_up else "down",
        },
    )
