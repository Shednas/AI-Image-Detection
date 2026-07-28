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

# content_type is supplied by the client, so this only screens obvious
# mismatches. The decode in validate_image is what actually proves the bytes
# are an image.
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


# last line of defence: anything not already turned into an HTTPException gets
# logged in full here and reduced to a generic message, so no stack trace or
# internal path reaches the browser
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on the server. Please try again."},
    )


# checked before any work is done, so a model whose weights failed to load is
# reported as a service state rather than as a failed analysis. Without this the
# batch endpoint would return an error on every file and look like a completed
# batch, the same shape of problem the model_name enum fixed in 2.1.
def require_model(model_key: str) -> None:
    if not pipeline.is_loaded(model_key):
        logger.error("Rejected a request for %s, which is not loaded", model_key)
        raise HTTPException(
            status_code=503,
            detail=f"The {model_key.upper()} model is unavailable. Try another model.",
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

    # the verdict is what the caller asked for and the row is bookkeeping, so a
    # database failure is reported alongside the result rather than replacing it
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

    # echoed back so the client can send it again and stay in one session
    formatted["session_id"] = session_id
    formatted["warning"] = warning
    return formatted


# the batch row is bookkeeping in the same way the per-image rows are, so a
# failure to update it is logged rather than raised over a result the caller
# already has. Nothing user-facing reads the status yet, so there is no warning
# to raise either.
def mark_batch(batch_id: str, status: BatchStatus, processed: int, skipped: int) -> None:
    try:
        db.update_batch_status(batch_id, status, processed, skipped)
    except Exception:
        logger.exception("Could not mark batch %s as %s", batch_id, status.value)


# skip empty/corrupt images but process the rest
@app.post("/api/batch")
async def analyze_batch(
    file: UploadFile = File(...),
    model_name: ModelName = Form(...),
    session_id: str | None = Form(default=None),
):
    zip_bytes = await file.read()
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
        # the per-image rows carry this batch_id as a foreign key, so without the
        # batch row they cannot be written either
        logger.exception("Could not save batch %s", batch_id)
        batch_saved = False
        warning = "This batch could not be saved, so it will not appear in History."

    try:
        raw_results = batch_processor.process_batch(files, model_name.value, pipeline)
    except Exception:
        logger.exception("Batch %s failed during processing", batch_id)
        # process_batch traps per-file errors, so reaching here means the run
        # collapsed as a whole and no image got a verdict
        if batch_saved:
            mark_batch(batch_id, BatchStatus.failed, processed=0, skipped=len(files))
        raise HTTPException(status_code=500, detail="Could not process this batch.")

    # store p_ai, not probability: the analyze endpoint writes P(AI) into these
    # same two columns, so writing P(real) here would put both meanings in one
    # column with nothing to tell them apart
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
            # one bad row should not cost the caller the whole batch result
            logger.exception("Could not save request %s for %r", request_id, r["file_name"])
            unsaved += 1

    if unsaved:
        warning = f"{unsaved} of these results could not be saved to History."

    # skipped counts images the batch could not use, which is the same set the
    # summary reports as invalid rows
    processed = sum(1 for r in raw_results if "error" not in r)
    if batch_saved:
        mark_batch(batch_id, BatchStatus.completed, processed, len(raw_results) - processed)

    summary = results_handler.format_batch_summary(raw_results)
    summary["session_id"] = session_id
    summary["warning"] = warning
    return summary


# thin pass-through; filtering is done in the DB layer
@app.get("/api/history")
async def get_history(
    search: str = Query(default=None),
    category: str = Query(default=None),
):
    return db.get_history(search=search, category=category)


# readiness rather than liveness: a 200 here means a request would actually be
# served, so every dependency is checked rather than just the process being up
@app.get("/api/health")
def health():
    models = pipeline.model_status()
    database_up = db.ping()
    ready = all(models.values()) and database_up

    # the load errors themselves stay in the log. They carry local paths, and the
    # count is enough to tell a caller to go and look
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "degraded",
            "device": str(pipeline.device),
            "models": models,
            "database": "up" if database_up else "down",
        },
    )
