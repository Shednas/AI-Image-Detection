import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from batch_handler import BatchProcessor
from database.database import DatabaseManager
from pipeline import InferencePipeline
from results import ResultsHandler
from session import SessionTracker

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
    model_name: str = Form(...),
):
    image_bytes = await file.read()

    if not pipeline.validate_image(image_bytes):
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

    session_id = session_tracker.create_session()
    request_id = str(uuid.uuid4())
    image_tensor = pipeline.preprocess(image_bytes)
    raw = pipeline.predict(image_tensor, model_name)

    model_obj = getattr(pipeline, model_name)
    formatted = results_handler.format_single(raw, image_bytes, model_name, image_tensor, model_obj)

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

    return formatted


# skip empty/corrupt images but process the rest
@app.post("/api/batch")
async def analyze_batch(
    file: UploadFile = File(...),
    model_name: str = Form(...),
):
    zip_bytes = await file.read()
    session_id = session_tracker.create_session()
    batch_id = str(uuid.uuid4())

    files = batch_processor.extract_zip(zip_bytes)
    if not files:
        raise HTTPException(status_code=400, detail="No valid images found in the zip file.")

    db.save_batch(batch_id, session_id, total_files=len(files))
    raw_results = batch_processor.process_batch(files, model_name, pipeline)

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
            "consensus_score": r["probability"],
        })
        db.save_model_output({
            "output_id": str(uuid.uuid4()),
            "request_id": request_id,
            "model_name": r["model_name"],
            "predicted_probability": r["probability"],
            "latency_ms": r["latency_ms"],
        })

    return results_handler.format_batch_summary(raw_results)


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
