import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from sqlalchemy import create_engine, text, Column, String, Integer, Numeric, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# anchored to backend/.env instead of letting find_dotenv search. Bare
# load_dotenv() walks up to the filesystem root, so on a clone with no .env yet
# it would silently adopt an unrelated one from a parent directory. It also
# falls back to the working directory under a frozen build, which is how the
# Phase 6.3 executable would run.
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

Base = declarative_base()


# the batch row is written before processing starts, so it needs a state that
# says as much. A row still reading processing once a request is over means the
# endpoint died between the insert and the update.
class BatchStatus(str, Enum):
    processing = "processing"
    completed = "completed"
    failed = "failed"


class SessionRecord(Base):
    __tablename__ = "sessions"
    session_id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BatchRecord(Base):
    __tablename__ = "batches"
    batch_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("sessions.session_id"), nullable=False)
    total_files = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default=BatchStatus.processing.value)
    processed_files = Column(Integer, nullable=False, default=0)
    skipped_files = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class InferenceRequest(Base):
    __tablename__ = "inference_requests"
    request_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("sessions.session_id"), nullable=False)
    batch_id = Column(String(36), ForeignKey("batches.batch_id"), nullable=True)
    file_name = Column(String(255), nullable=False)
    final_verdict = Column(String(20), nullable=False)
    consensus_score = Column(Numeric(6, 4), nullable=False)
    scanned_at = Column(DateTime, default=datetime.utcnow)


class ModelOutput(Base):
    __tablename__ = "model_outputs"
    output_id = Column(String(36), primary_key=True)
    request_id = Column(String(36), ForeignKey("inference_requests.request_id"), nullable=False)
    model_name = Column(String(50), nullable=False)
    predicted_probability = Column(Numeric(6, 4), nullable=False)
    latency_ms = Column(Integer, nullable=False)


class DatabaseManager:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        # create_engine(None) raises an ArgumentError that says nothing about the
        # missing file, which is the first thing a fresh clone hits
        if not self.db_url:
            raise RuntimeError(
                f"DATABASE_URL is not set. Copy backend/.env.example to {ENV_PATH} "
                "and set your PostgreSQL connection string."
            )
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self._migrate()

    # create_all only creates tables that are missing, so a column added to a
    # model afterwards never reaches a database that already holds the table.
    # These statements are written to be idempotent so a fresh clone and the
    # existing viva database converge on the same schema without dropping
    # anything and without pulling in a migration tool.
    def _migrate(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("ALTER TABLE batches ADD COLUMN IF NOT EXISTS status VARCHAR(20)"))
            conn.execute(text("ALTER TABLE batches ADD COLUMN IF NOT EXISTS processed_files INTEGER"))
            conn.execute(text("ALTER TABLE batches ADD COLUMN IF NOT EXISTS skipped_files INTEGER"))

            # rows predating the column are finished test batches, so they are
            # backfilled as completed. The column default is processing, which
            # is true of a row being inserted but would be a lie about history,
            # which is why the old rows are backfilled rather than left to take
            # the default.
            conn.execute(
                text("UPDATE batches SET status = :done WHERE status IS NULL"),
                {"done": BatchStatus.completed.value},
            )

            # the counts are recovered rather than guessed: inference_requests
            # holds one row per image that came through, so anything short of
            # total_files did not. A batch whose rows failed to save would be
            # understated here, but zero would misreport every historical batch
            # rather than just that one.
            conn.execute(text("""
                UPDATE batches b
                SET processed_files = c.n,
                    skipped_files = GREATEST(b.total_files - c.n, 0)
                FROM (
                    SELECT batch_id, COUNT(*) AS n
                    FROM inference_requests
                    WHERE batch_id IS NOT NULL
                    GROUP BY batch_id
                ) c
                WHERE b.batch_id = c.batch_id AND b.processed_files IS NULL
            """))
            conn.execute(text(
                "UPDATE batches SET processed_files = 0, skipped_files = total_files "
                "WHERE processed_files IS NULL"
            ))

            # only safe once every row is filled, which is what the backfills above
            # guarantee. The default is interpolated because PostgreSQL will not
            # accept a bind parameter in DDL; the value is a constant from the
            # enum above, never anything a caller supplied.
            conn.execute(text(
                f"ALTER TABLE batches ALTER COLUMN status SET DEFAULT '{BatchStatus.processing.value}'"
            ))
            conn.execute(text("ALTER TABLE batches ALTER COLUMN status SET NOT NULL"))
            conn.execute(text("ALTER TABLE batches ALTER COLUMN processed_files SET DEFAULT 0"))
            conn.execute(text("ALTER TABLE batches ALTER COLUMN processed_files SET NOT NULL"))
            conn.execute(text("ALTER TABLE batches ALTER COLUMN skipped_files SET DEFAULT 0"))
            conn.execute(text("ALTER TABLE batches ALTER COLUMN skipped_files SET NOT NULL"))

    def create_session_record(self, session_id: str) -> None:
        db = self.SessionLocal()
        try:
            db.add(SessionRecord(session_id=session_id))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()    

    # the sessions row has to exist before anything references it, so callers
    # check here rather than trusting a client-supplied id
    def session_exists(self, session_id: str) -> bool:
        db = self.SessionLocal()
        try:
            row = (
                db.query(SessionRecord.session_id)
                .filter(SessionRecord.session_id == session_id)
                .first()
            )
            return row is not None
        finally:
            db.close()

    def save_batch(self, batch_id: str, session_id: str, total_files: int) -> None:
        db = self.SessionLocal()
        try:
            db.add(BatchRecord(
                batch_id=batch_id,
                session_id=session_id,
                total_files=total_files,
                status=BatchStatus.processing.value,
            ))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def update_batch_status(
        self, batch_id: str, status: BatchStatus, processed: int, skipped: int
    ) -> None:
        db = self.SessionLocal()
        try:
            row = db.query(BatchRecord).filter(BatchRecord.batch_id == batch_id).first()
            # nothing to update if the insert failed earlier, and that failure has
            # already been logged and reported by the caller
            if row is None:
                return
            row.status = status.value
            row.processed_files = processed
            row.skipped_files = skipped
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def save_inference_request(self, data: dict) -> None:
        db = self.SessionLocal()
        try:
            db.add(InferenceRequest(**data))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def save_model_output(self, data: dict) -> None:
        db = self.SessionLocal()
        try:
            db.add(ModelOutput(**data))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_history(self, search: str = None, category: str = None) -> list:
        db = self.SessionLocal()
        try:
            query = (
                db.query(InferenceRequest, ModelOutput)
                .join(ModelOutput, ModelOutput.request_id == InferenceRequest.request_id)
            )
            if search:
                query = query.filter(InferenceRequest.file_name.ilike(f"%{search}%"))
            if category and category.lower() not in ("", "all"):
                query = query.filter(InferenceRequest.final_verdict == category)
            records = query.order_by(InferenceRequest.scanned_at.desc()).all()
            return [
                {
                    "record_id": req.request_id[:8],
                    "timestamp": req.scanned_at.isoformat() if req.scanned_at else "",
                    "file_name": req.file_name,
                    "model_name": out.model_name,
                    "verdict": req.final_verdict,
                    # consensus_score holds P(AI); named p_ai here so the History
                    # table cannot drift back into displaying the wrong direction
                    "p_ai": float(req.consensus_score),
                }
                for req, out in records
            ]
        finally:
            db.close()
