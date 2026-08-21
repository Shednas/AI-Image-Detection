import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from sqlalchemy import create_engine, text, Column, String, Integer, Numeric, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# anchored rather than letting find_dotenv search: it walks to the filesystem
# root, so a clone with no .env yet would silently adopt an unrelated one
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

logger = logging.getLogger("ai_detection.database")

Base = declarative_base()


# a row still reading processing after a request is over means the endpoint died
# between the insert and the update
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
        # create_engine(None) raises an ArgumentError that never mentions .env
        if not self.db_url:
            raise RuntimeError(
                f"DATABASE_URL is not set. Copy backend/.env.example to {ENV_PATH} "
                "and set your PostgreSQL connection string."
            )
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.schema_ready = False
        # create_engine does not connect, so this is the only startup step that
        # needs the server. It must not stop the backend from starting.
        try:
            self._ensure_schema()
        except Exception:
            logger.exception("Could not prepare the database schema, starting degraded")

    # deferred to the first successful connection, so a backend that started
    # while the database was down recovers without a restart
    def _ensure_schema(self) -> None:
        if self.schema_ready:
            return
        Base.metadata.create_all(bind=self.engine)
        self._migrate()
        self.schema_ready = True
        logger.info("Database schema is ready")

    # so a session is never handed out against a schema that was never created
    def _session(self):
        self._ensure_schema()
        return self.SessionLocal()

    # create_all only creates missing tables, so a column added later never
    # reaches an existing database. Idempotent, and drops nothing.
    def _migrate(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("ALTER TABLE batches ADD COLUMN IF NOT EXISTS status VARCHAR(20)"))
            conn.execute(text("ALTER TABLE batches ADD COLUMN IF NOT EXISTS processed_files INTEGER"))
            conn.execute(text("ALTER TABLE batches ADD COLUMN IF NOT EXISTS skipped_files INTEGER"))

            # rows predating the column are finished batches. The default is
            # processing, true of a new row but a lie about history, so they are
            # backfilled rather than left to take it.
            conn.execute(
                text("UPDATE batches SET status = :done WHERE status IS NULL"),
                {"done": BatchStatus.completed.value},
            )

            # recovered rather than guessed: inference_requests holds one row per
            # image that came through. A batch whose rows failed to save is
            # understated, but zero would misreport every batch instead of one.
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

            # only safe once the backfills above have filled every row.
            # Interpolated because PostgreSQL takes no bind parameter in DDL; the
            # value is the enum constant, never caller input.
            conn.execute(text(
                f"ALTER TABLE batches ALTER COLUMN status SET DEFAULT '{BatchStatus.processing.value}'"
            ))
            conn.execute(text("ALTER TABLE batches ALTER COLUMN status SET NOT NULL"))
            conn.execute(text("ALTER TABLE batches ALTER COLUMN processed_files SET DEFAULT 0"))
            conn.execute(text("ALTER TABLE batches ALTER COLUMN processed_files SET NOT NULL"))
            conn.execute(text("ALTER TABLE batches ALTER COLUMN skipped_files SET DEFAULT 0"))
            conn.execute(text("ALTER TABLE batches ALTER COLUMN skipped_files SET NOT NULL"))

    # a round trip, not a pool check: a pooled connection still looks healthy
    # after the server goes away
    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            logger.exception("Database ping failed")
            return False

        # first chance to build a schema that could not be created at startup.
        # Still down until it succeeds: no tables means no request can be served.
        try:
            self._ensure_schema()
        except Exception:
            logger.exception("Database is reachable but the schema is not ready")
            return False
        return True

    def create_session_record(self, session_id: str) -> None:
        db = self._session()
        try:
            db.add(SessionRecord(session_id=session_id))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()    

    # the sessions row must exist before anything references it, and the id is
    # client-supplied
    def session_exists(self, session_id: str) -> bool:
        db = self._session()
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
        db = self._session()
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
        db = self._session()
        try:
            row = db.query(BatchRecord).filter(BatchRecord.batch_id == batch_id).first()
            # the insert failed earlier, and the caller has already reported it
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
        db = self._session()
        try:
            db.add(InferenceRequest(**data))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def save_model_output(self, data: dict) -> None:
        db = self._session()
        try:
            db.add(ModelOutput(**data))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_history(self, search: str = None, category: str = None) -> list:
        db = self._session()
        try:
            query = (
                db.query(InferenceRequest, ModelOutput)
                .join(ModelOutput, ModelOutput.request_id == InferenceRequest.request_id)
            )
            if search:
                # % and _ are ilike wildcards, so a filename containing either would
                # otherwise match far more than the user typed
                escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                query = query.filter(
                    InferenceRequest.file_name.ilike(f"%{escaped}%", escape="\\")
                )
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
