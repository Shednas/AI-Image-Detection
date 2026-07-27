import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Numeric, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class SessionRecord(Base):
    __tablename__ = "sessions"
    session_id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BatchRecord(Base):
    __tablename__ = "batches"
    batch_id    = Column(String(36), primary_key=True)
    session_id  = Column(String(36), ForeignKey("sessions.session_id"), nullable=False)
    total_files = Column(Integer, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)


class InferenceRequest(Base):
    __tablename__ = "inference_requests"
    request_id      = Column(String(36), primary_key=True)
    session_id      = Column(String(36), ForeignKey("sessions.session_id"), nullable=False)
    batch_id        = Column(String(36), ForeignKey("batches.batch_id"), nullable=True)
    file_name       = Column(String(255), nullable=False)
    final_verdict   = Column(String(20), nullable=False)
    consensus_score = Column(Numeric(6, 4), nullable=False)
    scanned_at      = Column(DateTime, default=datetime.utcnow)


class ModelOutput(Base):
    __tablename__ = "model_outputs"
    output_id             = Column(String(36), primary_key=True)
    request_id            = Column(String(36), ForeignKey("inference_requests.request_id"), nullable=False)
    model_name            = Column(String(50), nullable=False)
    predicted_probability = Column(Numeric(6, 4), nullable=False)
    latency_ms            = Column(Integer, nullable=False)


class DatabaseManager:
    def __init__(self):
        self.db_url      = os.getenv("DATABASE_URL")
        self.engine      = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

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

    def save_batch(self, batch_id: str, session_id: str, total_files: int) -> None:
        db = self.SessionLocal()
        try:
            db.add(BatchRecord(batch_id=batch_id, session_id=session_id, total_files=total_files))
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
                    "record_id":  req.request_id[:8],
                    "timestamp":  req.scanned_at.isoformat() if req.scanned_at else "",
                    "file_name":  req.file_name,
                    "model_name": out.model_name,
                    "verdict":    req.final_verdict,
                    # consensus_score holds P(AI); named p_ai here so the History
                    # table cannot drift back into displaying the wrong direction
                    "p_ai":       float(req.consensus_score),
                }
                for req, out in records
            ]
        finally:
            db.close()
