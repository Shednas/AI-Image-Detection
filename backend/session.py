import uuid
from database.database import DatabaseManager


class SessionTracker:
    def __init__(self, db: DatabaseManager):
        self.db = db

    # new session per request; no cookie/auth, just used for history grouping
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.db.create_session_record(session_id)
        return session_id

    # UUID parse check is the only validation needed for a stateless session
    def validate_session(self, session_id: str) -> bool:
        try:
            uuid.UUID(session_id)
            return True
        except ValueError:
            return False

    # stub for future session enrichment
    def get_session(self, session_id: str) -> dict:
        return {"session_id": session_id}
