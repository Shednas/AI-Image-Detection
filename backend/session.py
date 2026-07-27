import uuid
from database.database import DatabaseManager


class SessionTracker:
    def __init__(self, db: DatabaseManager):
        self.db = db

    # no cookie/auth, just used for history grouping
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.db.create_session_record(session_id)
        return session_id

    # a well-formed id is not enough: an id that parses but has no sessions row
    # would violate the foreign key on inference_requests, so fall back to a new
    # session rather than trusting whatever the client sent
    def resolve_session(self, session_id: str | None) -> str:
        if session_id and self.validate_session(session_id) and self.db.session_exists(session_id):
            return session_id
        return self.create_session()

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
