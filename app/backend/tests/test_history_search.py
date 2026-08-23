# Search terms are escaped before they reach ilike, so % and _ match literally. Without
# that, a filename containing either behaved as a wildcard and the search returned far
# more than the user typed. Found during manual testing rather than by a failing
# assertion, so it traces to input validation rather than to code.
#
# These run against a real database on purpose: ilike escaping is PostgreSQL behaviour,
# so a stub would assert nothing about the thing that actually broke.

import uuid

import pytest
from sqlalchemy import text

from database.database import DatabaseManager


# Each name is paired with a decoy that differs from it only where a wildcard would let
# the search slide: % matching any run of characters, _ matching any single one. If the
# escaping is removed, the decoy is returned too and the count assertions fail.
@pytest.fixture
def seeded_db(postgres_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    db = DatabaseManager()

    # alphanumeric, so the tag itself contributes no wildcard characters
    tag = uuid.uuid4().hex[:12]
    session_id = str(uuid.uuid4())
    db.create_session_record(session_id)

    names = {
        "percent_literal": f"{tag}-100%.png",
        "percent_decoy": f"{tag}-100zz.png",
        "underscore_literal": f"{tag}-a_b.png",
        "underscore_decoy": f"{tag}-aZb.png",
    }

    request_ids = []
    for file_name in names.values():
        request_id = str(uuid.uuid4())
        request_ids.append(request_id)
        db.save_inference_request({
            "request_id": request_id,
            "session_id": session_id,
            "file_name": file_name,
            "final_verdict": "AI_GENERATED",
            "consensus_score": 0.75,
        })
        # get_history inner joins model_outputs, so a request with no output row
        # would never appear and every assertion below would pass vacuously
        db.save_model_output({
            "output_id": str(uuid.uuid4()),
            "request_id": request_id,
            "model_name": "cnn",
            "predicted_probability": 0.75,
            "latency_ms": 5,
        })

    yield db, tag, names

    with db.engine.begin() as conn:
        conn.execute(
            text("DELETE FROM model_outputs WHERE request_id = ANY(:ids)"),
            {"ids": request_ids},
        )
        conn.execute(
            text("DELETE FROM inference_requests WHERE request_id = ANY(:ids)"),
            {"ids": request_ids},
        )
        conn.execute(
            text("DELETE FROM sessions WHERE session_id = :sid"),
            {"sid": session_id},
        )


# Unescaped, ilike reads the trailing % as "anything follows" and picks up the decoy.
def test_percent_in_a_search_term_matches_literally(seeded_db):
    db, tag, names = seeded_db

    found = [row["file_name"] for row in db.get_history(search=f"{tag}-100%")]

    assert found == [names["percent_literal"]]
    assert names["percent_decoy"] not in found


# Unescaped, _ matches any single character, so a_b would also match aZb.
def test_underscore_in_a_search_term_matches_literally(seeded_db):
    db, tag, names = seeded_db

    found = [row["file_name"] for row in db.get_history(search=f"{tag}-a_b")]

    assert found == [names["underscore_literal"]]
    assert names["underscore_decoy"] not in found


# Guards the fixture itself: if seeding silently failed, the two tests above would pass
# on empty results and prove nothing.
def test_all_seeded_rows_are_visible_without_a_search_term(seeded_db):
    db, tag, names = seeded_db

    found = [row["file_name"] for row in db.get_history() if row["file_name"].startswith(tag)]

    assert sorted(found) == sorted(names.values())
