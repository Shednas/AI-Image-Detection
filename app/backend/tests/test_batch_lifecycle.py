# Batch rows record how the run ended, and the endpoint survives a failed write. Salvaged
# from the Phase 2.4 verification run, logged as T-007. A batch row is written before
# processing starts, so without a closing update a finished batch is indistinguishable in
# History from one that died halfway.

import pytest

from conftest import FakeDb, make_zip, png_bytes
from database.database import BatchStatus


def post_batch(client, entries=None, model="cnn"):
    entries = entries or [("a.png", png_bytes()), ("b.png", png_bytes())]
    return client.post(
        "/api/batch",
        files={"file": ("batch.zip", make_zip(entries), "application/zip")},
        data={"model_name": model},
    )


# A row left reading processing after the response has gone out means the endpoint died
# between the insert and the update.
def test_completed_batch_is_marked_with_its_counts(client, app_module):
    r = post_batch(client)
    assert r.status_code == 200
    assert app_module.db.status_updates == [(BatchStatus.completed, 2, 0)]


# processed plus skipped should equal what the caller sent, or the History row
# misrepresents the batch.
def test_skipped_images_are_counted_separately(client, app_module):
    entries = [
        ("good.png", png_bytes()),
        ("also_good.png", png_bytes()),
        ("broken.png", b"not an image at all"),
    ]
    r = post_batch(client, entries)
    assert r.status_code == 200

    status, processed, skipped = app_module.db.status_updates[0]
    assert (status, processed, skipped) == (BatchStatus.completed, 2, 1)
    assert processed + skipped == 3


# process_batch traps per-file errors, so an exception escaping it means the whole run
# collapsed and no image got a verdict.
def test_collapsed_run_is_marked_failed(client, app_module, monkeypatch):
    def explode(files, model_name, pipeline):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(app_module.batch_processor, "process_batch", explode)

    r = post_batch(client)
    assert r.status_code == 500
    status, processed, skipped = app_module.db.status_updates[0]
    assert status is BatchStatus.failed
    assert processed == 0


# The verdicts are what the caller asked for and the row is bookkeeping, so a database
# failure is reported alongside the results, not instead of them.
def test_results_survive_a_failed_batch_insert(client, app_module):
    app_module.db = FakeDb(fail_save_batch=True)

    r = post_batch(client)
    assert r.status_code == 200
    assert r.json()["warning"] is not None
    assert r.json()["valid"] == 2


# The per-image rows carry batch_id as a foreign key, so if the batch row failed there
# is nothing to update and nothing that could be written.
def test_no_status_update_for_a_row_that_was_never_inserted(client, app_module):
    app_module.db = FakeDb(fail_save_batch=True)

    post_batch(client)
    assert app_module.db.status_updates == []


# One bad row should cost that row, not the caller's whole result.
def test_per_image_write_failure_is_reported_but_not_fatal(client, app_module):
    app_module.db = FakeDb(fail_requests=True)

    r = post_batch(client)
    assert r.status_code == 200
    assert "could not be saved" in r.json()["warning"]


# analyze writes P(AI) into the same two columns, so writing P(real) here would put both
# meanings in one column. This is the inversion that has already caused one live bug.
def test_batch_stores_p_ai_not_p_real(client, app_module):
    post_batch(client)

    assert app_module.db.requests, "nothing was written, the assertion below is vacuous"
    for row in app_module.db.requests:
        assert row["consensus_score"] == pytest.approx(0.75)
    for row in app_module.db.outputs:
        assert row["predicted_probability"] == pytest.approx(0.75)
