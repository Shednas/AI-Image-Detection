# The health endpoint, and the 503 guard that depends on it. Salvaged from the Phase 2.5
# verification run, logged as T-010. Health is a readiness check, not a liveness check: a
# 200 means a request would actually be served. A 200 while a model is missing is worse
# than no endpoint at all, because it is the answer a deployment check trusts.

import pytest

from conftest import FakeDb, FakePipeline, make_zip, png_bytes

ALL_UP = {"cnn": True, "fft": True, "hybrid": True, "stm": True}
FFT_DOWN = {"cnn": True, "fft": False, "hybrid": True, "stm": True}


# The baseline. If this is not 200 with everything up, nothing else here distinguishes a
# real failure from a broken fixture.
def test_healthy_backend_returns_200(client, app_module):
    app_module.pipeline = FakePipeline(ALL_UP, device="cuda")
    app_module.db = FakeDb()

    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["device"] == "cuda"
    assert body["database"] == "up"
    assert body["models"] == ALL_UP


# Reporting overall health only would hide which model is unavailable, and that is the
# one thing the person reading it needs.
def test_one_missing_model_gives_503(client, app_module):
    app_module.pipeline = FakePipeline(FFT_DOWN)
    app_module.db = FakeDb()

    r = client.get("/api/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"
    assert r.json()["models"]["fft"] is False


# Models loaded but no database means History silently loses everything, so it is not a
# healthy state even though inference would work.
def test_unreachable_database_gives_503(client, app_module):
    app_module.pipeline = FakePipeline(ALL_UP)
    db = FakeDb()
    db.up = False
    app_module.db = db

    r = client.get("/api/health")
    assert r.status_code == 503
    assert r.json()["database"] == "down"
    assert all(r.json()["models"].values())


# Load failures carry filesystem paths, so they stay in the log. The endpoint is
# reachable without authentication.
def test_response_leaks_no_local_paths(client, app_module):
    app_module.pipeline = FakePipeline(FFT_DOWN)
    app_module.db = FakeDb()

    text = client.get("/api/health").text
    assert "load_errors" not in text
    assert "C:\\" not in text
    assert "/Users/" not in text


# 503, not 500: the model being unavailable is a service state, not a problem with the
# image the caller sent.
def test_analyze_rejects_an_unloaded_model_with_503(client, app_module):
    app_module.pipeline = FakePipeline(FFT_DOWN)

    r = client.post(
        "/api/analyze",
        files={"file": ("a.png", png_bytes(), "image/png")},
        data={"model_name": "fft"},
    )
    assert r.status_code == 503
    assert "FFT" in r.json()["detail"]


# Without this guard the batch runs, fails on every file, and returns 200 with a full
# set of errors, which reads as a completed batch rather than a rejected request.
def test_batch_rejects_an_unloaded_model_before_opening_the_zip(client, app_module):
    app_module.pipeline = FakePipeline(FFT_DOWN)

    r = client.post(
        "/api/batch",
        files={"file": ("b.zip", make_zip([("a.png", png_bytes())]), "application/zip")},
        data={"model_name": "fft"},
    )
    assert r.status_code == 503


# Degraded is not the same as broken. One missing model must not take the other three
# offline, which is the whole reason load_models is per model.
@pytest.mark.parametrize("model", ["cnn", "hybrid", "stm"])
def test_other_models_still_work_while_one_is_down(client, app_module, model):
    app_module.pipeline = FakePipeline(FFT_DOWN)

    r = client.post(
        "/api/analyze",
        files={"file": ("a.png", png_bytes(), "image/png")},
        data={"model_name": model},
    )
    assert r.status_code == 200
