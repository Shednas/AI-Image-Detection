"""Every way a bad upload can be rejected, and the status code it should carry.

Salvaged from the Phase 2.1 verification run, logged as T-001 and T-002. The
point of asserting on the exact status is that the frontend and the marker both
read it: a rejected upload that returns 500 says the server broke, when what
actually happened is the user picked the wrong file.
"""

import pytest

from conftest import png_bytes


def post(client, filename="a.png", data=None, content_type="image/png", model="cnn"):
    return client.post(
        "/api/analyze",
        files={"file": (filename, png_bytes() if data is None else data, content_type)},
        data={"model_name": model},
    )


def test_valid_upload_is_accepted(client):
    """If this fails, nothing below is meaningful: the rejections could be
    passing for the wrong reason."""
    r = post(client)
    assert r.status_code == 200
    assert r.json()["verdict"] == "AI_GENERATED"


def test_empty_file_is_rejected(client):
    """A zero-byte upload reaching PyTorch surfaces as a 500 rather than telling
    the user their file is empty."""
    r = post(client, data=b"")
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_oversized_file_is_rejected(client):
    """Without the size check a large upload is decoded in full before anything
    complains, so the limit has to be enforced before the decode."""
    from batch_handler import MAX_FILE_SIZE_BYTES

    r = post(client, data=b"\0" * (MAX_FILE_SIZE_BYTES + 1))
    assert r.status_code == 413


def test_unsupported_extension_is_rejected(client):
    """A .txt or .exe should never reach the decoder."""
    r = post(client, filename="notes.txt", content_type="text/plain")
    assert r.status_code == 415


def test_mismatched_content_type_is_rejected(client):
    """content_type is client-supplied, so this only screens obvious mismatches.
    The decode is what actually proves the bytes are an image."""
    r = post(client, filename="a.png", content_type="application/pdf")
    assert r.status_code == 415


def test_undecodable_bytes_are_rejected(client):
    """Correct extension and MIME, but the bytes are not an image. This is the
    case the extension check cannot catch."""
    r = post(client, data=b"this is not a png despite the name")
    assert r.status_code == 400
    assert "invalid" in r.json()["detail"].lower()


@pytest.mark.parametrize("model", ["cnn", "fft", "hybrid", "stm"])
def test_every_model_name_is_accepted(client, model):
    """All four must be reachable through the API, or one is unusable from the UI."""
    assert post(client, model=model).status_code == 200


def test_unknown_model_is_rejected_before_the_handler(client):
    """The ModelName enum turns this into a 422 at the boundary. Without it the
    request reaches predict and comes back as a 500 on analyze, or on batch as a
    per-file error on every file, which looks like a completed batch."""
    assert post(client, model="nonsense").status_code == 422


def test_validation_order_puts_the_caller_error_first(client):
    """A file that is both too large and the wrong type should report the size,
    since that is the check that runs first. Fixing the type would not help."""
    from batch_handler import MAX_FILE_SIZE_BYTES

    r = post(
        client,
        filename="notes.txt",
        data=b"\0" * (MAX_FILE_SIZE_BYTES + 1),
        content_type="text/plain",
    )
    assert r.status_code == 413
