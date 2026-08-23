# Zip extraction, including the hostile cases. Salvaged from the Phase 2.3 verification
# run, logged as T-005 and T-006. A batch upload is the only place the application opens
# an archive supplied by someone else, so it is the only place a zip bomb or a crafted
# path can reach it.

import struct
import zipfile

import pytest

import batch_handler
from batch_handler import BatchProcessor, ZipRejected
from conftest import make_zip, png_bytes


@pytest.fixture
def processor():
    return BatchProcessor()


# Non-images and directory entries must be dropped silently rather than reaching the
# decoder as failed rows.
def test_valid_zip_returns_only_images(processor):
    data = make_zip([
        ("a.png", png_bytes()),
        ("nested/b.jpg", png_bytes()),
        ("notes.txt", b"hello"),
        ("sub/", b""),
    ])
    assert sorted(name for name, _ in processor.extract_zip(data)) == ["a.png", "b.jpg"]


# A crafted path like ../../evil.png must not escape. Path().name drops the directory,
# so nothing can be written outside the intended location.
def test_nested_paths_are_flattened(processor):
    files = processor.extract_zip(make_zip([("../../evil.png", png_bytes())]))
    assert [name for name, _ in files] == ["evil.png"]


# Bytes that are not a zip at all should be a clean 400, not a stack trace.
def test_unreadable_archive_is_rejected(processor):
    with pytest.raises(ZipRejected) as exc:
        processor.extract_zip(b"this is not a zip")
    assert exc.value.status_code == 400


# An archive of PDFs is a caller error, and the message should say so rather than
# reporting an empty batch.
def test_zip_with_no_images_is_rejected(processor):
    with pytest.raises(ZipRejected, match="JPEG, PNG or WebP"):
        processor.extract_zip(make_zip([("a.txt", b"x"), ("b.pdf", b"y")]))


# Without a cap, a zip of 10,000 images ties up the server for an hour.
def test_file_count_cap_is_enforced(processor, monkeypatch):
    monkeypatch.setattr(batch_handler, "MAX_FILES_PER_ZIP", 3)
    data = make_zip([(f"{i}.png", png_bytes()) for i in range(4)])
    with pytest.raises(ZipRejected, match="more than the 3"):
        processor.extract_zip(data)


# Off-by-one guard: the limit is inclusive.
def test_exactly_at_the_file_cap_is_allowed(processor, monkeypatch):
    monkeypatch.setattr(batch_handler, "MAX_FILES_PER_ZIP", 3)
    data = make_zip([(f"{i}.png", png_bytes()) for i in range(3)])
    assert len(processor.extract_zip(data)) == 3


# The per-file limit alone does not stop a hundred files just under it. Raw blobs, not
# PNGs: a solid-colour PNG compresses to almost nothing and passed for the wrong reason.
def test_total_uncompressed_cap_is_enforced(processor, monkeypatch):
    monkeypatch.setattr(batch_handler, "MAX_TOTAL_UNCOMPRESSED_BYTES", 3 * 1024 * 1024)
    blob = b"\0" * (1024 * 1024)
    with pytest.raises(ZipRejected) as exc:
        processor.extract_zip(make_zip([(f"{i}.png", blob) for i in range(8)]))
    assert exc.value.status_code == 413


# One huge file in an otherwise usable archive should cost that file only.
def test_oversized_entry_is_skipped_not_fatal(processor):
    huge = b"\0" * (batch_handler.MAX_FILE_SIZE_BYTES + 1024)
    files = processor.extract_zip(make_zip([("big.png", huge), ("ok.png", png_bytes())]))
    assert [name for name, _ in files] == ["ok.png"]


# Zero usable images has four distinct causes, and the caller needs to know which one
# applies or they cannot fix it.
def test_zip_of_only_oversized_entries_explains_itself(processor):
    huge = b"\0" * (batch_handler.MAX_FILE_SIZE_BYTES + 1024)
    with pytest.raises(ZipRejected, match="per-file limit"):
        processor.extract_zip(make_zip([("big.png", huge)]))


# The declared size is checked before any decompression, so a header understating the
# real size cannot smuggle bytes past it. zipfile stops at the declared size and fails
# the CRC, so this lands as a damaged entry.
def test_falsified_header_cannot_exceed_the_limit(processor):
    true_size = batch_handler.MAX_FILE_SIZE_BYTES + 2 * 1024 * 1024
    raw = make_zip([("bomb.png", b"\0" * true_size)])
    patched = raw.replace(struct.pack("<I", true_size), struct.pack("<I", 1000))
    assert patched != raw, "header patch did not apply, the test proves nothing"

    try:
        files = processor.extract_zip(patched)
    except ZipRejected:
        return
    for _, payload in files:
        assert len(payload) <= batch_handler.MAX_FILE_SIZE_BYTES


# BadZipFile is raised from the constructor and from the per-entry read. Only the first
# was handled once, so a damaged member escaped extract_zip and became a generic 500.
def test_damaged_entry_does_not_cost_the_good_ones(processor):
    true_size = batch_handler.MAX_FILE_SIZE_BYTES + 2 * 1024 * 1024
    data = make_zip([("good.png", png_bytes()), ("bomb.png", b"\0" * true_size)])
    data = data.replace(struct.pack("<I", true_size), struct.pack("<I", 1000))
    assert [name for name, _ in processor.extract_zip(data)] == ["good.png"]


# A password-protected entry makes zipfile raise RuntimeError, which extract_zip does
# not catch, so it escapes as a generic 500. The stdlib cannot write an encrypted zip,
# so the failure is injected where zipfile would raise it.
@pytest.mark.xfail(
    reason="RuntimeError from an encrypted entry is not in the caught tuple",
    strict=False,
)
def test_encrypted_entry_does_not_escape_as_a_500(processor, monkeypatch):
    def raising_open(self, name, *args, **kwargs):
        raise RuntimeError("File 'a.png' is encrypted, password required for extraction")

    monkeypatch.setattr(zipfile.ZipFile, "open", raising_open)
    with pytest.raises(ZipRejected):
        processor.extract_zip(make_zip([("a.png", png_bytes())]))


# The endpoint maps the reason straight through, so the status has to travel with the
# exception rather than being guessed at the boundary.
def test_rejection_carries_its_own_status_code(processor):
    with pytest.raises(ZipRejected) as exc:
        processor.extract_zip(b"nope")
    assert isinstance(exc.value.status_code, int)


# End to end: a bad archive reaches the client as 400, not 500.
def test_endpoint_maps_rejection_to_the_right_status(client):
    r = client.post(
        "/api/batch",
        files={"file": ("b.zip", b"not a zip", "application/zip")},
        data={"model_name": "cnn"},
    )
    assert r.status_code == 400
