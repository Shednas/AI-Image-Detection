import io
import logging
import zipfile
from pathlib import Path

from PIL import Image

logger = logging.getLogger("ai_detection.batch")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MAX_FILE_SIZE_MB = MAX_FILE_SIZE_BYTES // (1024 * 1024)
MAX_FILES_PER_ZIP = 100
MAX_TOTAL_UNCOMPRESSED_BYTES = 200 * 1024 * 1024


# carries the status the endpoint should return, so the reason for the rejection
# survives the trip out of this module
class ZipRejected(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class BatchProcessor:
    # skip oversized or non-image entries without failing the whole batch, but
    # refuse the archive outright when it breaches a whole-zip limit
    def extract_zip(self, zip_bytes: bytes) -> list[tuple[str, bytes]]:
        try:
            archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
        except zipfile.BadZipFile as e:
            raise ZipRejected("This file is not a readable zip archive.") from e

        files = []
        oversized = 0
        corrupt = 0
        total_bytes = 0

        with archive as zf:
            entries = [
                info for info in zf.infolist()
                if not info.is_dir() and Path(info.filename).suffix.lower() in ALLOWED_EXTENSIONS
            ]
            if len(entries) > MAX_FILES_PER_ZIP:
                raise ZipRejected(
                    f"This zip holds {len(entries)} images, more than the "
                    f"{MAX_FILES_PER_ZIP} allowed in one batch."
                )

            for info in entries:
                # checking the declared size first refuses a bomb without
                # decompressing it at all. zipfile also stops decompressing at
                # the declared size and fails the CRC if the entry understates
                # it, so the bounded read below is defence in depth rather than
                # the only guard.
                if info.file_size > MAX_FILE_SIZE_BYTES:
                    oversized += 1
                    continue
                try:
                    with zf.open(info) as handle:
                        raw_bytes = handle.read(MAX_FILE_SIZE_BYTES + 1)
                except (zipfile.BadZipFile, OSError, EOFError) as e:
                    # one damaged entry should not cost the caller the archive.
                    # a tampered header lands here as a CRC failure
                    logger.warning("Skipping unreadable zip entry %r: %s", info.filename, e)
                    corrupt += 1
                    continue
                if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
                    oversized += 1
                    continue

                total_bytes += len(raw_bytes)
                if total_bytes > MAX_TOTAL_UNCOMPRESSED_BYTES:
                    # derived here rather than held as a second constant, which
                    # could disagree with the byte limit it describes
                    limit_mb = MAX_TOTAL_UNCOMPRESSED_BYTES // (1024 * 1024)
                    raise ZipRejected(
                        f"This zip expands to more than {limit_mb}MB.",
                        status_code=413,
                    )
                # Path().name drops any directory component, so nested archives
                # flatten and a crafted path cannot escape anywhere
                files.append((Path(info.filename).name, raw_bytes))

        if not files:
            if oversized and corrupt:
                raise ZipRejected(
                    f"No image in this zip could be used: {oversized} over the "
                    f"{MAX_FILE_SIZE_MB}MB per-file limit, {corrupt} damaged."
                )
            if oversized:
                raise ZipRejected(
                    f"Every image in this zip is over the {MAX_FILE_SIZE_MB}MB per-file limit."
                )
            if corrupt:
                raise ZipRejected("Every image in this zip is damaged or unreadable.")
            raise ZipRejected("This zip contains no JPEG, PNG or WebP images.")

        return files

    # double-checks extension and PIL decode since zip contents can be mislabelled.
    # load() rather than verify(): verify() only reads the header, so a truncated
    # entry would pass here and then abort the file later in process_batch.
    def validate_file(self, file_bytes: bytes, filename: str) -> bool:
        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            return False
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.load()
            return True
        except Exception:
            return False

    # per-file errors are captured so one bad image doesn't abort the batch
    def process_batch(self, files: list[tuple[str, bytes]], model_name: str, pipeline) -> list[dict]:
        results = []
        for filename, file_bytes in files:
            if not self.validate_file(file_bytes, filename):
                results.append({"file_name": filename, "error": "Invalid or unsupported file"})
                continue
            try:
                tensor = pipeline.preprocess(file_bytes)
                pred = pipeline.predict(tensor, model_name)
                pred["file_name"] = filename
                results.append(pred)
            except Exception as e:
                results.append({"file_name": filename, "error": str(e)})
        return results
