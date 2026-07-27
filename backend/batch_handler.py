import io
import zipfile
from pathlib import Path

from PIL import Image

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


class BatchProcessor:
    # skip oversized or non-image entries without failing the whole batch
    def extract_zip(self, zip_bytes: bytes) -> list[tuple[str, bytes]]:
        files = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
                    continue
                raw_bytes = zf.read(name)
                if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
                    continue
                files.append((Path(name).name, raw_bytes))
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

    # guard against empty-batch division by zero with `or 1`
    def get_batch_summary(self, results: list[dict]) -> dict:
        valid = [r for r in results if "error" not in r]
        ai_count = sum(1 for r in valid if r["verdict"] == "AI_GENERATED")
        real_count = len(valid) - ai_count
        total_valid = len(valid) or 1
        return {
            "total": len(results),
            "valid": len(valid),
            "ai_count": ai_count,
            "real_count": real_count,
            "ai_pct": round(ai_count / total_valid * 100, 1),
            "real_pct": round(real_count / total_valid * 100, 1),
        }
