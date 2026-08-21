import argparse
import logging
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# Validates and resizes all images in a processed stage directory
class DataProcessor:
    def __init__(self, target_size: int = 256, dry_run: bool = False):
        self.target_size = target_size
        self.dry_run = dry_run
        self.data_root = PROJECT_ROOT / "data" / "processed"

        self.processed = 0
        self.valid = 0
        self.corrupted = 0

    # Resize image to target_size in-place; delete it if corrupted or unreadable
    def process_image(self, img_path: Path) -> str:
        try:
            with Image.open(img_path) as img:
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_resized = img.resize(
                    (self.target_size, self.target_size),
                    Image.Resampling.LANCZOS,
                )

            if not self.dry_run:
                img_resized.save(img_path, format="JPEG", quality=95)

            return "ok"

        except Exception as e:
            logger.warning(f"Invalid/resize failed — {img_path.name}: {str(e)[:60]}")
            if not self.dry_run:
                img_path.unlink(missing_ok=True)
            return "corrupted"

    # Process all images in a single label folder and return counts
    def process_folder(self, folder_path: Path) -> dict:
        if not folder_path.exists():
            return {"total": 0, "valid": 0, "corrupted": 0}

        images = [
            p for p in folder_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        stats = {"total": len(images), "valid": 0, "corrupted": 0}

        if not images:
            return stats

        prefix = "[DRY RUN] " if self.dry_run else ""
        logger.info(f"{prefix}{folder_path.relative_to(PROJECT_ROOT)}: {len(images)} images")

        for img_path in images:
            self.processed += 1
            result = self.process_image(img_path)
            if result == "ok":
                self.valid += 1
                stats["valid"] += 1
            else:
                self.corrupted += 1
                stats["corrupted"] += 1

        return stats

    # Process real and ai_generated folders for all three splits of a stage
    def process_stage(self, stage: int) -> bool:
        stage_dir = self.data_root / f"stage_{stage}"

        if not stage_dir.exists():
            logger.error(
                f"Stage directory not found: {stage_dir}. "
                f"Run split_dataset.py first."
            )
            return False

        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Stage {stage}: {stage_dir.name}")

        all_stats: dict = {}
        for split in ("train", "validation", "test"):
            all_stats[split] = {
                "real": self.process_folder(stage_dir / split / "real"),
                "ai_generated": self.process_folder(stage_dir / split / "ai_generated"),
            }

        logger.info(f"\nStage {stage} summary:")
        for split, labels in all_stats.items():
            real = labels["real"]
            fake = labels["ai_generated"]
            total = real["valid"] + fake["valid"]
            logger.info(
                f"  {split.upper():12} — "
                f"Real: {real['valid']:4d}/{real['total']:4d}  |  "
                f"AI: {fake['valid']:4d}/{fake['total']:4d}  |  "
                f"Total: {total}"
            )

        return True

    def process_all_stages(self) -> None:
        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Processing all stages")
        for stage in (1, 2, 3):
            self.process_stage(stage)

    def print_summary(self) -> None:
        prefix = "[DRY RUN] " if self.dry_run else ""
        logger.info(f"{prefix}Processing complete")
        logger.info(f"  Total processed: {self.processed}")
        logger.info(f"  Valid:           {self.valid}")
        logger.info(f"  Corrupted:       {self.corrupted}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process dataset: validate and resize images"
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3], default=None,
        help="Process a specific stage (default: all stages)",
    )
    parser.add_argument(
        "--target-size", type=int, default=256,
        help="Target image size in pixels — images are resized to a square (default: 256)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and report without writing or deleting any files.",
    )
    args = parser.parse_args()

    processor = DataProcessor(target_size=args.target_size, dry_run=args.dry_run)

    if args.stage is not None:
        processor.process_stage(args.stage)
    else:
        processor.process_all_stages()
    processor.print_summary()


if __name__ == "__main__":
    main()
