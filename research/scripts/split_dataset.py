import argparse
import json
import random
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import shutil
from PIL import Image


PROJECT_ROOT = Path(__file__).parent.parent

STAGE_SIZES = {
    1: 500,
    2: 5000,
    3: 10000,
}

TOTAL_IMAGES = sum(STAGE_SIZES.values())

# Proportional source contributions for real and fake images
SOURCE_PERCENTAGES: Dict[str, Dict[str, float]] = {
    "real": {
        "cifake": 0.10,
        "coco": 0.10,
        "imagenet": 0.10,
        "unsplash": 0.10,
        "flickr30k": 0.10,
    },
    "fake": {
        "forensynths": 1 / 6,
        "cifake": 1 / 6,
        "genimage/BigGAN": 1 / 6,
        "genimage/MidJourney": 1 / 6,
        "genimage/stable_diffusion_v_1_4": 1 / 6,
        "genimage/stable_diffusion_v_1_5": 1 / 6,
    },
}

SPLIT_RATIOS = (0.7, 0.15, 0.15)

MAX_IMAGE_PIXELS = 80_000_000


# Derive a deterministic per-tag seed by XOR-ing with a stable hash of the tag string
def _make_seed(base_seed: int, tag: str) -> int:
    stable_hash = int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed ^ stable_hash) % (2 ** 32)


# Recursively collect valid image paths from a source directory, deduplicating by resolved path
def get_images_from_source(source_path: Path) -> List[Path]:
    if not source_path.exists():
        return []

    valid_suffixes = {".jpg", ".jpeg", ".png"}
    images: List[Path] = []
    seen: Set[Path] = set()
    try:
        for path in source_path.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in valid_suffixes:
                continue
            norm = path.resolve()
            if norm in seen:
                continue
            seen.add(norm)
            images.append(path)
    except Exception as e:
        print(f"    Error scanning {source_path}: {str(e)[:50]}")

    return sorted(images, key=lambda x: str(x))


# Gather images grouped by source name for a given label (real or fake)
def collect_images_by_source(raw_dir: Path, label: str) -> Dict[str, List[Path]]:
    source_images: Dict[str, List[Path]] = {}

    for source_name in SOURCE_PERCENTAGES[label]:
        if "/" in source_name:
            parts = source_name.split("/")
            source_path = raw_dir / parts[0] / parts[1]
        else:
            source_path = raw_dir / source_name

        images = get_images_from_source(source_path)
        if images:
            source_images[source_name] = images
            print(f"    {source_name:35} {len(images):6,} images")
        else:
            print(f"    {source_name:35} NOT FOUND or empty")

    return source_images


# Distribute total_images across sources according to configured percentages
def _calc_source_counts(total_images: int, label: str) -> Dict[str, int]:
    percentages = SOURCE_PERCENTAGES[label]

    ideal_counts = {src: total_images * pct for src, pct in percentages.items()}
    counts = {src: int(count) for src, count in ideal_counts.items()}

    remainder = total_images - sum(counts.values())

    # Distribute remainder to sources with the largest fractional parts
    fractional_parts = sorted(
        percentages.keys(),
        key=lambda s: ideal_counts[s] % 1,
        reverse=True,
    )

    for i in range(remainder):
        counts[fractional_parts[i % len(fractional_parts)]] += 1

    final_total = sum(counts.values())
    if final_total != total_images:
        raise ValueError(
            f"Failed to distribute correctly: {final_total} != {total_images}"
        )

    return counts


# Sample images per source and split into train/val/test, excluding already-used images
def _sample_stratified(
    source_images: Dict[str, List[Path]],
    source_counts: Dict[str, int],
    seed: int,
    label: str,
    used_images: Optional[Set[Path]] = None,
) -> Tuple[List[Path], List[Path], List[Path], Set[Path]]:
    if used_images is None:
        used_images = set()

    train_all: List[Path] = []
    val_all: List[Path] = []
    test_all: List[Path] = []
    bad_images: Set[Path] = set()

    # Skip images that are corrupt or exceed the pixel size limit
    def is_usable_image(img_path: Path) -> bool:
        if img_path in bad_images:
            return False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(img_path) as img:
                    width, height = img.size
                    if width * height > MAX_IMAGE_PIXELS:
                        bad_images.add(img_path)
                        return False
                    img.load()
            return True
        except Exception:
            bad_images.add(img_path)
            return False

    for source_name, target_count in source_counts.items():
        if target_count == 0:
            continue

        if source_name not in source_images:
            print(f"    {source_name:35} missing, skipped")
            continue

        available = [img for img in source_images[source_name] if img not in used_images]

        if not available:
            print(f"    {source_name:35} no available images left")
            continue

        rng = random.Random(seed)

        candidates = list(available)
        rng.shuffle(candidates)
        sampled: List[Path] = []
        rejected = 0
        for candidate in candidates:
            if is_usable_image(candidate):
                sampled.append(candidate)
                if len(sampled) == target_count:
                    break
            else:
                rejected += 1

        source_size = len(sampled)
        train_idx = int(source_size * SPLIT_RATIOS[0])
        val_idx = train_idx + int(source_size * SPLIT_RATIOS[1])

        source_train = sampled[:train_idx]
        source_val = sampled[train_idx:val_idx]
        source_test = sampled[val_idx:]

        train_all.extend(source_train)
        val_all.extend(source_val)
        test_all.extend(source_test)

        used_images.update(sampled)

        pct = SOURCE_PERCENTAGES[label][source_name]
        status = "OK" if len(sampled) == target_count else "SHORT"
        print(
            f"    {source_name:35} {status:5} {len(sampled):4}/{target_count:4} "
            f"train:{len(source_train):4} val:{len(source_val):3} "
            f"test:{len(source_test):3} ({pct * 100:5.2f}%), rejected:{rejected}"
        )

    return train_all, val_all, test_all, used_images


# Infer the dataset source name from a file's directory path
def _extract_source_name(img_path: Path) -> str:
    all_known_sources: Set[str] = set()

    for source in SOURCE_PERCENTAGES["real"]:
        all_known_sources.add(source)
    for source in SOURCE_PERCENTAGES["fake"]:
        if "/" in source:
            all_known_sources.add(source.split("/")[0])
            all_known_sources.add(source.split("/")[1])
        else:
            all_known_sources.add(source)

    for part in reversed(img_path.parts):
        if part in all_known_sources:
            return part

    return img_path.parts[-2] if len(img_path.parts) > 2 else "unknown"


# Copy images to dest_dir with source-prefixed filenames; skip invalid or oversized files
def _copy_images(image_paths: List[Path], dest_dir: Path) -> Tuple[int, List[str]]:
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped: List[str] = []

    for src_img in image_paths:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(src_img) as img:
                    width, height = img.size
                    if width * height > MAX_IMAGE_PIXELS:
                        raise ValueError("image too large")
                    img.load()

            source_name = _extract_source_name(src_img)
            prefixed_name = f"{source_name}_{src_img.name}"
            dest_img = dest_dir / prefixed_name

            shutil.copy2(src_img, dest_img)
            copied += 1
        except Exception as e:
            skipped.append(f"{src_img.name}: {str(e)[:40]}")

    return copied, skipped


class BalancedDatasetSplitter:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.processed_dir = PROJECT_ROOT / "data" / "processed"
        self.used_real_images: Set[Path] = set()
        self.used_fake_images: Set[Path] = set()

    # Check all configured data sources exist and are non-empty before splitting
    def verify_sources(self) -> bool:
        print("Checking data sources:")

        all_ok = True

        for label in ("real", "fake"):
            print(f"\n  {label}:")
            for source in SOURCE_PERCENTAGES[label]:
                if "/" in source:
                    parts = source.split("/")
                    source_path = self.raw_dir / parts[0] / parts[1]
                else:
                    source_path = self.raw_dir / source

                count = len(get_images_from_source(source_path))
                status = "OK" if count > 0 else "MISSING"
                print(f"    {status}  {source:35} {count:6,} images")

                if count == 0:
                    all_ok = False

        if not all_ok:
            print("\nOne or more sources are missing or empty. Aborting.")

        return all_ok

    # Build stage 1, 2, and 3 splits sequentially, ensuring no image is reused across stages
    def split_all_stages(self) -> bool:
        print(f"Building dataset (target: {TOTAL_IMAGES:,} images)")

        print("\nCollecting real images by source:")
        real_source_images = collect_images_by_source(self.raw_dir, "real")

        print("\nCollecting fake images by source:")
        fake_source_images = collect_images_by_source(self.raw_dir, "fake")

        counts_per_stage: Dict[int, Dict] = {}

        print("\nStage 1: 500 images (250 real + 250 fake)")

        real_s1 = _calc_source_counts(250, "real")
        fake_s1 = _calc_source_counts(250, "fake")

        rt, rv, rte, self.used_real_images = _sample_stratified(
            real_source_images, real_s1,
            _make_seed(self.seed, "stage1_real"), "real",
            self.used_real_images,
        )
        ft, fv, fte, self.used_fake_images = _sample_stratified(
            fake_source_images, fake_s1,
            _make_seed(self.seed, "stage1_fake"), "fake",
            self.used_fake_images,
        )

        counts_per_stage[1] = self._create_and_split_stratified(
            "stage_1", rt, rv, rte, ft, fv, fte, stage_num=1
        )

        print("\nStage 2: 5,000 images (2,500 real + 2,500 fake)")

        real_s2 = _calc_source_counts(2500, "real")
        fake_s2 = _calc_source_counts(2500, "fake")

        rt, rv, rte, self.used_real_images = _sample_stratified(
            real_source_images, real_s2,
            _make_seed(self.seed, "stage2_real"), "real",
            self.used_real_images,
        )
        ft, fv, fte, self.used_fake_images = _sample_stratified(
            fake_source_images, fake_s2,
            _make_seed(self.seed, "stage2_fake"), "fake",
            self.used_fake_images,
        )

        counts_per_stage[2] = self._create_and_split_stratified(
            "stage_2", rt, rv, rte, ft, fv, fte, stage_num=2
        )

        print("\nStage 3: 10,000 images (5,000 real + 5,000 fake)")

        real_s3 = _calc_source_counts(5000, "real")
        fake_s3 = _calc_source_counts(5000, "fake")

        rt, rv, rte, self.used_real_images = _sample_stratified(
            real_source_images, real_s3,
            _make_seed(self.seed, "stage3_real"), "real",
            self.used_real_images,
        )
        ft, fv, fte, self.used_fake_images = _sample_stratified(
            fake_source_images, fake_s3,
            _make_seed(self.seed, "stage3_fake"), "fake",
            self.used_fake_images,
        )

        counts_per_stage[3] = self._create_and_split_stratified(
            "stage_3", rt, rv, rte, ft, fv, fte, stage_num=3
        )

        self._print_summary(counts_per_stage)
        return True

    # Copy images into the processed directory structure and write split metadata
    def _create_and_split_stratified(
        self,
        stage_folder: str,
        real_train: List[Path], real_val: List[Path], real_test: List[Path],
        fake_train: List[Path], fake_val: List[Path], fake_test: List[Path],
        stage_num: int,
    ) -> Dict[str, int]:
        print("  Copying to processed directory...")

        stage_dir = self.processed_dir / stage_folder

        cr_tr, sk_cr_tr = _copy_images(real_train, stage_dir / "train" / "real")
        cf_tr, sk_cf_tr = _copy_images(fake_train, stage_dir / "train" / "ai_generated")
        cr_v, sk_cr_v = _copy_images(real_val, stage_dir / "validation" / "real")
        cf_v, sk_cf_v = _copy_images(fake_val, stage_dir / "validation" / "ai_generated")
        cr_te, sk_cr_te = _copy_images(real_test, stage_dir / "test" / "real")
        cf_te, sk_cf_te = _copy_images(fake_test, stage_dir / "test" / "ai_generated")

        total_skipped = sum(
            len(s) for s in (sk_cr_tr, sk_cf_tr, sk_cr_v, sk_cf_v, sk_cr_te, sk_cf_te)
        )
        if total_skipped:
            print(f"  Warning: {total_skipped} invalid files were skipped:")
            for label, skipped in [
                ("Real train", sk_cr_tr), ("Fake train", sk_cf_tr),
                ("Real val", sk_cr_v), ("Fake val", sk_cf_v),
                ("Real test", sk_cr_te), ("Fake test", sk_cf_te),
            ]:
                if skipped:
                    print(f"    {label}: {len(skipped)} skipped")

        total_real = cr_tr + cr_v + cr_te
        total_fake = cf_tr + cf_v + cf_te
        metadata = {
            "stage": stage_num,
            "total": total_real + total_fake,
            "real": total_real,
            "fake": total_fake,
            "train": cr_tr + cf_tr,
            "validation": cr_v + cf_v,
            "test": cr_te + cf_te,
        }

        metadata_file = PROJECT_ROOT / "data" / "metadata" / f"stage_{stage_num}_split_info.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Train: {cr_tr:4} real + {cf_tr:4} fake = {cr_tr + cf_tr:5}")
        print(f"  Val:   {cr_v:4} real + {cf_v:4} fake = {cr_v + cf_v:5}")
        print(f"  Test:  {cr_te:4} real + {cf_te:4} fake = {cr_te + cf_te:5}")

        return {
            "total_real_train": cr_tr,
            "total_fake_train": cf_tr,
            "total_real_val": cr_v,
            "total_fake_val": cf_v,
            "total_real_test": cr_te,
            "total_fake_test": cf_te,
        }

    def _print_summary(self, counts_per_stage: Dict[int, Dict]) -> None:
        stage_totals: Dict[int, int] = {}

        for stage_num, c in counts_per_stage.items():
            stage_totals[stage_num] = sum(c.values())

        total_images = sum(stage_totals.values())
        diff = TOTAL_IMAGES - total_images

        print("\nDataset complete:")
        for stage_num, total in sorted(stage_totals.items()):
            print(f"  stage {stage_num}: {total:,} images")
        print(f"  total: {total_images:,} / {TOTAL_IMAGES:,} (diff: {diff:,})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create balanced dataset with weighted source sampling"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    splitter = BalancedDatasetSplitter(seed=args.seed)

    if not splitter.verify_sources():
        raise SystemExit(1)

    splitter.split_all_stages()


if __name__ == "__main__":
    main()
