import os
import sys
import argparse
from pathlib import Path
import csv
import requests
from PIL import Image
from io import BytesIO
import time
import random

# every default path is built from here, so the script runs from any working
# directory rather than only from research/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
UNSPLASH_DIR = RAW_DIR / "unsplash"
UNSPLASH_MANIFEST = UNSPLASH_DIR / "photos.csv000"


# Load already-downloaded URLs from the manifest file to avoid re-downloading
def _load_downloaded_urls(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()

    urls = set()
    with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            url = line.strip()
            if url:
                urls.add(url)
    return urls


def _append_downloaded_urls(manifest_path: Path, urls: list[str]) -> None:
    if not urls:
        return

    with open(manifest_path, "a", encoding="utf-8") as f:
        for url in urls:
            f.write(url + "\n")


# Determine the next sequential index for naming output files
def _next_unsplash_index(output_path: Path) -> int:
    max_idx = -1
    for img in output_path.glob("unsplash_*.jpg"):
        stem = img.stem
        try:
            idx = int(stem.split("_")[1])
            max_idx = max(max_idx, idx)
        except (IndexError, ValueError):
            continue
    return max_idx + 1


# Download Unsplash images via direct URL, resuming from the manifest
def download_unsplash_images(target_images=4000, output_dir=UNSPLASH_DIR,
                             metadata_file=UNSPLASH_MANIFEST,
                             random_sample=False, delay=0.5):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "downloaded_urls.txt"

    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found: {metadata_file}")
        return False

    photo_urls = []
    try:
        with open(metadata_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if 'photo_image_url' in row:
                    photo_urls.append(row['photo_image_url'])
    except Exception as e:
        print(f"ERROR: Failed to read metadata: {e}")
        return False

    unique_urls = list(dict.fromkeys(photo_urls))
    total_available = len(unique_urls)
    print(f"Found {total_available} photos in metadata")

    if total_available == 0:
        print("ERROR: No photos found in metadata file")
        return False

    downloaded_urls = _load_downloaded_urls(manifest_path)
    available_urls = [u for u in unique_urls if u and u != "None" and u not in downloaded_urls]

    print(f"Already downloaded (manifest): {len(downloaded_urls)}")
    print(f"Remaining unique URLs available: {len(available_urls)}")

    if len(available_urls) == 0:
        print("No new URLs left to download.")
        return True

    if random_sample:
        photo_urls = random.sample(available_urls, min(target_images, len(available_urls)))
    else:
        photo_urls = available_urls[:target_images]

    images_to_download = len(photo_urls)
    print(f"Downloading {images_to_download} images to {output_dir}...")

    successful = 0
    failed = 0
    new_downloaded_urls = []
    next_idx = _next_unsplash_index(output_path)

    for idx, url in enumerate(photo_urls):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))

            if image.mode != "RGB":
                image = image.convert("RGB")

            output_file = output_path / f"unsplash_{next_idx:06d}.jpg"
            image.save(output_file, quality=95)
            next_idx += 1

            successful += 1
            new_downloaded_urls.append(url)

            if (idx + 1) % 50 == 0:
                print(f"  Downloaded {idx + 1}/{images_to_download}...")

            if delay > 0:
                time.sleep(delay)

        except Exception as e:
            failed += 1
            if idx < 5:
                print(f"  Image {idx} failed: {str(e)[:60]}")

    print(f"\nDone: {successful} downloaded, {failed} failed, into {output_dir}/")

    _append_downloaded_urls(manifest_path, new_downloaded_urls)
    print(f"  Updated manifest: {manifest_path}")

    return successful > 0


# Count downloaded images and verify against manifest
def verify_unsplash_images(output_dir=UNSPLASH_DIR):
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"Directory not found: {output_dir}")
        return 0

    image_files = list(output_path.glob("unsplash_*.jpg"))
    manifest_path = output_path / "downloaded_urls.txt"
    manifest_count = len(_load_downloaded_urls(manifest_path))
    print(f"\nUnsplash images downloaded: {len(image_files)}")
    print(f"Manifest URLs tracked: {manifest_count}")

    if len(image_files) > 0:
        total_size = sum(f.stat().st_size for f in image_files)
        avg_size = total_size / len(image_files) / (1024 * 1024)
        print(f"Total size: {total_size / (1024**3):.2f} GB")
        print(f"Average size: {avg_size:.2f} MB per image")

    return len(image_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Unsplash images from dataset metadata"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=4000,
        help="Target number of images to download (default: 4000)"
    )
    parser.add_argument(
        "--output-dir",
        default=UNSPLASH_DIR,
        help="Output directory for images (default: data/raw/unsplash)"
    )
    parser.add_argument(
        "--metadata-file",
        default=UNSPLASH_MANIFEST,
        help="Path to photos.csv metadata file"
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample images instead of taking first N"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between downloads (default: 0.5)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify/count existing images, don't download"
    )

    args = parser.parse_args()

    if args.verify:
        verify_unsplash_images(args.output_dir)
    else:
        success = download_unsplash_images(
            target_images=args.target,
            output_dir=args.output_dir,
            metadata_file=args.metadata_file,
            random_sample=args.random_sample,
            delay=args.delay
        )
        verify_unsplash_images(args.output_dir)
        sys.exit(0 if success else 1)
