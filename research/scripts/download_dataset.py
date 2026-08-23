import sys
import argparse
from pathlib import Path
from PIL import Image
from io import BytesIO

# every default path is built from here, so the script runs from any working
# directory rather than only from research/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FORENSYNTHS_DIR = RAW_DIR / "forensynths"


# Stream ForenSynths from HuggingFace and save as JPEGs
def download_forensynths(target_images=10000, output_dir=FORENSYNTHS_DIR):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print("Loading ForenSynths from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found. Install with:")
        print("pip install datasets pillow huggingface_hub")
        return False

    try:
        dataset = load_dataset("nebula/DF-arrow", split="test", streaming=True)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        print("Make sure you have internet and datasets library installed.")
        return False

    print(f"Downloading {target_images} images...")

    successful = 0
    failed = 0

    for idx, item in enumerate(dataset):
        if idx >= target_images:
            break
        try:
            img = item.get("image") or item.get("img")

            if img is None:
                raise ValueError("No image field found in dataset item")

            if isinstance(img, Image.Image):
                image = img
            else:
                image = Image.open(BytesIO(img)) if isinstance(img, bytes) else img

            if image.mode != "RGB":
                image = image.convert("RGB")

            output_file = output_path / f"forensynths_{idx:06d}.jpg"
            image.save(output_file, quality=95)

            successful += 1

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{target_images}...")

        except Exception as e:
            failed += 1
            if idx < 10:
                print(f"   Image {idx} failed: {str(e)[:50]}")

    print(f"\nDone: {successful} downloaded, {failed} failed, into {output_dir}/")

    return successful > 0


# Print a status summary of all expected raw data directories
def check_data_sources():
    sources = {
        "ImageNet": RAW_DIR / "imagenet",
        "COCO": RAW_DIR / "coco",
        "Unsplash": RAW_DIR / "unsplash",
        "GenImage": RAW_DIR / "genimage",
        "ForenSynths": RAW_DIR / "forensynths",
        "CIFAKE": RAW_DIR / "cifake",
        "Flickr30k": RAW_DIR / "flickr30k",
    }

    print("\nData Sources Status:")
    print("-" * 50)

    for name, path in sources.items():
        path_obj = Path(path)
        if path_obj.exists():
            # rglob, not glob: coco, genimage, cifake and forensynths keep their
            # images in subdirectories, so a top-level glob reported them empty
            count = len(list(path_obj.rglob("*.*")))
            status = "OK" if count > 0 else "EMPTY"
            print(f"{name:15} {status:15} ({count} files)")
        else:
            print(f"{name:15} NOT FOUND")

    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download datasets: ForenSynths from HuggingFace"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="forensynths",
        choices=["forensynths"],
        help="Which dataset to download (default: forensynths)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target number of images to download"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (auto-determined if not specified)"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only check data source status, don't download"
    )

    args = parser.parse_args()

    if args.status_only:
        check_data_sources()
    elif args.dataset == "forensynths":
        target = args.target or 10000
        output_dir = args.output_dir or FORENSYNTHS_DIR
        success = download_forensynths(
            target_images=target,
            output_dir=output_dir
        )
        check_data_sources()
        sys.exit(0 if success else 1)
