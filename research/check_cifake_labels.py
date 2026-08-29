"""Measure how many CIFAKE images carry the wrong label in the processed splits.

split_dataset.py lists "cifake" under both the real and the fake source maps,
and get_images_from_source recurses, so both labels draw from the same pool of
FAKE/ and REAL/ together. This measures the consequence rather than estimating
it.

Run from research/:
    python check_cifake_labels.py

Read-only. Writes nothing, changes nothing.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "cifake"
PROC = ROOT / "data" / "processed"

EXT = {".jpg", ".jpeg", ".png"}


def names(folder: Path) -> set[str]:
    if not folder.exists():
        return set()
    return {p.name for p in folder.rglob("*") if p.suffix.lower() in EXT}


def main() -> None:
    fake_names = names(RAW / "FAKE")
    real_names = names(RAW / "REAL")

    if not fake_names and not real_names:
        sys.exit(f"No CIFAKE data found under {RAW}")

    collide = fake_names & real_names

    print(f"raw cifake/FAKE : {len(fake_names):,} files")
    print(f"raw cifake/REAL : {len(real_names):,} files")
    print(f"names appearing in both folders: {len(collide):,}")
    if collide:
        print("  Filenames are not unique across the two folders, so any image "
              "whose name appears in both cannot be traced, and copies into the "
              "same split directory will have overwritten one another.")
    print()

    total_wrong = 0
    total_imgs = 0

    for stage in (1, 2, 3):
        stage_dir = PROC / f"stage_{stage}"
        if not stage_dir.exists():
            continue
        print(f"stage {stage}")
        for split in ("train", "validation", "test"):
            for label, wrong_src in (("real", fake_names),
                                     ("ai_generated", real_names)):
                d = stage_dir / split / label
                if not d.exists():
                    continue

                all_files = [p for p in d.iterdir() if p.suffix.lower() in EXT]
                cif = [p for p in all_files if p.name.startswith("cifake_")]
                # strip the source prefix that _copy_images adds
                origin = [p.name[len("cifake_"):] for p in cif]

                wrong = sum(1 for n in origin if n in wrong_src and n not in collide)
                unknown = sum(1 for n in origin if n in collide)

                total_wrong += wrong
                total_imgs += len(all_files)

                if cif:
                    pct = 100.0 * wrong / len(all_files) if all_files else 0.0
                    note = f", {unknown} untraceable" if unknown else ""
                    print(f"  {split:11} {label:13} "
                          f"{len(all_files):5,} images, "
                          f"{len(cif):5,} from cifake, "
                          f"{wrong:5,} mislabelled ({pct:.1f}%){note}")
        print()

    if total_imgs:
        print(f"TOTAL: {total_wrong:,} mislabelled of {total_imgs:,} images "
              f"({100.0 * total_wrong / total_imgs:.2f}%)")
        print()
        print("Mislabelled here means a file drawn from cifake/FAKE that was "
              "copied into a real/ directory, or one from cifake/REAL copied "
              "into ai_generated/.")


if __name__ == "__main__":
    main()
