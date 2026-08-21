"""Print every recorded metric as a table, for checking the dissertation against
the files rather than by eye.

Run from the research directory:
    python scripts/report_metrics.py
    python scripts/report_metrics.py --csv metrics.csv

Values are printed to four decimal places because a transcription error already
reached the report at two: Hybrid stage 3 accuracy was written as 84.41% when
the file says 84.1791%.
"""

import argparse
import csv
import json
from pathlib import Path

# every default path is built from here, so the script runs from any working
# directory rather than only from research/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS = PROJECT_ROOT / "results"
MODELS = ["cnn", "fft", "hybrid", "hybrid_norm", "hybrid_proj", "stm"]
STAGES = [1, 2, 3]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
DEGRADATIONS = ["none", "light", "heavy"]


def load(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # NaN appears in the unseen files where a metric is undefined
        return json.loads(path.read_text(encoding="utf-8").replace("NaN", "null"))


def pct(value):
    return "-" if value is None else f"{value * 100:8.4f}"


def collect():
    rows = []

    for model in MODELS:
        for stage in STAGES:
            d = load(RESULTS / model / f"stage_{stage}" / "test_metrics.json")
            if d:
                rows.append({
                    "table": "test",
                    "model": model,
                    "stage": stage,
                    "split": "held-out test",
                    **{m: d.get(m) for m in METRICS},
                })

    for model in MODELS:
        for stage in STAGES:
            d = load(RESULTS / model / f"stage_{stage}" / "unseen" / "unseen_chameleon.json")
            for deg in DEGRADATIONS:
                r = (d or {}).get("results", {}).get(deg)
                if r:
                    rows.append({
                        "table": "chameleon",
                        "model": model,
                        "stage": stage,
                        "split": f"chameleon/{deg}",
                        **{m: r.get(m) for m in METRICS},
                    })

    for model in MODELS:
        for stage in STAGES:
            d = load(RESULTS / model / f"stage_{stage}" / "unseen" / "unseen_mnw.json")
            for deg in DEGRADATIONS:
                r = (d or {}).get("results", {}).get(deg)
                if r:
                    # MNW is AI-only, so accuracy is the detection rate
                    rows.append({
                        "table": "mnw",
                        "model": model,
                        "stage": stage,
                        "split": f"mnw/{deg}",
                        "detection_rate": r.get("accuracy"),
                    })

    return rows


def print_table(title, rows, columns):
    # width follows the longest model name, so adding a variant does not shunt
    # every column right of it out of alignment
    name_w = max([len(m) for m in MODELS] + [5]) + 2
    print(f"\n{title}")
    print("-" * (name_w + 24 + 10 * len(columns)))
    print(f"{'model':{name_w}}{'stage':>6}  {'split':16}" + "".join(f"{c:>10}" for c in columns))
    for r in rows:
        print(f"{r['model']:{name_w}}{r['stage']:>6}  {r['split']:16}"
              + "".join(f"{pct(r.get(c)):>10}" for c in columns))


def main():
    parser = argparse.ArgumentParser(description="Tabulate every recorded metric")
    parser.add_argument("--csv", type=Path, help="also write the rows to a CSV")
    args = parser.parse_args()

    if not RESULTS.is_dir():
        raise SystemExit(f"No results directory at {RESULTS}")

    rows = collect()

    print_table("Held-out test set, percent",
                [r for r in rows if r["table"] == "test"], METRICS)
    print_table("Chameleon, unseen generators, percent",
                [r for r in rows if r["table"] == "chameleon"], METRICS)
    print_table("MNW, AI-only, detection rate percent",
                [r for r in rows if r["table"] == "mnw"], ["detection_rate"])

    if args.csv:
        fields = ["table", "model", "stage", "split", *METRICS, "detection_rate"]
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
