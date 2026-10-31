"""Score original-Paddle and diffusion-selected predictions against ground truth."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pretrained_ocr.metrics import DEFAULT_NORMALIZATION, aggregate, score


def read_predictions(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("predictions.csv contains no rows")
    required = {"image", "prediction", "original_prediction", "selected"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"predictions.csv is missing columns: {sorted(missing)}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="predictions.csv written by diffusion_paddle_infer.py")
    parser.add_argument("--output", type=Path,
                        help="Default: diffusion_metrics.json beside predictions.csv")
    parser.add_argument("--normalization", choices=("raw", "standard", "arabic-light"),
                        default=DEFAULT_NORMALIZATION)
    args = parser.parse_args(argv)
    prediction_path = args.predictions.resolve()
    output = (args.output or prediction_path.with_name("diffusion_metrics.json")).resolve()
    rows, original_rows, selected_rows = [], [], []
    for record in read_predictions(prediction_path):
        image = Path(record["image"])
        label = image.parent / "ground_truth.txt"
        if not label.is_file():
            raise ValueError(f"Missing ground_truth.txt next to image: {image}")
        reference = label.read_text(encoding="utf-8-sig")
        original = score(reference, record["original_prediction"], args.normalization)
        selected = score(reference, record["prediction"], args.normalization)
        original_rows.append(original)
        selected_rows.append(selected)
        rows.append({
            "image": str(image), "selected": record["selected"], "reference": reference,
            "original_prediction": record["original_prediction"], "prediction": record["prediction"],
            "original_cer": original["cer"], "selected_cer": selected["cer"],
            "original_exact_match": original["exact_match"], "selected_exact_match": selected["exact_match"],
        })
    summary = {
        "predictions": str(prediction_path), "normalization": args.normalization,
        "samples": len(rows),
        "original_paddle": aggregate(original_rows),
        "paddle_diffusion": aggregate(selected_rows),
        "selection_counts": {name: sum(row["selected"] == name for row in rows)
                             for name in sorted({row["selected"] for row in rows})},
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    scored = output.with_name("diffusion_scored_predictions.csv")
    with scored.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    def pct(value):
        return "n/a" if value is None else f"{100 * value:.2f}%"
    print(f"Samples: {summary['samples']}")
    print("Original Paddle: CER " + pct(summary["original_paddle"]["cer"]) +
          "; exact lines " + pct(summary["original_paddle"]["exact_match_rate"]))
    print("Paddle + diffusion: CER " + pct(summary["paddle_diffusion"]["cer"]) +
          "; exact lines " + pct(summary["paddle_diffusion"]["exact_match_rate"]))
    print(f"Metrics: {output}\nScored rows: {scored}")


if __name__ == "__main__":
    main()
