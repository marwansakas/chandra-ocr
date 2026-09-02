"""
Evaluate custom CTC line predictions against their line references.

Reads:
  thesis/predictions_ctc/predictions.csv

Writes:
  thesis/reports/ctc_line_evaluation.csv
  thesis/reports/ctc_line_evaluation_summary.md

Run:
  python thesis/scripts/evaluate_ctc_predictions.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import common


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=common.PROJECT_ROOT / "thesis" / "predictions_ctc" / "predictions.csv",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=common.PROJECT_ROOT / "thesis" / "reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise SystemExit(f"Missing predictions file: {args.predictions}")

    rows = []
    with args.predictions.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            reference = common.normalize_text_for_evaluation(row.get("reference") or "")
            prediction = common.normalize_text_for_evaluation(row.get("prediction") or "")
            ref_words = common.tokenize_words(reference)
            pred_words = common.tokenize_words(prediction)
            cer = common.ratio_error(reference, prediction)
            wer = common.ratio_error(ref_words, pred_words)
            rows.append(
                {
                    **row,
                    "reference_char_count": len(reference),
                    "prediction_char_count": len(prediction),
                    "cer": f"{cer:.4f}",
                    "wer": f"{wer:.4f}",
                    "line_reliability_score": f"{max(0.0, 1.0 - cer):.4f}",
                }
            )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    common.write_csv(args.reports_dir / "ctc_line_evaluation.csv", rows)

    by_page: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_page[row["page_id"]].append(row)

    lines = [
        "# CTC Line Evaluation Summary",
        "",
        f"- Evaluated lines: {len(rows)}",
        f"- Evaluated pages: {len(by_page)}",
        f"- Mean CER %: {pct(mean([float(r['cer']) for r in rows]))}",
        f"- Mean WER %: {pct(mean([float(r['wer']) for r in rows]))}",
        "",
        "| page_id | lines | mean CER % | mean WER % |",
        "| --- | ---: | ---: | ---: |",
    ]
    for page_id, page_rows in sorted(by_page.items()):
        lines.append(
            f"| {page_id} | {len(page_rows)} | "
            f"{pct(mean([float(r['cer']) for r in page_rows]))} | "
            f"{pct(mean([float(r['wer']) for r in page_rows]))} |"
        )

    summary_path = args.reports_dir / "ctc_line_evaluation_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluated {len(rows)} CTC line predictions.")
    print(f"Wrote {args.reports_dir / 'ctc_line_evaluation.csv'}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
