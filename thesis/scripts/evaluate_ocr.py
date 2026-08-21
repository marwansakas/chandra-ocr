"""
Evaluate Chandra OCR against manually transcribed ground truth.

Ground-truth files are expected at:
  thesis/labels/ground_truth/<page_id>.txt

Writes:
  thesis/reports/ocr_evaluation_pages.csv
  thesis/reports/ocr_evaluation_summary.md

Run:
  python thesis/scripts/evaluate_ocr.py

Evaluate a compatible baseline without relinking metadata:
    python thesis/scripts/evaluate_ocr.py --ocr-root thesis/invizo/outputs \
            --reports-dir thesis/invizo/reports
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import common


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ocr-root",
        default=None,
        help=(
            "Alternative OCR root with the standard "
            "<place>/<grade>/<stem>/<stem>/<stem>.md layout."
        ),
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory for the CSV and Markdown reports (defaults to config.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = common.load_config()
    metadata_dir = common.resolve(cfg["metadata_dir"])
    gt_root = common.resolve(cfg["ground_truth_dir"])
    reports_dir = (
        common.resolve(args.reports_dir)
        if args.reports_dir is not None
        else common.resolve(cfg["reports_dir"])
    )
    ocr_root = common.resolve(args.ocr_root) if args.ocr_root is not None else None

    records = list(common.read_jsonl(metadata_dir / "pages.jsonl"))
    rows = []

    for record in records:
        gt_path = common.ground_truth_path(gt_root, record)
        if not gt_path.exists():
            continue
        gt_text = common.normalize_text_for_evaluation(common.read_text(gt_path))
        if not gt_text:
            continue

        if ocr_root is None:
            if not record.get("ocr_output_path"):
                continue
            ocr_path = common.resolve(record["ocr_output_path"])
            ocr_output_path = record["ocr_output_path"]
        else:
            ocr_path = common.ocr_markdown_path(ocr_root, record)
            if ocr_path is None:
                continue
            ocr_output_path = str(ocr_path.relative_to(common.PROJECT_ROOT))

        ocr_text = common.normalize_text_for_evaluation(common.read_text(ocr_path))

        gt_words = common.tokenize_words(gt_text)
        ocr_words = common.tokenize_words(ocr_text)
        cer = common.ratio_error(gt_text, ocr_text)
        wer = common.ratio_error(gt_words, ocr_words)
        rows.append(
            {
                "page_id": record["page_id"],
                "student_id": record["student_id"],
                "place": record["place"],
                "grade": record.get("grade") or "unknown",
                "gender": record.get("gender") or "unknown",
                "gt_char_count": len(gt_text),
                "ocr_char_count": len(ocr_text),
                "gt_word_count": len(gt_words),
                "ocr_word_count": len(ocr_words),
                "cer": f"{cer:.4f}",
                "wer": f"{wer:.4f}",
                "ocr_reliability_score": f"{max(0.0, 1.0 - wer):.4f}",
                "ground_truth_path": str(gt_path.relative_to(common.PROJECT_ROOT)),
                "ocr_output_path": ocr_output_path,
            }
        )

    common.write_csv(reports_dir / "ocr_evaluation_pages.csv", rows)

    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[f"grade={row['grade']}"].append(row)
        groups[f"place={row['place']}"].append(row)

    lines = [
        "# OCR Evaluation Summary",
        "",
        f"- Evaluated pages: {len(rows)}",
        "",
        "| group | pages | mean CER % | mean WER % | mean reliability |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for group, group_rows in sorted(groups.items()):
        cers = [float(r["cer"]) for r in group_rows]
        wers = [float(r["wer"]) for r in group_rows]
        reliabilities = [float(r["ocr_reliability_score"]) for r in group_rows]
        lines.append(
            f"| {group} | {len(group_rows)} | {pct(mean(cers))} | "
            f"{pct(mean(wers))} | {mean(reliabilities):.3f} |"
        )

    lines.extend(
        [
            "",
            "Use pages with low WER/CER for spelling analysis. High OCR error means the page needs manual correction before judging the child.",
        ]
    )
    summary_path = reports_dir / "ocr_evaluation_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Evaluated {len(rows)} labeled pages.")
    print(f"Wrote {reports_dir / 'ocr_evaluation_pages.csv'}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
