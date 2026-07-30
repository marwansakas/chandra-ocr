"""
Report OCR coverage and metadata gaps.

Writes:
  thesis/reports/ocr_coverage_summary.md
  thesis/reports/ocr_coverage_by_group.csv
  thesis/reports/missing_ocr_pages.csv

Run:
  python thesis/scripts/report_coverage.py
"""

from __future__ import annotations

from collections import Counter, defaultdict

import common


GROUPS = [
    ("place",),
    ("grade",),
    ("place", "grade"),
    ("place", "grade", "gender"),
]


def key_value(record: dict, field: str):
    value = record.get(field)
    return "unknown" if value is None or value == "" else value


def pct(part: int, whole: int) -> str:
    return "0.0" if whole == 0 else f"{(100 * part / whole):.1f}"


def main() -> None:
    cfg = common.load_config()
    metadata_dir = common.resolve(cfg["metadata_dir"])
    reports_dir = common.resolve(cfg["reports_dir"])
    pages_path = metadata_dir / "pages.jsonl"

    if not pages_path.exists():
        raise SystemExit("Missing pages.jsonl. Run build_metadata.py and collect_ocr.py first.")

    records = list(common.read_jsonl(pages_path))
    total = len(records)
    linked = sum(1 for r in records if r.get("ocr_output_path"))
    nonempty = sum(1 for r in records if r.get("ocr_text_present"))
    missing = [r for r in records if not r.get("ocr_output_path")]

    group_rows = []
    for fields in GROUPS:
        buckets = defaultdict(list)
        for record in records:
            key = tuple(key_value(record, field) for field in fields)
            buckets[key].append(record)
        for key, rows in sorted(buckets.items(), key=lambda item: tuple(map(str, item[0]))):
            row_total = len(rows)
            row_linked = sum(1 for r in rows if r.get("ocr_output_path"))
            row_nonempty = sum(1 for r in rows if r.get("ocr_text_present"))
            group_rows.append(
                {
                    "group": " x ".join(fields),
                    "group_value": " | ".join(map(str, key)),
                    "pages": row_total,
                    "ocr_linked": row_linked,
                    "ocr_nonempty": row_nonempty,
                    "ocr_missing": row_total - row_linked,
                    "coverage_pct": pct(row_linked, row_total),
                }
            )

    missing_rows = [
        {
            "page_id": r["page_id"],
            "image_path": r["image_path"],
            "place": r["place"],
            "grade": key_value(r, "grade"),
            "gender": key_value(r, "gender"),
            "student_id": r["student_id"],
        }
        for r in missing
    ]

    common.write_csv(reports_dir / "ocr_coverage_by_group.csv", group_rows)
    common.write_csv(reports_dir / "missing_ocr_pages.csv", missing_rows)

    anomaly_counts = Counter()
    anomaly_counts["unknown_grade"] = sum(1 for r in records if r.get("grade") is None)
    anomaly_counts["missing_gender"] = sum(1 for r in records if r.get("gender") is None)
    anomaly_counts["unparsed_filename"] = sum(1 for r in records if not r.get("filename_parsed"))

    summary = [
        "# OCR Coverage Summary",
        "",
        f"- Total source pages: {total}",
        f"- OCR files linked: {linked} ({pct(linked, total)}%)",
        f"- Non-empty OCR files: {nonempty} ({pct(nonempty, total)}%)",
        f"- Missing OCR files: {total - linked}",
        "",
        "## Metadata Anomalies",
        "",
    ]
    for name, value in anomaly_counts.items():
        summary.append(f"- {name}: {value}")
    summary.extend(
        [
            "",
            "## Next Actions",
            "",
            "1. Finish OCR for missing pages or choose a balanced sample.",
            "2. Create ground-truth labels for a subset before judging spelling.",
            "3. Use OCR evaluation results to decide which pages are reliable enough for scoring.",
        ]
    )
    summary_path = reports_dir / "ocr_coverage_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {reports_dir / 'ocr_coverage_by_group.csv'}")
    print(f"Wrote {reports_dir / 'missing_ocr_pages.csv'}")


if __name__ == "__main__":
    main()
