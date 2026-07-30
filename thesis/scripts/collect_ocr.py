"""
Attach existing Chandra OCR outputs to page metadata.

Reads:
  thesis/metadata/pages.jsonl
  output/**/<stem>.md

Writes:
  thesis/metadata/pages.jsonl
  thesis/metadata/pages.csv
  thesis/metadata/ocr_index.jsonl

Run:
  python thesis/scripts/collect_ocr.py
"""

from __future__ import annotations

import json
from pathlib import Path

import common


def main() -> None:
    cfg = common.load_config()
    metadata_dir = common.resolve(cfg["metadata_dir"])
    ocr_root = common.resolve(cfg["existing_ocr_dir"])
    gt_root = common.resolve(cfg["ground_truth_dir"])
    pages_path = metadata_dir / "pages.jsonl"

    if not pages_path.exists():
        raise SystemExit("Missing pages.jsonl. Run: python thesis/scripts/build_metadata.py")

    records = list(common.read_jsonl(pages_path))
    index_records = []
    linked = 0
    nonempty = 0

    for record in records:
        md_path = common.ocr_markdown_path(ocr_root, record)
        record["ocr_output_path"] = None
        record["ocr_text_present"] = False
        gt_path = common.ground_truth_path(gt_root, record)
        record["ground_truth_path"] = None
        record["ground_truth_present"] = False
        if gt_path.exists():
            record["ground_truth_path"] = str(gt_path.relative_to(common.PROJECT_ROOT))
            record["ground_truth_present"] = bool(common.read_text(gt_path).strip())

        if md_path is None:
            continue

        text = common.read_text(md_path).strip()
        rel_md_path = str(md_path.relative_to(common.PROJECT_ROOT))
        record["ocr_output_path"] = rel_md_path
        record["ocr_text_present"] = bool(text)
        linked += 1
        nonempty += int(bool(text))
        index_records.append(
            {
                "page_id": record["page_id"],
                "image_path": record["image_path"],
                "ocr_output_path": rel_md_path,
                "ocr_char_count": len(text),
                "ocr_word_count": len(common.tokenize_words(text)),
            }
        )

    common.write_jsonl(pages_path, records)
    common.write_csv(metadata_dir / "pages.csv", records, fields=list(records[0].keys()))
    common.write_jsonl(metadata_dir / "ocr_index.jsonl", index_records)

    print(f"Updated {len(records)} page records.")
    print(f"Linked OCR files : {linked}")
    print(f"Non-empty OCR    : {nonempty}")
    print(f"Missing OCR      : {len(records) - linked}")
    print(f"Wrote index      : {metadata_dir / 'ocr_index.jsonl'}")
    print("\nNext: python thesis/scripts/report_coverage.py")


if __name__ == "__main__":
    main()
