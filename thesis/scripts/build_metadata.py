"""
Build the master metadata for every page.

Outputs (in thesis/metadata/):
  pages.jsonl   - one JSON object per page (full record, primary source of truth)
  pages.csv     - same data, spreadsheet-friendly (for the advisor / quick filters)

Read-only with respect to split_papers. Only writes into thesis/metadata/.

    python thesis/scripts/build_metadata.py
"""

from __future__ import annotations

import csv

import common


def main() -> None:
    cfg = common.load_config()
    raw_root = common.resolve(cfg["raw_data_dir"])
    meta_dir = common.resolve(cfg["metadata_dir"])
    meta_dir.mkdir(parents=True, exist_ok=True)

    records = [common.parse_pdf(p, raw_root) for p in common.iter_pdfs(raw_root)]
    dicts = [r.to_dict() for r in records]

    jsonl_path = meta_dir / "pages.jsonl"
    common.write_jsonl(jsonl_path, dicts)

    csv_path = meta_dir / "pages.csv"
    if dicts:
        fields = list(dicts[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for d in dicts:
                writer.writerow(d)

    print(f"Wrote {len(dicts)} page records:")
    print(f"  {jsonl_path}")
    print(f"  {csv_path}")
    print("\nNext: python thesis/scripts/collect_ocr.py "
          "(link existing OCR outputs to these pages)")


if __name__ == "__main__":
    main()
