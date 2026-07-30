"""
Create a balanced manual-labeling batch for OCR ground truth.

The script samples pages that already have OCR output, then creates:
  thesis/labels/ground_truth_manifest.csv
  thesis/labels/ground_truth/<page_id>.txt

Fill each .txt file with the correct transcription of the child handwriting.

Run:
  python thesis/scripts/sample_ground_truth.py
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--overwrite-empty-files",
        action="store_true",
        help="Recreate empty ground-truth files even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = common.load_config()
    size = args.size or int(cfg.get("sample_size", 30))
    seed = args.seed if args.seed is not None else int(cfg.get("random_seed", 42))
    random.seed(seed)

    metadata_dir = common.resolve(cfg["metadata_dir"])
    labels_dir = common.resolve(cfg["labels_dir"])
    gt_root = common.resolve(cfg["ground_truth_dir"])
    pages_path = metadata_dir / "pages.jsonl"

    records = [r for r in common.read_jsonl(pages_path) if r.get("ocr_text_present")]
    if not records:
        raise SystemExit("No OCR text linked. Run collect_ocr.py first.")

    # Balance by place x grade x gender where possible.
    buckets = defaultdict(list)
    for record in records:
        key = (
            record.get("place") or "unknown",
            record.get("grade") or "unknown",
            record.get("gender") or "unknown",
        )
        buckets[key].append(record)

    selected = []
    bucket_items = sorted(buckets.items(), key=lambda item: tuple(map(str, item[0])))
    while len(selected) < min(size, len(records)):
        progressed = False
        for _, rows in bucket_items:
            remaining = [r for r in rows if r not in selected]
            if not remaining:
                continue
            selected.append(random.choice(remaining))
            progressed = True
            if len(selected) >= min(size, len(records)):
                break
        if not progressed:
            break

    manifest_rows = []
    for record in selected:
        gt_path = common.ground_truth_path(gt_root, record)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite_empty_files or not gt_path.exists():
            gt_path.write_text("", encoding="utf-8")
        manifest_rows.append(
            {
                "page_id": record["page_id"],
                "student_id": record["student_id"],
                "place": record["place"],
                "grade": record.get("grade") or "unknown",
                "gender": record.get("gender") or "unknown",
                "image_path": record["image_path"],
                "ocr_output_path": record["ocr_output_path"],
                "ground_truth_path": str(gt_path.relative_to(common.PROJECT_ROOT)),
                "status": "empty" if not gt_path.read_text(encoding="utf-8").strip() else "filled",
            }
        )

    manifest_path = labels_dir / "ground_truth_manifest.csv"
    common.write_csv(manifest_path, manifest_rows)

    print(f"Selected {len(selected)} pages for labeling.")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Ground truth files are under: {gt_root}")
    print("\nAfter filling labels: python thesis/scripts/evaluate_ocr.py")


if __name__ == "__main__":
    main()
