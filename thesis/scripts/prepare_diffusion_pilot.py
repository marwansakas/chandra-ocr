"""Verify the first-100-page Drive download and freeze an individual-line 90/10 pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis.diffusion.data import audit_dataset, freeze_line_pilot, json_bytes, load_frozen, resolve_relative


def page_name(sample_id):
    match = re.fullmatch(r"\d+_(.+)_line_\d+", sample_id)
    if not match:
        raise ValueError(f"Unexpected clean ground-truth sample name: {sample_id}")
    return match.group(1)


def prepare(download_root, split_root, stats_dir, *, train_ratio=0.9, seed=42, capacity=None):
    inventory = json.loads((download_root / "drive_inventory.json").read_text(encoding="utf-8"))
    folders = {f["id"]: f for f in inventory["ground_folders"]}
    selected_names = {page["title"] for page in inventory["selection"]}
    if len(folders) != len(inventory["ground_folders"]):
        raise ValueError("Duplicate ground-truth folder ID in inventory")
    if any(page_name(folder["title"]) not in selected_names for folder in folders.values()):
        raise ValueError("A ground-truth sample belongs to a page outside the selected pages")
    dataset = download_root / "ground_truth_lines"
    expected = set()
    file_keys = set()
    for file in inventory["images"] + inventory["labels"]:
        parents = [parent for parent in file["parent_ids"] if parent in folders]
        if len(parents) != 1:
            raise ValueError(f"Ambiguous ground-truth file parent: {file['id']}")
        folder = folders[parents[0]]["title"]
        key = (parents[0], file["title"])
        if file["title"] not in ("image.png", "ground_truth.txt") or key in file_keys:
            raise ValueError("Unexpected or duplicate file in ground-truth inventory")
        file_keys.add(key)
        path = resolve_relative(dataset, folder + "/" + file["title"])
        expected.add(folder)
        if not path.is_file() or path.stat().st_size != int(file["size"]):
            raise ValueError(f"Download incomplete or changed size: {folder}/{file['title']}")
    if len(expected) != len(folders) or len(file_keys) != 2 * len(folders):
        raise ValueError("Not every selected ground-truth folder has one inventoried image/text pair")
    for folder in folders.values():
        for name in ("image.png", "ground_truth.txt"):
            if not resolve_relative(dataset, folder["title"] + "/" + name).is_file():
                raise ValueError(f"Missing required file: {folder['title']}/{name}")
    actual = {path.name for path in dataset.iterdir() if path.is_dir()}
    if actual != expected:
        raise ValueError("Local sample directories differ from the frozen first-100-page Drive selection")
    audit = audit_dataset(dataset.resolve())
    fatal = [issue for issue in audit["issues"] if issue["reason"] != "empty label" and not issue["reason"].startswith("label contains ignore")]
    if fatal:
        raise ValueError("Invalid downloaded samples: " + json.dumps(fatal[:10], ensure_ascii=False))
    if stats_dir.exists():
        raise FileExistsError(f"Stats output exists; choose a new directory: {stats_dir}")
    lock = freeze_line_pilot(audit, split_root, train_ratio=train_ratio, seed=seed, capacity=capacity)
    partitions, _, tokenizer = load_frozen(split_root, dataset)
    selected = inventory["selection"]
    surya_counts = Counter(parent for line in inventory["surya_line_folders"] for parent in line["parent_ids"])
    ground_counts = Counter(page_name(folder["title"]) for folder in folders.values())
    usable_counts = Counter(page_name(record["sample_id"]) for record in audit["records"])
    train_counts = Counter(page_name(record["sample_id"]) for record in partitions["train"])
    test_counts = Counter(page_name(record["sample_id"]) for record in partitions["test"])
    rows = [{"page_rank": index, "page_id": page["title"], "drive_page_id": page["id"],
             "surya_detected_lines": surya_counts[page["id"]], "ground_truth_line_folders": ground_counts[page["title"]],
             "usable_labeled_lines": usable_counts[page["title"]], "training_lines": train_counts[page["title"]],
             "test_lines": test_counts[page["title"]]} for index, page in enumerate(selected, 1)]
    training_widths_64 = [max(1, round(r["width"] * 64 / r["height"])) for r in partitions["train"]]
    report = {"selected_pages": len(selected), "page_order": "numeric ascending folder prefix",
              "first_page": selected[0]["title"], "last_page": selected[-1]["title"],
              "surya_detected_line_folders": len(inventory["surya_line_folders"]),
              "ground_truth_line_folders": len(folders), "usable_labeled_lines": len(audit["records"]),
              "excluded_empty_or_ignore": audit["issues"], "training_lines": lock["counts"]["train"], "test_lines": lock["counts"]["test"],
              "actual_training_fraction": lock["counts"]["train"] / len(audit["records"]),
              "actual_test_fraction": lock["counts"]["test"] / len(audit["records"]),
              "seed": seed, "rounding": lock["rounding"], "capacity_including_eos": lock["capacity"],
              "vocabulary_size_with_specials": len(tokenizer), "literal_characters_observed": len(audit["characters"]),
              "lengths_codepoints": audit["lengths_codepoints"], "image_widths": audit["widths"], "image_heights": audit["heights"],
              "labels_with_internal_line_breaks": sum("\n" in r["text"] or "\r" in r["text"] for r in audit["records"]),
              "training_max_width_at_height_64_rounded_to_patch_8": math.ceil(max(training_widths_64) / 8) * 8,
              "duplicate_image_groups": audit["duplicates"],
              "cross_split_duplicate_pixel_hashes": lock["cross_split_duplicate_pixel_hashes"],
              "pages_in_both_partitions": len(set(train_counts) & set(test_counts)),
              "test_oov_lines": sum(bool(set(r["text"]) - set(tokenizer.characters)) for r in partitions["test"]),
              "test_capacity_overflow": sum(len(r["text"]) + 1 > lock["capacity"] for r in partitions["test"]),
              "dataset_fingerprint": lock["dataset_fingerprint"], "dataset_root": str(dataset.resolve()), "split_root": str(split_root.resolve()),
              "evaluation_policy": lock["evaluation_policy"], "writer_isolation": False,
              "downloaded_source_bytes": sum(int(f["size"]) for f in inventory["images"] + inventory["labels"])}
    stats_dir.mkdir(parents=True, exist_ok=False)
    (stats_dir / "data_stats.json").write_bytes(json_bytes(report))
    (stats_dir / "audit.json").write_bytes(json_bytes(audit))
    with (stats_dir / "pages.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (stats_dir / "image_text_pairs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["partition", "sample_id", "image_path", "label_path", "text"])
        writer.writeheader()
        for partition in ("train", "test"):
            for record in partitions[partition]:
                writer.writerow({"partition": partition, **{key: record[key] for key in ("sample_id", "image_path", "label_path", "text")}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--stats-dir", type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, help="Includes EOS; defaults to training maximum plus EOS")
    args = parser.parse_args()
    result = prepare(args.download_root, args.split_root or args.download_root / "splits_90_10",
                     args.stats_dir or args.download_root / "stats", train_ratio=args.train_ratio, seed=args.seed, capacity=args.capacity)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
