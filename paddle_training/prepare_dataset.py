"""Create PaddleOCR recognition manifests from image.png/ground_truth.txt samples.

The source dataset is never modified.  A group is kept wholly in one split; use
--group-key with a writer/student field in metadata.json for a writer-disjoint
evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


def clean_label(value: str) -> str:
    value = unicodedata.normalize("NFC", value.strip())
    if not value or value.casefold() == "ignore":
        raise ValueError("label is blank or marked ignore")
    if "\n" in value or "\r" in value or "\t" in value:
        raise ValueError("label must be exactly one line and contain no tab")
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise ValueError("label contains a control character")
    return value


def paddle_arabic_label(value: str, input_order: str) -> str:
    """Return PP-OCR Arabic's visual-order training label.

    The upstream Arabic decoder reverses its visual CTC stream back to logical
    reading order.  Ground-truth files in this project are logical Unicode, so
    they need one display-order conversion for training.  Do not use this for
    a dataset that has already stored display-order strings.
    """
    if input_order == "visual":
        return value
    try:
        from bidi.algorithm import get_display
    except ImportError as error:
        raise ValueError("Install python-bidi: pip install 'python-bidi>=0.6,<0.7'") from error
    return get_display(value, base_dir="R")


def metadata_value(metadata: dict, key: str):
    value = metadata
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def split_name(group: str, seed: int, train: float, validation: float) -> str:
    # Hashing is deterministic across machines and independent of filesystem order.
    bucket = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "val"
    return "test"


def collect(root: Path, group_key: str | None, label_order: str):
    rows, skipped = [], []
    for image in sorted(root.rglob("image.png"), key=lambda item: item.relative_to(root).as_posix()):
        folder = image.parent
        sample_id = folder.relative_to(root).as_posix()
        label_path = folder / "ground_truth.txt"
        if not label_path.is_file():
            skipped.append({"sample_id": sample_id, "reason": "missing_ground_truth"})
            continue
        try:
            text = paddle_arabic_label(clean_label(label_path.read_text(encoding="utf-8-sig")), label_order)
        except (OSError, UnicodeError, ValueError) as error:
            skipped.append({"sample_id": sample_id, "reason": str(error)})
            continue
        metadata = {}
        metadata_path = folder / "metadata.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, UnicodeError) as error:
                raise ValueError(f"Invalid metadata for {sample_id}: {error}") from error
        group = metadata_value(metadata, group_key) if group_key else sample_id
        if group is None or not str(group).strip():
            raise ValueError(f"{sample_id} has no nonempty metadata key {group_key!r}")
        rows.append({"sample_id": sample_id, "image": image, "text": text, "group": str(group)})
    if not rows:
        raise ValueError("No usable image.png + ground_truth.txt samples found")
    return rows, skipped


def write_manifests(rows: list[dict], root: Path, output: Path, seed: int, train: float, validation: float):
    output.mkdir(parents=True, exist_ok=False)
    lists = {name: [] for name in ("train", "val", "test")}
    for row in rows:
        row["split"] = split_name(row["group"], seed, train, validation)
        # PaddleOCR joins data_dir to this path. POSIX separators also work on Linux.
        relative = row["image"].relative_to(root).as_posix()
        lists[row["split"]].append(f"{relative}\t{row['text']}\n")
    for name, lines in lists.items():
        if not lines:
            raise ValueError(f"The hash split left {name} empty; adjust ratios/seed or add data")
        (output / f"{name}_list.txt").write_text("".join(lines), encoding="utf-8")
    return lists


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-key", help="Dot-path in metadata.json, e.g. student_id. Required for writer-disjoint splits.")
    parser.add_argument("--label-order", choices=("logical", "visual"), default="logical",
                        help="Order used by ground_truth.txt; logical is converted for Arabic PP-OCRv5 training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=.80)
    parser.add_argument("--val-ratio", type=float, default=.10)
    args = parser.parse_args(argv)
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1 or args.train_ratio + args.val_ratio >= 1:
        parser.error("ratios must be positive and sum to less than 1")
    root, output = args.dataset.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error(f"dataset does not exist: {root}")
    if output.exists():
        parser.error(f"output must not exist: {output}")
    rows, skipped = collect(root, args.group_key, args.label_order)
    lists = write_manifests(rows, root, output, args.seed, args.train_ratio, args.val_ratio)
    summary = {
        "dataset": str(root), "group_key": args.group_key, "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": 1 - args.train_ratio - args.val_ratio},
        "counts": {name: len(items) for name, items in lists.items()},
        "groups": {name: len({row["group"] for row in rows if row["split"] == name}) for name in lists},
        "skipped": skipped,
        "source_label_order": args.label_order,
        "training_label_order": "visual (required by Arabic PP-OCRv5's reverse decoder)",
    }
    (output / "split.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
