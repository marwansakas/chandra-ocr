"""Save deterministic TRAIN-only before/after augmentation previews for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW preview directory")
    parser.add_argument("--policy", choices=("mild_v1", "trocr_views_v1", "trocr_all_views_v1"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    from PIL import Image
    from thesis.diffusion.data import file_hash, load_frozen, resolve_relative
    from thesis.diffusion.images import AugmentationConfig, augmentation_seed, augmentation_view_index, augment_image

    records, lock, _ = load_frozen(args.split_root, args.dataset_root)
    selected = sorted(records["train"], key=lambda row: row["sample_id"])[:args.count]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, record in enumerate(selected):
        with Image.open(resolve_relative(args.dataset_root, record["image_path"])) as image:
            original = image.convert("RGB")
        original.save(args.output_dir / f"{index:03d}_original.png")
        views = []
        if args.policy == "trocr_all_views_v1":
            preview_schedule = ((1, 0), (1, 1), (1, 2))
        elif args.policy == "trocr_views_v1":
            preview_schedule = tuple((epoch, augmentation_view_index(epoch)) for epoch in (1, 2, 3))
        else:
            preview_schedule = ((1, None),)
        for epoch, view_index in preview_schedule:
            seed = augmentation_seed(record["sample_id"], epoch, args.seed)
            augmented, details = augment_image(original, AugmentationConfig(args.policy), seed, view_index=view_index)
            view_name = details.get("view", "augmented")
            if view_name != "original":
                augmented.save(args.output_dir / f"{index:03d}_{view_name}.png")
            views.append({"epoch_in_cycle": epoch, "size": augmented.size, "details": details})
        rows.append({"sample_id": record["sample_id"], "text": record["text"], "original_size": original.size,
                     "views": views})
    report = {"partition": "train", "dataset_fingerprint": lock["dataset_fingerprint"],
              "split_lock_sha256": file_hash(args.split_root / "split_lock.json"), "policy": AugmentationConfig(args.policy).as_dict(),
              "preview_schedule": [{"epoch": epoch, "view_index": view_index} for epoch, view_index in preview_schedule],
              "seed": args.seed, "samples": rows,
              "inspection_note": "Review dots, hamza/madda, tatweel, edges and visible crop content before training. Targets are unchanged."}
    (args.output_dir / "preview.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "samples": len(rows), "partition": "train"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
