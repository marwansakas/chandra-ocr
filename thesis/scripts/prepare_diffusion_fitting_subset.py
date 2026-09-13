"""Freeze a varied selection from existing TRAIN pairs; never changes the split."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis.diffusion.data import file_hash, json_bytes, load_frozen
from thesis.diffusion.subsets import make_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    partitions, lock, _ = load_frozen(args.split_root, args.dataset_root)
    manifest = make_subset(partitions["train"], args.samples, args.seed, lock["dataset_fingerprint"],
                           file_hash(args.split_root / "split_lock.json"))
    content = json_bytes(manifest)
    if args.output.exists():
        if args.output.read_bytes() != content:
            raise ValueError("Existing subset differs; choose a new filename, never overwrite it")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(content)
    print(json.dumps({"subset": str(args.output), "sha256": file_hash(args.output), **manifest["stats"]}, indent=2))


if __name__ == "__main__":
    main()
