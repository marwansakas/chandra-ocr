"""Audit literal line data, freeze writer-grouped splits, or verify an existing freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis.diffusion.data import audit_dataset, freeze_splits, json_bytes, load_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Audit JSON output; refuses to overwrite")
    parser.add_argument("--writer-map", type=Path, help="JSON object mapping every sample_id to an authoritative writer ID")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--freeze", type=Path, help="Create a NEW immutable split directory")
    action.add_argument("--verify", type=Path, help="Verify frozen files, data content and partition leakage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--capacity", type=int, help="Including EOS; defaults to max train/validation length + 1")
    parser.add_argument("--extra-inventory", type=Path, help="UTF-8 literal character inventory, LF/CR ignored")
    parser.add_argument("--allow-exclusions", action="store_true", help="Accept the explicit audit exclusions when freezing")
    args = parser.parse_args()
    if args.verify:
        _, lock, _ = load_frozen(args.verify, args.dataset_root.resolve())
        print(json.dumps({"verified": True, **lock}, indent=2))
        return
    writer_map = json.loads(args.writer_map.read_text(encoding="utf-8")) if args.writer_map else None
    if writer_map is not None and not isinstance(writer_map, dict):
        parser.error("--writer-map must contain a JSON object")
    audit = audit_dataset(args.dataset_root.resolve(), writer_map)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("xb") as handle:
            handle.write(json_bytes(audit))
    if args.freeze:
        inventory = args.extra_inventory.read_text(encoding="utf-8").replace("\r", "").replace("\n", "") if args.extra_inventory else ""
        lock = freeze_splits(audit, args.freeze, seed=args.seed, train_ratio=args.train_ratio,
                             validation_ratio=args.validation_ratio, capacity=args.capacity,
                             extra_inventory=inventory, allow_exclusions=args.allow_exclusions)
        print(json.dumps(lock, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({k: v for k, v in audit.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
