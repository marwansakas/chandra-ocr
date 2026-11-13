"""python -m thesis.khatt_transfer --help"""
from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

from .data import prepare_khatt, prepare_children, load_prepared


def model_options(parser):
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--max-width", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--init-from", type=Path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="KHATT-only MDiff pretraining, then children-only fine-tuning")
    sub = parser.add_subparsers(dest="action", required=True)
    assets = sub.add_parser("setup-assets", help="Download and verify the pinned official visual encoder")
    assets.add_argument("--output", type=Path, required=True)
    khatt = sub.add_parser("prepare-khatt", help="Download pinned HF train/validation; restore this mirror's reversed labels once")
    khatt.add_argument("--output", type=Path, required=True)
    khatt.add_argument("--revision", help="HF dataset revision; default resolves current commit and records it")
    khatt.add_argument("--hf-root", type=Path, help="Reuse previously saved khatt_hf_train/ and khatt_hf_validation/ directories")
    khatt.add_argument("--inventory-file", type=Path, help="Explicit extra characters, declared before pretraining")
    children = sub.add_parser("prepare-children", help="Freeze copied children data with the parent KHATT tokenizer")
    children.add_argument("--dataset-root", type=Path, required=True)
    children.add_argument("--init-from", type=Path, required=True)
    children.add_argument("--output", type=Path, required=True)
    children.add_argument("--assignments", type=Path, help="Optional exact JSON {sample_id: train/validation/test}; preserve an existing three-way split")
    grouping = children.add_mutually_exclusive_group(required=True)
    grouping.add_argument("--writer-map", type=Path, help="JSON {sample_folder_id: authoritative_child_id}")
    grouping.add_argument("--split-strategy", choices=("notebook", "line"))
    children.add_argument("--validation-ratio", type=float, default=.1)
    children.add_argument("--test-ratio", type=float, default=.2)
    children.add_argument("--seed", type=int, default=54101)
    check = sub.add_parser("check", help="Verify prepared train/validation and preflight dimensions/vocabulary; no training")
    check.add_argument("--prepared", type=Path, required=True)
    model_options(check)
    train = sub.add_parser("train", help="New KHATT or children stage; never runs test evaluation")
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    model_options(train)
    train.add_argument("--encoder-checkpoint", type=Path)
    train.add_argument("--random-init", action="store_true", help="Explicitly train without official visual weights")
    train.add_argument("--resume", action="store_true", help="Resume same --output, data, environment and training options")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--accumulation", type=int, default=16)
    train.add_argument("--decoder-lr", type=float, help="Default KHATT 3e-4; children 3e-5")
    train.add_argument("--encoder-lr", type=float, help="Default KHATT 1e-5; children 1e-6")
    train.add_argument("--freeze-epochs", type=int, help="Default KHATT 5; children 0")
    train.add_argument("--warmup-epochs", type=int, default=5)
    train.add_argument("--weight-decay", type=float, default=.01)
    train.add_argument("--steps", type=int, default=3, help="MDiff rounds, two decoder passes each")
    evaluation = sub.add_parser("evaluate", help="Explicit validation or reserved test evaluation; never called by train")
    evaluation.add_argument("--prepared", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--partition", choices=("validation", "test"), required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    for item in (train, evaluation):
        item.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        item.add_argument("--cpu-threads", type=int, default=2)
        item.add_argument("--seed", type=int, default=54101)
    args = parser.parse_args(argv)
    for name in ("capacity", "max_width", "dim", "layers", "epochs", "patience", "batch_size", "accumulation", "warmup_epochs", "steps", "cpu_threads"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    if getattr(args, "dim", None) and args.dim % 8:
        parser.error("--dim must be a multiple of eight attention heads")
    if getattr(args, "max_width", None) and args.max_width % 4:
        parser.error("--max-width must be a multiple of four")
    if getattr(args, "dropout", None) is not None and not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if getattr(args, "freeze_epochs", None) is not None and args.freeze_epochs < 0:
        parser.error("--freeze-epochs must be nonnegative")
    for name in ("decoder_lr", "encoder_lr"):
        if getattr(args, name, None) is not None and getattr(args, name) <= 0:
            parser.error(name + " must be positive")
    if getattr(args, "weight_decay", 0) < 0:
        parser.error("--weight-decay must be nonnegative")
    if getattr(args, "random_init", False) and getattr(args, "encoder_checkpoint", None):
        parser.error("Choose --encoder-checkpoint or --random-init")
    return args


def setup_assets(output):
    from thesis.diffusion.data import file_hash
    from thesis.mdiff_arabic_v1.artifacts import ASSETS, HF_REVISION
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        target = output / name
        if not target.exists():
            url = f"https://huggingface.co/topdu/MDiff4STR/resolve/{HF_REVISION}/mdiff4str_small/{name}"
            temporary = target.with_suffix(target.suffix + ".download")
            print("Downloading", name, flush=True)
            urllib.request.urlretrieve(url, temporary)
            if file_hash(temporary) != expected:
                raise ValueError("Downloaded encoder asset hash mismatch: " + name)
            temporary.replace(target)
        if file_hash(target) != expected:
            raise ValueError("Encoder asset hash mismatch: " + name)
        print("Verified", target)


def main(argv=None):
    args = parse_args(argv)
    if args.action == "setup-assets":
        setup_assets(args.output)
    elif args.action == "prepare-khatt":
        prepare_khatt(args)
    elif args.action == "prepare-children":
        prepare_children(args)
    elif args.action == "check":
        from .training import build_config, check_records, load_checkpoint
        metadata, tokenizer, parts = load_prepared(args.prepared)
        parent = load_checkpoint(args.init_from) if args.init_from else None
        if metadata["stage"] == "children" and parent is None:
            raise ValueError("Children preflight requires --init-from KHATT_RUN/best.pt")
        config = build_config(args, tokenizer, parts, parent)
        for rows in parts.values():
            check_records(rows, tokenizer, config)
        print("Verified train/validation:", metadata["counts"], "configuration:", config)
    elif args.action == "train":
        from .training import train
        train(args)
    else:
        from .training import evaluate
        evaluate(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(f"ERROR: {error}") from None
