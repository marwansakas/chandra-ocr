"""Train an experimental CNN-fronted TrOCR line OCR model.

This keeps the regular TrOCR baseline untouched. The added CNN block learns a
small residual image feature transform before the existing Transformer encoder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import common
import train_line_ocr_model as base
from cnn_trocr_model import CNNTrOCRConfig, CNNTrOCRModel, save_cnn_trocr_model


DEFAULT_OUTPUT_DIR = common.PROJECT_ROOT / "thesis" / "models" / "ground_truth_line_cnn_trocr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN-fronted TrOCR on the clean line OCR dataset.")
    parser.add_argument("--dataset-root", type=Path, default=base.DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-root", type=Path, default=base.DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-strategy", choices=("student", "line"), default="student")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-empty-labels", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--khatt-samples", type=int, default=0)
    parser.add_argument("--khatt-cache-root", type=Path, default=base.DEFAULT_KHATT_CACHE_ROOT)
    parser.add_argument("--khatt-dataset-name", default=base.DEFAULT_KHATT_DATASET)
    parser.add_argument("--refresh-khatt-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-label-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--views-per-line", type=int, default=1)
    parser.add_argument("--augmentation-profile", choices=base.AUGMENTATION_PROFILES, default="light")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--cnn-hidden-channels", type=int, default=32)
    parser.add_argument("--cnn-layers", type=int, default=2)
    parser.add_argument("--cnn-kernel-size", type=int, default=3)
    parser.add_argument("--cnn-residual-scale", type=float, default=0.1)
    return parser.parse_args()


def write_experiment_config(
    output_dir: Path,
    args: argparse.Namespace,
    own_train: list[base.LineRecord],
    khatt_train: list[base.LineRecord],
    combined_train: list[base.LineRecord],
    test: list[base.LineRecord],
    khatt_summary: dict,
    cnn_config: CNNTrOCRConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "CNN-TrOCR",
        "base_model": args.model_name,
        "checkpoint": base.portable_path(output_dir),
        "dataset_root": base.portable_path(base.project_path(args.dataset_root)),
        "split_root": base.portable_path(base.project_path(args.split_root)),
        "test_split": base.portable_path(base.project_path(args.split_root) / "test.jsonl"),
        "split_strategy": args.split_strategy,
        "train_ratio": args.train_ratio,
        "random_seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "views_per_line": args.views_per_line,
        "augmentation_profile": args.augmentation_profile,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "cnn_config": cnn_config.__dict__,
        "own_training_samples": len(own_train),
        "khatt_enabled": args.khatt_samples > 0,
        "khatt_samples_requested": args.khatt_samples,
        "khatt_samples_used": len(khatt_train),
        "khatt_dataset_name": args.khatt_dataset_name,
        "khatt_cache_dir": khatt_summary.get("khatt_cache_dir", ""),
        "khatt_cache_reused": khatt_summary.get("khatt_cache_reused", False),
        "khatt_skipped": khatt_summary.get("skipped", {}),
        "combined_training_samples": len(combined_train),
        "test_samples": len(test),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def train_model(train: list[base.LineRecord], test: list[base.LineRecord], args: argparse.Namespace) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise SystemExit(f"Training dependencies are missing: {exc}") from exc

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not available; training on CPU will be slow.", file=sys.stderr)

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    trocr = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    cnn_config = CNNTrOCRConfig(
        hidden_channels=args.cnn_hidden_channels,
        layers=args.cnn_layers,
        kernel_size=args.cnn_kernel_size,
        residual_scale=args.cnn_residual_scale,
    )
    model = CNNTrOCRModel(trocr, cnn_config).to(device)

    trocr.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    trocr.config.pad_token_id = processor.tokenizer.pad_token_id
    trocr.config.vocab_size = trocr.config.decoder.vocab_size
    trocr.config.eos_token_id = processor.tokenizer.sep_token_id
    trocr.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    trocr.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    trocr.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    trocr.generation_config.max_length = args.max_label_length

    train_ds = base.LineOCRDataset(
        train,
        processor,
        args.max_label_length,
        views_per_line=args.views_per_line,
        augment=args.views_per_line > 1,
        augmentation_profile=args.augmentation_profile,
    )
    test_ds = base.LineOCRDataset(test, processor, args.max_label_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=base.collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=base.collate_batch,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as metrics_fh:
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1
                running_loss += loss.item()
                if step % 25 == 0 or step == len(train_loader):
                    print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running_loss / step:.4f}")

            eval_result = base.evaluate_model(model, processor, test_loader, device, args.eval_limit)
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": running_loss / max(1, len(train_loader)),
                **eval_result,
            }
            metrics_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_fh.flush()
            print(json.dumps(row, ensure_ascii=False, indent=2))

            checkpoint = args.output_dir / f"checkpoint-epoch-{epoch}"
            save_cnn_trocr_model(model, checkpoint)
            processor.save_pretrained(checkpoint)

    save_cnn_trocr_model(model, args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved CNN-TrOCR model and processor to {args.output_dir}")


def main() -> None:
    args = parse_args()
    args.dataset_root = base.project_path(args.dataset_root)
    args.split_root = base.project_path(args.split_root)
    args.output_dir = base.project_path(args.output_dir)
    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be between 0 and 1")
    if args.views_per_line < 1:
        raise SystemExit("--views-per-line must be at least 1")
    if args.khatt_samples < 0:
        raise SystemExit("--khatt-samples must be zero or greater")
    if args.cnn_hidden_channels < 1 or args.cnn_layers < 1:
        raise SystemExit("CNN hidden channels and layers must be at least 1")
    if args.cnn_kernel_size < 1 or args.cnn_kernel_size % 2 != 1:
        raise SystemExit("--cnn-kernel-size must be a positive odd number")

    records = base.read_line_records(args.dataset_root)
    if not records:
        raise SystemExit(f"No line records found under {args.dataset_root}")
    usable = [r for r in records if base.keep_for_training(r, args.include_empty_labels)]
    usable.sort(key=lambda r: (r.rank, r.line_index, r.sample_id))
    if args.max_samples is not None:
        usable = usable[: args.max_samples]
    if len(usable) < 2:
        raise SystemExit("Not enough labeled line records to split into train/test.")

    train, test = base.split_records(usable, args.train_ratio, args.seed, args.split_strategy)
    args.split_root.mkdir(parents=True, exist_ok=True)
    base.write_manifest(args.split_root / "train.csv", train)
    base.write_manifest(args.split_root / "test.csv", test)
    base.write_jsonl(args.split_root / "train.jsonl", train)
    base.write_jsonl(args.split_root / "test.jsonl", test)
    base.write_summary(args.split_root / "summary.md", records, usable, train, test, args.split_strategy)

    khatt_records, khatt_summary = base.prepare_khatt_records(args)
    combined_train = base.combine_training_records(train, khatt_records, args.seed)
    if khatt_records:
        base.write_jsonl(args.split_root / "train_with_khatt.jsonl", combined_train)

    print(f"Total line records: {len(records)}")
    print(f"Usable labeled line records: {len(usable)}")
    print(f"Train: {len(train)} lines")
    print(f"Test: {len(test)} lines")
    print(f"Split strategy: {args.split_strategy}")
    print(f"Combined training lines: {len(combined_train)}")

    if args.prepare_only:
        return

    cnn_config = CNNTrOCRConfig(
        hidden_channels=args.cnn_hidden_channels,
        layers=args.cnn_layers,
        kernel_size=args.cnn_kernel_size,
        residual_scale=args.cnn_residual_scale,
    )
    write_experiment_config(args.output_dir, args, train, khatt_records, combined_train, test, khatt_summary, cnn_config)
    train_model(combined_train, test, args)


if __name__ == "__main__":
    main()
