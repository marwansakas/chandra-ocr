"""Reusable inference implementation for local TrOCR-compatible models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from line_runner_common import DEFAULT_INPUT, ROOT, find_line_images, prediction_dir, select_batch, write_prediction, write_run_files


def run(default_model: Path, default_output: Path, model_name: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run {model_name} on Surya line images.")
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.max_length < 1:
        raise SystemExit("--max-length must be at least 1")

    images = find_line_images(args.input_root)
    selected = select_batch(images, args.start_line, args.limit)
    if args.dry_run:
        for position, image in selected:
            print(f"{position:06d}: {image.relative_to(args.input_root)}")
        return
    if not args.model.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model}")

    try:
        import torch
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise SystemExit(f"Model dependencies are missing: {exc}") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()

    rows: list[dict] = []
    skipped = 0
    started = time.monotonic()
    for offset in range(0, len(selected), args.batch_size):
        batch = selected[offset : offset + args.batch_size]
        pending = []
        for position, image_path in batch:
            destination = prediction_dir(args.output_root, args.input_root, image_path)
            if (destination / "prediction.txt").is_file() and not args.force:
                skipped += 1
            else:
                pending.append((position, image_path, destination))
        if not pending:
            continue
        pil_images = [Image.open(image).convert("RGB") for _, image, _ in pending]
        pixels = processor(images=pil_images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated = model.generate(pixels, max_length=args.max_length)
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        for (position, image_path, destination), text in zip(pending, texts):
            row = {
                "line_position": position,
                "image_path": str(image_path.relative_to(ROOT)),
                "prediction_path": str((destination / "prediction.txt").relative_to(ROOT)),
                "model": model_name,
                "model_path": str(args.model),
            }
            write_prediction(destination, text, row)
            rows.append(row)
            print(f"[{position}] {image_path.parent.name}: {text.strip()}", flush=True)

    summary = {
        "model": model_name,
        "selected": len(selected),
        "completed": len(rows),
        "skipped": skipped,
        "device": str(device),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_run_files(args.output_root, rows, summary)
    print(json.dumps(summary, indent=2), flush=True)
