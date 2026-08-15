"""Run Chandra inference on Surya line images without reading ground truth."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from line_runner_common import DEFAULT_INPUT, ROOT, find_line_images, prediction_dir, select_batch, write_prediction, write_run_files


def chandra_text(raw_output: Path, image_path: Path) -> str:
    stem = image_path.stem
    candidates = (raw_output / stem / f"{stem}.md", raw_output / f"{stem}.md")
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Chandra Markdown result for {image_path}, found {len(matches)}")
    return matches[0].read_text(encoding="utf-8", errors="replace").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chandra on Surya line images.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "thesis" / "predictions" / "chandra")
    parser.add_argument("--chandra", default=shutil.which("chandra") or "chandra")
    parser.add_argument("--method", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = select_batch(find_line_images(args.input_root), args.start_line, args.limit)
    if args.dry_run:
        for position, image in selected:
            print(f"{position:06d}: {image.relative_to(args.input_root)}")
        return

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rows: list[dict] = []
    skipped = failed = 0
    started = time.monotonic()
    for batch_index, (position, image_path) in enumerate(selected, 1):
        destination = prediction_dir(args.output_root, args.input_root, image_path)
        prediction_path = destination / "prediction.txt"
        if prediction_path.is_file() and not args.force:
            skipped += 1
            continue
        raw_output = destination / "raw"
        if raw_output.exists():
            shutil.rmtree(raw_output)
        raw_output.mkdir(parents=True)
        command = [
            str(args.chandra), str(image_path), str(raw_output),
            "--method", args.method, "--no-images",
            "--max-output-tokens", str(args.max_output_tokens),
        ]
        try:
            subprocess.run(command, check=True, env=env)
            text = chandra_text(raw_output, image_path)
            row = {
                "line_position": position,
                "image_path": str(image_path.relative_to(ROOT)),
                "prediction_path": str(prediction_path.relative_to(ROOT)),
                "model": "chandra",
            }
            write_prediction(destination, text, row)
            rows.append(row)
            print(f"[{batch_index}/{len(selected)}] {image_path.parent.name}: {text}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"ERROR {image_path}: {exc}", flush=True)
            if not args.continue_on_error:
                raise

    summary = {
        "model": "chandra", "selected": len(selected), "completed": len(rows),
        "skipped": skipped, "failed": failed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_run_files(args.output_root, rows, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)
