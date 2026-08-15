"""Run a local Invizo inference command on Surya line images."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from line_runner_common import DEFAULT_INPUT, ROOT, find_line_images, prediction_dir, select_batch, write_prediction, write_run_files


def normalize_text(raw_output: Path) -> str:
    text_files = sorted((*raw_output.rglob("*.txt"), *raw_output.rglob("*.md")))
    if len(text_files) == 1:
        return text_files[0].read_text(encoding="utf-8", errors="replace").strip()
    json_files = sorted(raw_output.rglob("*.json"))
    if len(json_files) == 1:
        payload = json.loads(json_files[0].read_text(encoding="utf-8"))
        for key in ("text", "paragraph", "transcription", "prediction"):
            if isinstance(payload, dict) and isinstance(payload.get(key), str):
                return payload[key].strip()
    raise RuntimeError("Invizo must produce one TXT/Markdown file or a supported JSON text field")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Invizo on Surya line images.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "thesis" / "predictions" / "invizo")
    parser.add_argument("--command", required=True, help="Command template containing {input} and {output}.")
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if "{input}" not in args.command or "{output}" not in args.command:
        raise SystemExit("--command must contain {input} and {output}")

    selected = select_batch(find_line_images(args.input_root), args.start_line, args.limit)
    if args.dry_run:
        for position, image in selected:
            print(f"{position:06d}: {image.relative_to(args.input_root)}")
        return

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
        command = args.command.format(input=str(image_path), output=str(raw_output))
        try:
            subprocess.run(shlex.split(command, posix=os.name != "nt"), check=True)
            text = normalize_text(raw_output)
            row = {
                "line_position": position,
                "image_path": str(image_path.relative_to(ROOT)),
                "prediction_path": str(prediction_path.relative_to(ROOT)),
                "model": "invizo",
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
        "model": "invizo", "selected": len(selected), "completed": len(rows),
        "skipped": skipped, "failed": failed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_run_files(args.output_root, rows, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)
