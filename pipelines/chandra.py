"""Run Chandra on Surya line images and update the line dataset safely."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "thesis" / "labels" / "surya_line_ground_truth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chandra", default=shutil.which("chandra") or "chandra")
    parser.add_argument("--method", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def markdown_for(output_root: Path, stem: str) -> Path | None:
    candidates = (
        output_root / stem / f"{stem}.md",
        output_root / f"{stem}.md",
    )
    matches = [path for path in candidates if path.is_file()]
    return matches[0] if len(matches) == 1 else None


def write_page(page_dir: Path, rows: list[dict]) -> None:
    lines_path = page_dir / "lines.jsonl"
    with lines_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    text = "\n".join(str(row.get("text") or "").strip() for row in rows)
    (page_dir / "ground_truth.txt").write_text(text + ("\n" if rows else ""), encoding="utf-8")


def run_page(page_dir: Path, args: argparse.Namespace) -> bool:
    lines_path = page_dir / "lines.jsonl"
    rows = [
        json.loads(line)
        for line in lines_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pending = [
        row
        for row in rows
        if args.force or row.get("status") != "ocr_complete"
    ]
    if not pending:
        return False

    image_dir = page_dir / "line_images"
    output_dir = page_dir / "chandra_line_ocr"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.chandra),
        str(image_dir),
        str(output_dir),
        "--method",
        args.method,
        "--no-images",
        "--batch-size",
        str(args.batch_size),
        "--max-output-tokens",
        str(args.max_output_tokens),
    ]
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, check=True, env=env)

    missing: list[str] = []
    for row in rows:
        image = ROOT / row["line_image"]
        result = markdown_for(output_dir, image.stem)
        if result is None:
            if row in pending:
                row["status"] = "ocr_missing"
                missing.append(image.name)
            continue
        row["text"] = result.read_text(encoding="utf-8", errors="replace").strip()
        row["status"] = "ocr_complete"

    write_page(page_dir, rows)
    if missing:
        raise RuntimeError(f"Chandra produced no unique output for: {', '.join(missing)}")
    return True


def main() -> None:
    args = parse_args()
    pages = sorted(path.parent for path in args.dataset_root.glob("*/lines.jsonl"))
    if args.limit is not None:
        pages = pages[: args.limit]
    completed = failed = 0
    for index, page in enumerate(pages, 1):
        print(f"[{index}/{len(pages)}] {page.name}", flush=True)
        try:
            completed += int(run_page(page, args))
        except Exception as exc:
            failed += 1
            print(f"ERROR: {exc}", flush=True)
            if not args.continue_on_error:
                raise
    print(json.dumps({"pages": len(pages), "completed": completed, "failed": failed}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

