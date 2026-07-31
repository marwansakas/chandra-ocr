from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
RANKED_CSV = PROJECT_ROOT / "pdf_quality_scores.csv"
RANKED_DIR = PROJECT_ROOT / "ranked_pdfs"
OUTPUT_ROOT = PROJECT_ROOT / "thesis" / "labels" / "surya_line_ground_truth"
CHANDRA_ENV = Path("/home/sakas/.conda/envs/chandra")
CHANDRA_METHOD = "hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build OCR-derived line-level ground-truth files from the ranked PDFs "
            "using Surya text-line detection and Chandra OCR."
        )
    )
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--ranked-csv", type=Path, default=RANKED_CSV)
    parser.add_argument("--ranked-dir", type=Path, default=RANKED_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--chandra", type=Path, default=None)
    parser.add_argument("--chandra-method", default=CHANDRA_METHOD)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--chandra-batch-size", type=int, default=1)
    parser.add_argument("--render-scale", type=float, default=2.0)
    parser.add_argument("--crop-margin-px", type=int, default=8)
    parser.add_argument(
        "--per-line-timeout",
        type=int,
        default=None,
        help="Maximum seconds to allow one page-level Chandra subprocess to run.",
    )
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_chandra(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise SystemExit(f"Chandra executable does not exist: {explicit}")

    found = shutil.which("chandra")
    if found:
        return Path(found)

    candidate = CHANDRA_ENV / "bin" / "chandra"
    if candidate.exists():
        return candidate

    raise SystemExit(
        "Could not find `chandra`. Activate the Chandra environment or pass --chandra."
    )


def read_ranked_inputs(csv_path: Path, ranked_dir: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ranked_path = ranked_dir / row["ranked_name"]
            source_path = ranked_path if ranked_path.exists() else Path(row["original_path"])
            records.append(
                {
                    "rank": int(row["rank"]),
                    "ranked_name": row["ranked_name"],
                    "original_name": row["original_name"],
                    "original_path": row["original_path"],
                    "source_path": source_path,
                }
            )

    records.sort(key=lambda item: item["rank"])
    return records[:limit]


def safe_stem(path: Path) -> str:
    return path.stem.replace("/", "__").replace("\\", "__")


def render_first_page(input_path: Path, scale: float) -> Any:
    from PIL import Image

    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(input_path))
        if len(doc) == 0:
            raise ValueError(f"PDF has no pages: {input_path}")
        page = doc[0]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil().convert("RGB")
        page.close()
        doc.close()
        return image

    return Image.open(input_path).convert("RGB")


def bbox_from_polygon_box(box: Any, width: int, height: int, margin: int) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = box.bbox
    left = max(0, int(x_min) - margin)
    top = max(0, int(y_min) - margin)
    right = min(width, int(x_max) + margin)
    bottom = min(height, int(y_max) + margin)
    return left, top, right, bottom


def sort_line_boxes(boxes: list[Any]) -> list[Any]:
    return sorted(boxes, key=lambda box: (box.bbox[1], box.bbox[0]))


def output_already_done(page_dir: Path) -> bool:
    ground_truth_path = page_dir / "ground_truth.txt"
    lines_path = page_dir / "lines.jsonl"
    if not ground_truth_path.exists() or not lines_path.exists():
        return False

    with lines_path.open(encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    if not lines:
        return False

    return all(line.get("status") == "ocr_complete" for line in lines)


def run_streaming(
    command: list[str],
    env: dict[str, str],
    timeout: int | None,
    heartbeat_seconds: int,
) -> int:
    start = time.monotonic()
    next_heartbeat = start + max(1, heartbeat_seconds)
    process = subprocess.Popen(command, env=env, start_new_session=True)

    while True:
        return_code = process.poll()
        if return_code is not None:
            return return_code

        now = time.monotonic()
        elapsed = int(now - start)
        if timeout is not None and elapsed >= timeout:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise subprocess.TimeoutExpired(command, timeout)

        if heartbeat_seconds and now >= next_heartbeat:
            print(f"Still processing line OCR... elapsed={elapsed}s", flush=True)
            next_heartbeat = now + heartbeat_seconds

        time.sleep(2)


def find_chandra_markdown(output_dir: Path, input_file: Path) -> Path | None:
    expected = output_dir / input_file.stem / f"{input_file.stem}.md"
    if expected.exists():
        return expected

    expected_flat = output_dir / f"{input_file.stem}.md"
    if expected_flat.exists():
        return expected_flat

    # Never fall back to an unrelated Markdown file. Doing so silently assigns
    # another line's transcription to this image and corrupts the dataset.
    matches = [
        path
        for path in output_dir.rglob("*.md")
        if path.stem == input_file.stem
    ]
    return matches[0] if len(matches) == 1 else None


def run_chandra_folder(
    chandra: Path,
    method: str,
    max_output_tokens: int,
    batch_size: int,
    input_dir: Path,
    output_dir: Path,
    env: dict[str, str],
    timeout: int | None,
    heartbeat_seconds: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(chandra),
        str(input_dir),
        str(output_dir),
        "--method",
        method,
        "--no-images",
        "--max-output-tokens",
        str(max_output_tokens),
        "--batch-size",
        str(batch_size),
    ]
    return_code = run_streaming(command, env, timeout, heartbeat_seconds)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def write_page_outputs(page_dir: Path, lines: list[dict[str, Any]]) -> None:
    with (page_dir / "lines.jsonl").open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")

    text = "\n".join(line["text"] for line in lines)
    if text:
        text += "\n"
    (page_dir / "ground_truth.txt").write_text(text, encoding="utf-8")


def process_page(
    record: dict[str, Any],
    predictor: Any,
    args: argparse.Namespace,
    chandra: Path,
    env: dict[str, str],
) -> str:
    page_id = f"{record['rank']:04d}_{safe_stem(record['source_path'])}"
    page_dir = args.output_root / page_id
    already_done = output_already_done(page_dir)
    if already_done and not args.force:
        return "skipped"

    page_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = page_dir / "line_images"
    ocr_dir = page_dir / "chandra_line_ocr"
    if args.force or not already_done:
        shutil.rmtree(crops_dir, ignore_errors=True)
        shutil.rmtree(ocr_dir, ignore_errors=True)
    crops_dir.mkdir(exist_ok=True)
    ocr_dir.mkdir(exist_ok=True)

    page_image = render_first_page(record["source_path"], args.render_scale)
    page_image_path = page_dir / "page.png"
    page_image.save(page_image_path)

    prediction = predictor([page_image])[0]
    boxes = sort_line_boxes(list(prediction.bboxes))

    metadata = {
        "rank": record["rank"],
        "ranked_name": record["ranked_name"],
        "original_name": record["original_name"],
        "original_path": record["original_path"],
        "source_path": str(record["source_path"]),
        "page_image": str(page_image_path.relative_to(PROJECT_ROOT)),
        "image_width": page_image.width,
        "image_height": page_image.height,
        "line_count": len(boxes),
        "surya_image_bbox": prediction.image_bbox,
        "ocr_engine": "chandra",
        "line_detector": "surya-ocr",
        "line_order": "top-to-bottom, then left-to-right for ties",
    }
    (page_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines: list[dict[str, Any]] = []
    line_image_paths: list[Path] = []
    for line_index, box in enumerate(boxes, start=1):
        bbox = bbox_from_polygon_box(box, page_image.width, page_image.height, args.crop_margin_px)
        line_image = page_image.crop(bbox)
        line_image_path = crops_dir / f"line_{line_index:03d}.png"
        line_image.save(line_image_path)
        line_image_paths.append(line_image_path)

        lines.append(
            {
                "line_index": line_index,
                "bbox": list(bbox),
                "polygon": box.polygon,
                "confidence": box.confidence,
                "line_image": str(line_image_path.relative_to(PROJECT_ROOT)),
                "text": "",
                "status": "detected",
            }
        )

    if not args.detect_only and line_image_paths:
        try:
            run_chandra_folder(
                chandra=chandra,
                method=args.chandra_method,
                max_output_tokens=args.max_output_tokens,
                batch_size=args.chandra_batch_size,
                input_dir=crops_dir,
                output_dir=ocr_dir,
                env=env,
                timeout=args.per_line_timeout,
                heartbeat_seconds=args.heartbeat_seconds,
            )
            for line, line_image_path in zip(lines, line_image_paths):
                markdown_path = find_chandra_markdown(ocr_dir, line_image_path)
                if markdown_path is None:
                    line["status"] = "ocr_missing"
                    if not args.continue_on_error:
                        raise FileNotFoundError(
                            f"Chandra did not produce markdown for {line_image_path}"
                        )
                    continue

                line["text"] = markdown_path.read_text(encoding="utf-8").strip()
                line["status"] = "ocr_complete"
        except Exception as exc:
            print(f"Page OCR failed for {crops_dir}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
            for line in lines:
                if line["status"] == "detected":
                    line["status"] = "ocr_failed"

    write_page_outputs(page_dir, lines)
    return "completed"


def main() -> None:
    args = parse_args()
    args.ranked_csv = args.ranked_csv.resolve()
    args.ranked_dir = args.ranked_dir.resolve()
    args.output_root = args.output_root.resolve()

    records = read_ranked_inputs(args.ranked_csv, args.ranked_dir, args.limit)
    print(f"Selected ranked pages: {len(records)}", flush=True)

    if args.dry_run:
        for record in records[:20]:
            print(f"{record['rank']:04d}: {record['source_path']}")
        if len(records) > 20:
            print(f"... {len(records) - 20} more")
        return

    env = os.environ.copy()
    chandra = Path()
    if not args.detect_only:
        chandra = find_chandra(args.chandra)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env["PATH"] = str(chandra.parent) + os.pathsep + env.get("PATH", "")

    args.output_root.mkdir(parents=True, exist_ok=True)
    from surya.detection import DetectionPredictor

    predictor = DetectionPredictor()

    completed = 0
    skipped = 0
    failed = 0
    for index, record in enumerate(records, start=1):
        print(
            f"\n[{index}/{len(records)}] rank={record['rank']:04d} {record['source_path']}",
            flush=True,
        )
        try:
            status = process_page(record, predictor, args, chandra, env)
            if status == "skipped":
                skipped += 1
                print("Skipped existing page output.", flush=True)
            else:
                completed += 1
                print("Completed page output.", flush=True)
        except Exception as exc:
            failed += 1
            print(f"Failed page: {exc}", flush=True)
            if not args.continue_on_error:
                raise

    summary = {
        "selected_pages": len(records),
        "completed_pages": completed,
        "skipped_pages": skipped,
        "failed_pages": failed,
        "output_root": str(args.output_root),
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
