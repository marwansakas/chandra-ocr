"""Build a manually labelable line-image dataset with Surya.

The script recursively finds PDFs under ``ranked_pdfs``, selects a resumable
batch in ranked order, renders the first page of each PDF, and uses Surya to
detect text lines. It never imports or runs Chandra.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "ranked_pdfs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "thesis" / "datasets" / "surya_lines"
DEFAULT_RANKED_INPUT_ROOT = PROJECT_ROOT / "ranked_pdfs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a resumable batch of PDF pages into line images with Surya."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_RANKED_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--start-pdf",
        type=int,
        default=1,
        help="1-based position of the first PDF to process after deterministic sorting.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of PDFs to process. Omit to process every ranked PDF from --start-pdf.",
    )
    parser.add_argument("--render-scale", type=float, default=2.0)
    parser.add_argument("--crop-margin-px", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_pdfs(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")
    pdfs = sorted(
        (path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: ranked_sort_key(path, input_root),
    )
    return pdfs

def ranked_sort_key(path: Path, input_root: Path) -> tuple[int, int | str, str]:
    relative = path.relative_to(input_root).as_posix()
    match = re.match(r"^(\d+)[_-]", path.name)
    if match:
        return (0, int(match.group(1)), relative.casefold())
    return (1, relative.casefold(), relative.casefold())

def ranked_sort_key(path: Path, input_root: Path) -> tuple[int, int | str, str]:
    relative = path.relative_to(input_root).as_posix()
    match = re.match(r"^(\d+)[_-]", path.name)
    if match:
        return (0, int(match.group(1)), relative.casefold())
    return (1, relative.casefold(), relative.casefold())


def select_pdf_batch(pdfs: list[Path], start_pdf: int, limit: int | None) -> list[tuple[int, Path]]:
    if start_pdf < 1:
        raise ValueError("--start-pdf must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    start_offset = start_pdf - 1
    selected = pdfs[start_offset:] if limit is None else pdfs[start_offset : start_offset + limit]
    return list(enumerate(selected, start=start_pdf))


def safe_page_id(pdf_path: Path, input_root: Path, position: int) -> str:
    relative_stem = pdf_path.relative_to(input_root).with_suffix("").as_posix()
    slug = re.sub(r"[^\w.-]+", "__", relative_stem, flags=re.UNICODE).strip("_.")
    return f"{position:04d}_{slug or 'page'}"


def render_first_page(pdf_path: Path, scale: float) -> Any:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        if len(document) == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")
        page = document[0]
        try:
            return page.render(scale=scale).to_pil().convert("RGB")
        finally:
            page.close()
    finally:
        document.close()


def bbox_from_box(box: Any, width: int, height: int, margin: int) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = box.bbox
    return (
        max(0, int(x_min) - margin),
        max(0, int(y_min) - margin),
        min(width, int(x_max) + margin),
        min(height, int(y_max) + margin),
    )


def sort_line_boxes(boxes: list[Any]) -> list[Any]:
    return sorted(boxes, key=lambda box: (box.bbox[1], -box.bbox[0]))


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def page_is_complete(page_dir: Path) -> bool:
    manifest = page_dir / "lines.jsonl"
    if not manifest.is_file():
        return False
    if not (page_dir / "metadata.json").is_file():
        return False
    rows = [json.loads(raw) for raw in manifest.read_text(encoding="utf-8").splitlines() if raw.strip()]
    return all(
        (PROJECT_ROOT / row["image_path"]).is_file()
        and (PROJECT_ROOT / row["ground_truth_path"]).is_file()
        for row in rows
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_pdf(
    pdf_path: Path,
    position: int,
    predictor: Any,
    args: argparse.Namespace,
) -> str:
    page_id = safe_page_id(pdf_path, args.input_root, position)
    page_dir = args.output_root / page_id
    if page_is_complete(page_dir) and not args.force:
        return "skipped"
    if page_dir.exists():
        if not args.force:
            raise RuntimeError(f"Incomplete output already exists; inspect it or rerun with --force: {page_dir}")
        shutil.rmtree(page_dir)
    page_dir.mkdir(parents=True)

    page_image = render_first_page(pdf_path, args.render_scale)
    prediction = predictor([page_image])[0]
    boxes = sort_line_boxes(list(prediction.bboxes))
    rows: list[dict[str, Any]] = []

    for line_index, box in enumerate(boxes, 1):
        line_id = f"line_{line_index:03d}"
        line_dir = page_dir / line_id
        line_dir.mkdir()
        bbox = bbox_from_box(box, page_image.width, page_image.height, args.crop_margin_px)
        image_path = line_dir / "image.png"
        label_path = line_dir / "ground_truth.txt"
        metadata_path = line_dir / "metadata.json"
        page_image.crop(bbox).save(image_path)
        label_path.write_text("", encoding="utf-8")

        row = {
            "line_id": line_id,
            "line_index": line_index,
            "image_path": portable_path(image_path),
            "ground_truth_path": portable_path(label_path),
            "bbox": list(bbox),
            "polygon": box.polygon,
            "confidence": box.confidence,
            "status": "awaiting_ground_truth",
        }
        write_json(metadata_path, row)
        rows.append(row)

    with (page_dir / "lines.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        page_dir / "metadata.json",
        {
            "page_id": page_id,
            "source_pdf": portable_path(pdf_path),
            "source_page": 1,
            "image_width": page_image.width,
            "image_height": page_image.height,
            "line_count": len(rows),
            "line_detector": "surya-ocr",
            "line_order": "top-to-bottom, then right-to-left for ties",
            "label_status": "manual_labels_required",
        },
    )
    return "completed"


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    all_pdfs = find_pdfs(args.input_root)
    pdfs = select_pdf_batch(all_pdfs, args.start_pdf, args.limit)
    print(
        f"Found PDFs: {len(all_pdfs)}; selected: {len(pdfs)} "
        f"starting at position {args.start_pdf}",
        flush=True,
    )
    if args.dry_run:
        for position, pdf_path in pdfs:
            print(f"{position:04d}: {pdf_path.relative_to(args.input_root)}")
        return

    from surya.detection import DetectionPredictor

    args.output_root.mkdir(parents=True, exist_ok=True)
    predictor = DetectionPredictor()
    completed = skipped = failed = 0
    for batch_index, (position, pdf_path) in enumerate(pdfs, 1):
        print(
            f"[{batch_index}/{len(pdfs)}] pdf_position={position} "
            f"{pdf_path.relative_to(args.input_root)}",
            flush=True,
        )
        try:
            status = process_pdf(pdf_path, position, predictor, args)
            completed += status == "completed"
            skipped += status == "skipped"
        except Exception as exc:
            failed += 1
            print(f"ERROR: {exc}", flush=True)
            if not args.continue_on_error:
                raise

    summary = {
        "available_pdfs": len(all_pdfs),
        "start_pdf": args.start_pdf,
        "requested_limit": args.limit,
        "selected_pdfs": len(pdfs),
        "last_selected_pdf": pdfs[-1][0] if pdfs else None,
        "completed_pdfs": completed,
        "skipped_pdfs": skipped,
        "failed_pdfs": failed,
        "output_root": str(args.output_root),
    }
    write_json(args.output_root / "run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
