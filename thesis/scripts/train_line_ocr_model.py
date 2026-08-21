"""
Fine-tune a line-level handwriting OCR model from Surya line crops and labels.

This script consumes the clean dataset produced by build_ground_truth_lines_dataset.py:

  thesis/datasets/ground_truth_lines/<global>_<page>_<line>/image.png
  thesis/datasets/ground_truth_lines/<global>_<page>_<line>/ground_truth.txt

It writes deterministic train/test manifests, then optionally fine-tunes a
Transformer OCR model such as microsoft/trocr-base-handwritten.

Each line folder contains an image and a manually editable ground-truth file.
Lines with empty labels are excluded from training by default because they do
not teach the model what text appears in the image.

Prepare splits only:
  python thesis/scripts/train_line_ocr_model.py --prepare-only

Train on a GPU allocation:
  python thesis/scripts/train_line_ocr_model.py --epochs 5 --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import common


PROJECT_ROOT = common.PROJECT_ROOT
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "thesis" / "datasets" / "ground_truth_lines"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "thesis" / "datasets" / "ground_truth_lines_ml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "thesis" / "models" / "ground_truth_line_trocr"
DEFAULT_CHANDRA = Path("/home/sakas/.conda/envs/chandra/bin/chandra")
DEFAULT_KHATT_DATASET = "johnlockejrr/KHATT_v1.0_dataset"
DEFAULT_KHATT_CACHE_ROOT = PROJECT_ROOT / "thesis" / "datasets" / "khatt"


@dataclass
class LineRecord:
    sample_id: str
    page_id: str
    page_dir: str
    rank: int
    line_index: int
    image_path: str
    text: str
    bbox: list[int]
    polygon: list
    confidence: float | None
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--split-strategy",
        choices=("student", "line"),
        default="student",
        help=(
            "student keeps pages from the same notebook/student together; "
            "line gives an even random line-level split."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-empty-labels", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--khatt-samples",
        type=int,
        default=0,
        help="Number of streamed KHATT train samples to add to TrOCR training only. 0 disables KHATT.",
    )
    parser.add_argument("--khatt-cache-root", type=Path, default=DEFAULT_KHATT_CACHE_ROOT)
    parser.add_argument("--khatt-dataset-name", default=DEFAULT_KHATT_DATASET)
    parser.add_argument(
        "--refresh-khatt-cache",
        action="store_true",
        help="Rebuild the selected KHATT cache for this sample count and seed.",
    )

    parser.add_argument("--fill-missing-ocr", action="store_true")
    parser.add_argument(
        "--treat-empty-text-as-missing",
        action="store_true",
        help="When repairing OCR, rerun Chandra for lines whose text is blank.",
    )
    parser.add_argument("--chandra-bin", type=Path, default=DEFAULT_CHANDRA)
    parser.add_argument("--chandra-method", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--chandra-batch-size", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-label-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--views-per-line",
        type=int,
        default=1,
        help="Full-line training views per record. Extra views use label-preserving augmentation.",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=None)
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def dataset_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "dataset"


def iter_page_dirs(dataset_root: Path) -> Iterable[Path]:
    return sorted(
        (p for p in dataset_root.iterdir() if p.is_dir() and p.name[:4].isdigit()),
        key=lambda p: (int(p.name[:4]), p.name),
    )


def find_line_image(line_dir: Path) -> Path | None:
    for name in ("image.png", "image.jpg", "image.jpeg", "image.tif", "image.tiff"):
        candidate = line_dir / name
        if candidate.is_file():
            return candidate
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.webp"):
        matches = sorted(line_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def flat_line_sort_key(line_dir: Path) -> tuple[int, str]:
    match = re.match(r"^(\d+)_", line_dir.name)
    return (int(match.group(1)) if match else 10**12, line_dir.name.casefold())


def parse_flat_line_folder_name(name: str) -> tuple[str, int, int]:
    match = re.match(r"^(?P<global>\d+)_(?P<page>.+)_(?P<line>line_(?P<line_index>\d+))$", name)
    if not match:
        return name, 0, 0
    return match.group("page"), int(match.group("global")), int(match.group("line_index"))


def read_flat_line_records(dataset_root: Path) -> list[LineRecord]:
    records: list[LineRecord] = []
    for line_dir in sorted((p for p in dataset_root.iterdir() if p.is_dir()), key=flat_line_sort_key):
        label_path = line_dir / "ground_truth.txt"
        image_path = find_line_image(line_dir)
        if image_path is None or not label_path.is_file():
            continue
        text = label_path.read_text(encoding="utf-8", errors="replace").strip()
        page_id, rank, line_index = parse_flat_line_folder_name(line_dir.name)
        metadata_path = line_dir / "metadata.json"
        metadata = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        records.append(
            LineRecord(
                sample_id=line_dir.name,
                page_id=page_id,
                page_dir=portable_path(line_dir),
                rank=rank,
                line_index=line_index,
                image_path=str(image_path),
                text=text,
                bbox=metadata.get("bbox") or [],
                polygon=metadata.get("polygon") or [],
                confidence=metadata.get("confidence"),
                status=metadata.get("status") or "ground_truth_ready",
            )
        )
    return records


def read_line_records(dataset_root: Path) -> list[LineRecord]:
    return read_flat_line_records(dataset_root)


def khatt_cache_dir(cache_root: Path, dataset_name: str, samples: int, seed: int) -> Path:
    return cache_root / dataset_slug(dataset_name) / f"samples_{samples}_seed_{seed}"


def normalize_khatt_text(text: str) -> str:
    """Convert KHATT's stored LTR-flipped label back to normal Arabic order.

    The Hugging Face KHATT dataset card documents that its ``text`` value was
    intentionally flipped from RTL to LTR for PyLaia. Reversing the codepoint
    sequence restores labels such as "هبحصب" -> "بصحبه", then the project's
    existing whitespace-only normalization is applied.
    """
    return common.normalize_text((text or "")[::-1])


def read_khatt_manifest(path: Path) -> list[LineRecord]:
    records: list[LineRecord] = []
    if not path.is_file():
        return records
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        records.append(
            LineRecord(
                sample_id=row["sample_id"],
                page_id=row["page_id"],
                page_dir=row["page_dir"],
                rank=int(row["rank"]),
                line_index=int(row["line_index"]),
                image_path=str(project_path(row["image_path"])),
                text=row["text"],
                bbox=row.get("bbox") or [],
                polygon=row.get("polygon") or [],
                confidence=row.get("confidence"),
                status=row.get("status") or "khatt_ready",
            )
        )
    return records


def write_khatt_manifest(path: Path, records: list[LineRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(
                json.dumps(
                    {
                        "sample_id": record.sample_id,
                        "page_id": record.page_id,
                        "page_dir": record.page_dir,
                        "rank": record.rank,
                        "line_index": record.line_index,
                        "image_path": portable_path(Path(record.image_path)),
                        "text": record.text,
                        "bbox": record.bbox,
                        "polygon": record.polygon,
                        "confidence": record.confidence,
                        "status": record.status,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def prepare_khatt_records(args: argparse.Namespace) -> tuple[list[LineRecord], dict]:
    requested = args.khatt_samples
    summary = {
        "khatt_enabled": requested > 0,
        "khatt_dataset_name": args.khatt_dataset_name,
        "khatt_samples_requested": requested,
        "khatt_samples_used": 0,
        "khatt_cache_dir": "",
        "khatt_cache_reused": False,
        "skipped": {},
    }
    if requested <= 0:
        return [], summary

    cache_root = project_path(args.khatt_cache_root)
    cache_dir = khatt_cache_dir(cache_root, args.khatt_dataset_name, requested, args.seed)
    manifest_path = cache_dir / "manifest.jsonl"
    summary_path = cache_dir / "summary.json"
    summary["khatt_cache_dir"] = portable_path(cache_dir)

    if manifest_path.is_file() and not args.refresh_khatt_cache:
        records = read_khatt_manifest(manifest_path)
        if len(records) >= requested:
            if summary_path.is_file():
                summary.update(json.loads(summary_path.read_text(encoding="utf-8")))
            summary["khatt_samples_used"] = len(records)
            summary["khatt_cache_reused"] = True
            return records, summary

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "KHATT support requires the Hugging Face datasets package. "
            "Install it with: python -m pip install datasets"
        ) from exc

    tmp_dir = cache_dir.with_name(cache_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    skipped: dict[str, int] = {}
    records: list[LineRecord] = []
    stream = load_dataset(args.khatt_dataset_name, split="train", streaming=True)
    for source_index, sample in enumerate(stream):
        if len(records) >= requested:
            break

        image = sample.get("image") if isinstance(sample, dict) else None
        if image is None:
            skipped["missing image"] = skipped.get("missing image", 0) + 1
            continue

        raw_text = sample.get("text") if isinstance(sample, dict) else None
        if not isinstance(raw_text, str):
            skipped["invalid transcription"] = skipped.get("invalid transcription", 0) + 1
            continue

        try:
            text = normalize_khatt_text(raw_text)
        except Exception:
            skipped["failed normalization"] = skipped.get("failed normalization", 0) + 1
            continue
        if not text:
            skipped["empty transcription"] = skipped.get("empty transcription", 0) + 1
            continue

        sample_number = len(records) + 1
        sample_id = f"khatt_{sample_number:06d}"
        sample_dir = tmp_dir / sample_id
        final_sample_dir = cache_dir / sample_id
        sample_dir.mkdir(parents=True)
        image_path = sample_dir / "image.png"
        final_image_path = final_sample_dir / "image.png"
        try:
            image.convert("RGB").save(image_path)
        except Exception:
            skipped["missing image"] = skipped.get("missing image", 0) + 1
            continue
        (sample_dir / "ground_truth.txt").write_text(text + "\n", encoding="utf-8")
        (sample_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "dataset": args.khatt_dataset_name,
                    "source_split": "train",
                    "source_index": source_index,
                    "original_text": raw_text,
                    "text": text,
                    "text_transform": "reverse_codepoint_sequence_then_project_whitespace_normalize",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        records.append(
            LineRecord(
                sample_id=sample_id,
                page_id="khatt/train",
                page_dir=portable_path(final_sample_dir),
                rank=1_000_000_000 + sample_number,
                line_index=sample_number,
                image_path=str(final_image_path),
                text=text,
                bbox=[],
                polygon=[],
                confidence=None,
                status="khatt_ready",
            )
        )

    if len(records) < requested:
        shutil.rmtree(tmp_dir)
        raise SystemExit(f"Only accepted {len(records)} KHATT samples out of requested {requested}. Skipped: {skipped}")

    write_khatt_manifest(tmp_dir / "manifest.jsonl", records)
    summary.update(
        {
            "khatt_samples_used": len(records),
            "khatt_cache_reused": False,
            "skipped": skipped,
        }
    )
    (tmp_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    tmp_dir.rename(cache_dir)
    return read_khatt_manifest(cache_dir / "manifest.jsonl"), summary


def combine_training_records(train: list[LineRecord], khatt: list[LineRecord], seed: int) -> list[LineRecord]:
    combined = [*train, *khatt]
    random.Random(seed).shuffle(combined)
    return combined


def read_surya_line_records(dataset_root: Path) -> list[LineRecord]:
    records: list[LineRecord] = []
    for page_dir in iter_page_dirs(dataset_root):
        lines_path = page_dir / "lines.jsonl"
        if not lines_path.exists():
            continue
        rank = int(page_dir.name[:4])
        for raw in lines_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            line_index = int(row["line_index"])
            image_path = project_path(row.get("image_path") or row["line_image"])
            label_value = row.get("ground_truth_path")
            label_path = project_path(label_value) if label_value else None
            text = row.get("text") or ""
            if label_path is not None and label_path.is_file():
                text = label_path.read_text(encoding="utf-8", errors="replace").strip()
            sample_id = f"{page_dir.name}__line_{line_index:03d}"
            records.append(
                LineRecord(
                    sample_id=sample_id,
                    page_id=page_dir.name,
                    page_dir=portable_path(page_dir),
                    rank=rank,
                    line_index=line_index,
                    image_path=str(image_path),
                    text=text,
                    bbox=row.get("bbox") or [],
                    polygon=row.get("polygon") or [],
                    confidence=row.get("confidence"),
                    status=row.get("status") or "",
                )
            )
    return records


def record_needs_ocr(record: LineRecord, treat_empty_text_as_missing: bool) -> bool:
    if record.status != "ocr_complete":
        return True
    if record.text is None:
        return True
    return treat_empty_text_as_missing and not record.text.strip()


def read_chandra_markdown(output_dir: Path, image_path: Path) -> str:
    stem = image_path.stem
    candidates = [
        output_dir / stem / f"{stem}.md",
        output_dir / f"{stem}.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace").strip()
    return ""


def update_record_in_lines_jsonl(record: LineRecord, new_text: str) -> None:
    page_dir = PROJECT_ROOT / record.page_dir
    lines_path = page_dir / "lines.jsonl"
    rows = []
    for raw in lines_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if int(row["line_index"]) == record.line_index:
            row["text"] = new_text
            row["status"] = "ocr_complete"
        rows.append(row)
    with open(lines_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def refresh_ground_truth_txt(page_dir: Path) -> None:
    lines_path = page_dir / "lines.jsonl"
    if not lines_path.exists():
        return
    lines = [json.loads(raw) for raw in lines_path.read_text(encoding="utf-8").splitlines() if raw.strip()]
    lines.sort(key=lambda row: int(row.get("line_index", 0)))
    text = "\n".join((row.get("text") or "").strip() for row in lines)
    (page_dir / "ground_truth.txt").write_text(text + "\n", encoding="utf-8")


def fill_missing_ocr(records: list[LineRecord], args: argparse.Namespace) -> int:
    missing = [r for r in records if record_needs_ocr(r, args.treat_empty_text_as_missing)]
    if not missing:
        return 0
    if not args.chandra_bin.exists():
        raise SystemExit(f"Chandra executable was not found: {args.chandra_bin}")

    repaired = 0
    touched_pages: set[Path] = set()
    for idx, record in enumerate(missing, start=1):
        image_path = Path(record.image_path)
        if not image_path.exists():
            print(f"[{idx}/{len(missing)}] missing image, skipping: {image_path}", file=sys.stderr)
            continue
        with tempfile.TemporaryDirectory(prefix="chandra_line_ocr_") as tmp:
            out_dir = Path(tmp) / "out"
            cmd = [
                str(args.chandra_bin),
                str(image_path),
                str(out_dir),
                "--method",
                args.chandra_method,
                "--no-images",
                "--max-output-tokens",
                str(args.max_output_tokens),
                "--batch-size",
                str(args.chandra_batch_size),
            ]
            print(f"[{idx}/{len(missing)}] repairing OCR for {record.sample_id}")
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.returncode != 0:
                print(result.stdout, file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                continue
            new_text = read_chandra_markdown(out_dir, image_path)
            update_record_in_lines_jsonl(record, new_text)
            page_dir = PROJECT_ROOT / record.page_dir
            line_out = page_dir / "chandra_line_ocr" / image_path.stem
            line_out.mkdir(parents=True, exist_ok=True)
            for produced in (out_dir / image_path.stem).glob("*") if (out_dir / image_path.stem).exists() else []:
                shutil.copy2(produced, line_out / produced.name)
            touched_pages.add(page_dir)
            repaired += 1

    for page_dir in touched_pages:
        refresh_ground_truth_txt(page_dir)
    return repaired


def keep_for_training(record: LineRecord, include_empty_labels: bool) -> bool:
    if not Path(record.image_path).exists():
        return False
    if "ignore" in (record.text or "").casefold():
        return False
    if include_empty_labels:
        return record.text is not None
    return bool((record.text or "").strip())


def student_group(page_id: str) -> str:
    """Group every page from the same notebook/student to prevent leakage."""
    without_rank = re.sub(r"^\d{4}_\d{4}_", "", page_id)
    return re.sub(r"_Page_\d+$", "", without_rank, flags=re.IGNORECASE)


def split_records_by_student(records: list[LineRecord], train_ratio: float, seed: int) -> tuple[list[LineRecord], list[LineRecord]]:
    by_student: dict[str, list[LineRecord]] = {}
    for record in records:
        by_student.setdefault(student_group(record.page_id), []).append(record)
    students = sorted(by_student)
    random.Random(seed).shuffle(students)
    train_count = (
        max(1, min(len(students) - 1, round(len(students) * train_ratio)))
        if len(students) > 1
        else len(students)
    )
    train_students = set(students[:train_count])
    train = [r for r in records if student_group(r.page_id) in train_students]
    test = [r for r in records if student_group(r.page_id) not in train_students]
    return train, test


def split_records_by_line(records: list[LineRecord], train_ratio: float, seed: int) -> tuple[list[LineRecord], list[LineRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * train_ratio)))
    train = sorted(shuffled[:train_count], key=lambda r: (r.rank, r.line_index, r.sample_id))
    test = sorted(shuffled[train_count:], key=lambda r: (r.rank, r.line_index, r.sample_id))
    return train, test


def split_records(
    records: list[LineRecord],
    train_ratio: float,
    seed: int,
    split_strategy: str,
) -> tuple[list[LineRecord], list[LineRecord]]:
    if split_strategy == "line":
        return split_records_by_line(records, train_ratio, seed)
    if split_strategy == "student":
        return split_records_by_student(records, train_ratio, seed)
    raise ValueError(f"Unknown split strategy: {split_strategy}")


def write_manifest(path: Path, records: list[LineRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "page_id",
        "rank",
        "line_index",
        "image_path",
        "text",
        "bbox",
        "confidence",
        "status",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "page_id": record.page_id,
                    "rank": record.rank,
                    "line_index": record.line_index,
                    "image_path": portable_path(Path(record.image_path)),
                    "text": record.text,
                    "bbox": json.dumps(record.bbox, ensure_ascii=False),
                    "confidence": "" if record.confidence is None else record.confidence,
                    "status": record.status,
                }
            )


def write_jsonl(path: Path, records: list[LineRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(
                json.dumps(
                    {
                        "sample_id": record.sample_id,
                        "page_id": record.page_id,
                        "rank": record.rank,
                        "line_index": record.line_index,
                        "image_path": portable_path(Path(record.image_path)),
                        "text": record.text,
                        "bbox": record.bbox,
                        "polygon": record.polygon,
                        "confidence": record.confidence,
                        "status": record.status,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_summary(
    path: Path,
    all_records: list[LineRecord],
    usable: list[LineRecord],
    train: list[LineRecord],
    test: list[LineRecord],
    split_strategy: str,
) -> None:
    blank = sum(1 for r in all_records if not (r.text or "").strip())
    pages = len({r.page_id for r in all_records})
    train_pages = len({r.page_id for r in train})
    test_pages = len({r.page_id for r in test})
    lines = [
        "# Surya Line OCR ML Split",
        "",
        f"- Source pages: {pages}",
        f"- Source lines: {len(all_records)}",
        f"- Usable labeled lines: {len(usable)}",
        f"- Empty ground-truth labels excluded by default: {blank}",
        f"- Split strategy: {split_strategy}",
        f"- Train lines: {len(train)} ({train_pages} pages)",
        f"- Test lines: {len(test)} ({test_pages} pages)",
        "",
    ]
    if split_strategy == "student":
        lines.append("The split is notebook/student-level, so pages from the same child do not appear in both train and test.")
    else:
        lines.append("The split is line-level, so it gives a more even line count but may place lines from the same page/student in both train and test.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_experiment(
    output_dir: Path,
    args: argparse.Namespace,
    own_train: list[LineRecord],
    khatt_train: list[LineRecord],
    combined_train: list[LineRecord],
    test: list[LineRecord],
    khatt_summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "TrOCR",
        "base_model": args.model_name,
        "checkpoint": portable_path(output_dir),
        "dataset_root": portable_path(project_path(args.dataset_root)),
        "split_root": portable_path(project_path(args.split_root)),
        "test_split": portable_path(project_path(args.split_root) / "test.jsonl"),
        "split_strategy": args.split_strategy,
        "train_ratio": args.train_ratio,
        "random_seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "views_per_line": args.views_per_line,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
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


class LineOCRDataset:
    def __init__(
        self,
        records: list[LineRecord],
        processor,
        max_label_length: int,
        views_per_line: int = 1,
        augment: bool = False,
    ):
        self.records = records
        self.processor = processor
        self.max_label_length = max_label_length
        self.views_per_line = max(1, views_per_line)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records) * self.views_per_line

    def __getitem__(self, idx: int):
        from PIL import Image, ImageEnhance, ImageOps

        record = self.records[idx // self.views_per_line]
        view = idx % self.views_per_line
        image = Image.open(record.image_path).convert("RGB")
        if self.augment and view:
            # Preserve the complete line. A fragment cannot be paired with the
            # full transcription without introducing a false training label.
            rng = random.Random(f"{record.sample_id}:{view}")
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
            image = ImageOps.expand(image, border=rng.randint(1, 5), fill="white")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            record.text,
            padding="max_length",
            max_length=self.max_label_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100
        return {"pixel_values": pixel_values, "labels": labels, "record": record}


def collate_batch(batch):
    import torch

    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "records": [item["record"] for item in batch],
    }


def decode_predictions(processor, generated_ids, label_ids) -> tuple[list[str], list[str]]:
    import torch

    labels = label_ids.clone()
    labels[labels == -100] = processor.tokenizer.pad_token_id
    pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    label_texts = processor.batch_decode(labels, skip_special_tokens=True)
    return pred_texts, label_texts


def cer(reference: str, hypothesis: str) -> float:
    return common.ratio_error(common.normalize_text(reference), common.normalize_text(hypothesis))


def train_model(train: list[LineRecord], test: list[LineRecord], args: argparse.Namespace) -> None:
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
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    model.to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = args.max_label_length

    train_ds = LineOCRDataset(
        train,
        processor,
        args.max_label_length,
        views_per_line=args.views_per_line,
        augment=args.views_per_line > 1,
    )
    test_ds = LineOCRDataset(test, processor, args.max_label_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
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
                    print(
                        f"epoch={epoch} step={step}/{len(train_loader)} "
                        f"loss={running_loss / step:.4f}"
                    )

            eval_result = evaluate_model(model, processor, test_loader, device, args.eval_limit)
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
            model.save_pretrained(checkpoint)
            processor.save_pretrained(checkpoint)

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved model and processor to {args.output_dir}")


def evaluate_model(model, processor, loader, device, eval_limit: int | None) -> dict:
    import torch

    model.eval()
    total_cer = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            generated = model.generate(pixel_values, max_length=model.generation_config.max_length)
            preds, labels = decode_predictions(processor, generated.cpu(), batch["labels"])
            for pred, label in zip(preds, labels):
                total_cer += cer(label, pred)
                count += 1
                if eval_limit is not None and count >= eval_limit:
                    return {"eval_samples": count, "eval_cer": total_cer / count}
    return {"eval_samples": count, "eval_cer": total_cer / count if count else math.nan}


def main() -> None:
    args = parse_args()
    dataset_root = project_path(args.dataset_root)
    split_root = project_path(args.split_root)
    output_dir = project_path(args.output_dir)
    args.output_dir = output_dir
    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be between 0 and 1")
    if args.views_per_line < 1:
        raise SystemExit("--views-per-line must be at least 1")
    if args.khatt_samples < 0:
        raise SystemExit("--khatt-samples must be zero or greater")

    records = read_line_records(dataset_root)
    if not records:
        raise SystemExit(f"No line records found under {dataset_root}")

    if args.fill_missing_ocr:
        raise SystemExit("--fill-missing-ocr is not supported for the clean ground_truth_lines dataset")

    usable = [r for r in records if keep_for_training(r, args.include_empty_labels)]
    usable.sort(key=lambda r: (r.rank, r.line_index, r.sample_id))
    if args.max_samples is not None:
        usable = usable[: args.max_samples]
    if len(usable) < 2:
        raise SystemExit("Not enough labeled line records to split into train/test.")

    train, test = split_records(usable, args.train_ratio, args.seed, args.split_strategy)
    split_root.mkdir(parents=True, exist_ok=True)
    write_manifest(split_root / "train.csv", train)
    write_manifest(split_root / "test.csv", test)
    write_jsonl(split_root / "train.jsonl", train)
    write_jsonl(split_root / "test.jsonl", test)
    write_summary(split_root / "summary.md", records, usable, train, test, args.split_strategy)

    khatt_records, khatt_summary = prepare_khatt_records(args)
    combined_train = combine_training_records(train, khatt_records, args.seed)
    if khatt_records:
        write_jsonl(split_root / "train_with_khatt.jsonl", combined_train)

    print(f"Total line records: {len(records)}")
    print(f"Usable labeled line records: {len(usable)}")
    print(f"Train: {len(train)} lines")
    print(f"Test: {len(test)} lines")
    print(f"Split strategy: {args.split_strategy}")
    print(f"Wrote split files to: {split_root}")
    print(f"Own training lines: {len(train)}")
    print(f"KHATT requested: {args.khatt_samples}")
    print(f"KHATT accepted: {len(khatt_records)}")
    print(f"Combined training lines: {len(combined_train)}")
    if khatt_summary.get("skipped"):
        print(f"KHATT skipped: {json.dumps(khatt_summary['skipped'], ensure_ascii=False)}")

    if args.prepare_only:
        return

    write_training_experiment(args.output_dir, args, train, khatt_records, combined_train, test, khatt_summary)
    train_model(combined_train, test, args)


if __name__ == "__main__":
    main()
