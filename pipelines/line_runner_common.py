"""Shared input selection and prediction-output contract for line OCR runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "thesis" / "datasets" / "surya_lines"


def find_line_images(input_root: Path) -> list[Path]:
    return sorted(
        (path for path in input_root.glob("*/line_*/image.png") if path.is_file()),
        key=lambda path: path.relative_to(input_root).as_posix().casefold(),
    )


def select_batch(images: list[Path], start_line: int, limit: int | None) -> list[tuple[int, Path]]:
    if start_line < 1:
        raise ValueError("--start-line must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    start = start_line - 1
    selected = images[start:] if limit is None else images[start : start + limit]
    return list(enumerate(selected, start=start_line))


def prediction_dir(output_root: Path, input_root: Path, image_path: Path) -> Path:
    relative_line_dir = image_path.parent.relative_to(input_root)
    return output_root / relative_line_dir


def write_prediction(output_dir: Path, text: str, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction.txt").write_text(text.strip() + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_run_files(output_root: Path, rows: Iterable[dict], summary: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
