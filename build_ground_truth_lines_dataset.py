"""Build a clean ground-truth line dataset from labeled Surya line folders."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "thesis" / "datasets" / "surya_lines"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "thesis" / "datasets" / "ground_truth_lines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy labeled Surya line folders into a flat clean dataset, skipping "
            "empty labels and labels containing 'ignore'."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Number of page folders to process after deterministic sorting. Omit to process all pages.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the output folder if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without creating or deleting folders.",
    )
    return parser.parse_args()


def sorted_child_dirs(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )


def select_pages(input_root: Path, pages: int | None) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")
    if pages is not None and pages < 1:
        raise ValueError("--pages must be at least 1")
    all_pages = sorted_child_dirs(input_root)
    return all_pages if pages is None else all_pages[:pages]


def should_skip_ground_truth(ground_truth_path: Path) -> bool:
    text = ground_truth_path.read_text(encoding="utf-8").strip()
    return not text or "ignore" in text.casefold()


def build_output_name(global_line_number: int, page_dir: Path, line_dir: Path) -> str:
    return f"{global_line_number:06d}_{page_dir.name}_{line_dir.name}"


def prepare_output_root(output_root: Path, force: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if output_root.exists():
        if not force:
            raise FileExistsError(f"Output folder already exists; rerun with --force to replace it: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def build_dataset(input_root: Path, output_root: Path, pages: int | None, force: bool, dry_run: bool) -> dict[str, int]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if input_root == output_root:
        raise ValueError("--input-root and --output-root must be different folders")

    selected_pages = select_pages(input_root, pages)
    prepare_output_root(output_root, force=force, dry_run=dry_run)

    copied = skipped_empty_or_ignore = skipped_missing_ground_truth = total_lines = 0
    for page_dir in selected_pages:
        for line_dir in sorted_child_dirs(page_dir):
            total_lines += 1
            ground_truth_path = line_dir / "ground_truth.txt"
            if not ground_truth_path.is_file():
                skipped_missing_ground_truth += 1
                continue
            if should_skip_ground_truth(ground_truth_path):
                skipped_empty_or_ignore += 1
                continue

            global_line_number = copied + 1
            output_name = build_output_name(global_line_number, page_dir, line_dir)
            destination = output_root / output_name
            if dry_run:
                print(f"{output_name} <- {line_dir.relative_to(input_root)}")
            else:
                shutil.copytree(line_dir, destination)
            copied += 1

    return {
        "selected_pages": len(selected_pages),
        "total_line_folders": total_lines,
        "copied_lines": copied,
        "skipped_empty_or_ignore": skipped_empty_or_ignore,
        "skipped_missing_ground_truth": skipped_missing_ground_truth,
    }


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        pages=args.pages,
        force=args.force,
        dry_run=args.dry_run,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
