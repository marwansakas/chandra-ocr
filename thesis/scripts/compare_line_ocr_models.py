"""Compare Chandra, Invizo, and thesis TrOCR predictions on the same test lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common


DEFAULT_SPLIT_ROOT = common.PROJECT_ROOT / "thesis" / "datasets" / "ground_truth_lines_ml"
DEFAULT_REPORTS_DIR = common.PROJECT_ROOT / "thesis" / "reports"
DEFAULT_MODELS = {
    "chandra": common.PROJECT_ROOT / "thesis" / "predictions" / "chandra",
    "invizo": common.PROJECT_ROOT / "thesis" / "predictions" / "invizo",
    "my_TrOCR": common.PROJECT_ROOT / "thesis" / "predictions" / "my_TrOCR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score line OCR predictions against held-out ground truth.")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_SPLIT_ROOT / "test.jsonl")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--chandra-root", type=Path, default=DEFAULT_MODELS["chandra"])
    parser.add_argument("--invizo-root", type=Path, default=DEFAULT_MODELS["invizo"])
    parser.add_argument("--trocr-root", type=Path, default=DEFAULT_MODELS["my_TrOCR"])
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else common.PROJECT_ROOT / path


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Missing test manifest: {path}")
    return [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip()]


def prediction_path(model_root: Path, sample_id: str) -> Path:
    return model_root / sample_id / "prediction.txt"


def read_prediction(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def score_line(reference: str, prediction: str) -> dict[str, float | int]:
    return common.score_line_ocr(reference, prediction)


def main() -> None:
    args = parse_args()
    args.test_manifest = project_path(args.test_manifest)
    args.reports_dir = project_path(args.reports_dir)
    args.chandra_root = project_path(args.chandra_root)
    args.invizo_root = project_path(args.invizo_root)
    args.trocr_root = project_path(args.trocr_root)
    records = read_manifest(args.test_manifest)
    model_roots = {
        "chandra": args.chandra_root,
        "invizo": args.invizo_root,
        "my_TrOCR": args.trocr_root,
    }

    detail_rows = []
    aggregates = {
        model: common.empty_ocr_aggregate()
        for model in model_roots
    }

    for record in records:
        reference = common.normalize_text(record.get("text") or "")
        for model, root in model_roots.items():
            pred_path = prediction_path(root, record["sample_id"])
            prediction = common.normalize_text(read_prediction(pred_path))
            scores = score_line(reference, prediction)
            missing = not pred_path.is_file()
            detail_rows.append(
                {
                    "model": model,
                    "sample_id": record["sample_id"],
                    "page_id": record["page_id"],
                    "line_index": record["line_index"],
                    "prediction_present": int(not missing),
                    "letter_accuracy_pct": pct(float(scores["letter_accuracy"])),
                    "word_accuracy_pct": pct(float(scores["word_accuracy"])),
                    "line_accuracy_pct": pct(float(scores["line_accuracy"])),
                    "reference_chars": scores["reference_chars"],
                    "prediction_chars": scores["prediction_chars"],
                    "reference_words": scores["reference_words"],
                    "prediction_words": scores["prediction_words"],
                    "char_distance": scores["char_distance"],
                    "word_distance": scores["word_distance"],
                    "ground_truth": reference,
                    "prediction": prediction,
                    "prediction_path": str(pred_path.relative_to(common.PROJECT_ROOT)) if pred_path.is_file() else "",
                }
            )
            common.add_ocr_score(aggregates[model], scores, missing)

    summary_rows = []
    for model, agg in aggregates.items():
        summary = common.finalize_ocr_aggregate(agg)
        summary_rows.append(
            {
                "model": model,
                "evaluated_lines": summary["evaluated_lines"],
                "missing_predictions": summary["missing_predictions"],
                "letter_accuracy_pct": pct(float(summary["letter_accuracy"])),
                "word_accuracy_pct": pct(float(summary["word_accuracy"])),
                "line_accuracy_pct": pct(float(summary["line_accuracy"])),
                "total_reference_chars": summary["reference_chars"],
                "total_char_distance": summary["char_distance"],
                "total_reference_words": summary["reference_words"],
                "total_word_distance": summary["word_distance"],
            }
        )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.reports_dir / "line_ocr_model_comparison.csv"
    summary_path = args.reports_dir / "line_ocr_model_comparison_summary.csv"
    common.write_csv(detail_path, detail_rows)
    common.write_csv(summary_path, summary_rows)
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
