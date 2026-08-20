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


def safe_accuracy(error_rate: float) -> float:
    return max(0.0, 1.0 - error_rate)


def score_line(reference: str, prediction: str) -> dict[str, float | int]:
    reference = common.normalize_text(reference)
    prediction = common.normalize_text(prediction)
    ref_words = common.tokenize_words(reference)
    pred_words = common.tokenize_words(prediction)
    char_distance = common.edit_distance(reference, prediction)
    word_distance = common.edit_distance(ref_words, pred_words)
    char_error = char_distance / len(reference) if reference else (0.0 if not prediction else 1.0)
    word_error = word_distance / len(ref_words) if ref_words else (0.0 if not pred_words else 1.0)
    return {
        "reference_chars": len(reference),
        "prediction_chars": len(prediction),
        "reference_words": len(ref_words),
        "prediction_words": len(pred_words),
        "char_distance": char_distance,
        "word_distance": word_distance,
        "letter_accuracy": safe_accuracy(char_error),
        "word_accuracy": safe_accuracy(word_error),
        "line_accuracy": 1.0 if reference == prediction else 0.0,
    }


def main() -> None:
    args = parse_args()
    records = read_manifest(args.test_manifest)
    model_roots = {
        "chandra": args.chandra_root,
        "invizo": args.invizo_root,
        "my_TrOCR": args.trocr_root,
    }

    detail_rows = []
    aggregates = {
        model: {
            "evaluated_lines": 0,
            "missing_predictions": 0,
            "exact_lines": 0,
            "reference_chars": 0,
            "char_distance": 0,
            "reference_words": 0,
            "word_distance": 0,
        }
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
            agg = aggregates[model]
            agg["evaluated_lines"] += 1
            agg["missing_predictions"] += int(missing)
            agg["exact_lines"] += int(scores["line_accuracy"])
            agg["reference_chars"] += int(scores["reference_chars"])
            agg["char_distance"] += int(scores["char_distance"])
            agg["reference_words"] += int(scores["reference_words"])
            agg["word_distance"] += int(scores["word_distance"])

    summary_rows = []
    for model, agg in aggregates.items():
        line_accuracy = agg["exact_lines"] / agg["evaluated_lines"] if agg["evaluated_lines"] else 0.0
        letter_accuracy = (
            safe_accuracy(agg["char_distance"] / agg["reference_chars"])
            if agg["reference_chars"]
            else 0.0
        )
        word_accuracy = (
            safe_accuracy(agg["word_distance"] / agg["reference_words"])
            if agg["reference_words"]
            else 0.0
        )
        summary_rows.append(
            {
                "model": model,
                "evaluated_lines": agg["evaluated_lines"],
                "missing_predictions": agg["missing_predictions"],
                "letter_accuracy_pct": pct(letter_accuracy),
                "word_accuracy_pct": pct(word_accuracy),
                "line_accuracy_pct": pct(line_accuracy),
                "total_reference_chars": agg["reference_chars"],
                "total_char_distance": agg["char_distance"],
                "total_reference_words": agg["reference_words"],
                "total_word_distance": agg["word_distance"],
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
