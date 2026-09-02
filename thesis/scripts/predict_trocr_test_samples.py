"""Run the trained thesis TrOCR model on the held-out ground-truth test split."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import common


DEFAULT_SPLIT_ROOT = common.PROJECT_ROOT / "thesis" / "datasets" / "ground_truth_lines_ml"
DEFAULT_MODEL_DIR = common.PROJECT_ROOT / "thesis" / "models" / "ground_truth_line_trocr"
DEFAULT_OUTPUT_ROOT = common.PROJECT_ROOT / "thesis" / "predictions" / "my_TrOCR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict text for held-out test lines with the trained TrOCR model.")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_SPLIT_ROOT / "test.jsonl")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Missing test manifest: {path}")
    return [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip()]


def prediction_dir(output_root: Path, sample_id: str) -> Path:
    return output_root / sample_id


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else common.PROJECT_ROOT / path


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def write_prediction(output_dir: Path, text: str, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction.txt").write_text(text.strip() + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_prediction(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def evaluate_predictions(records: list[dict], output_root: Path) -> dict:
    aggregate = common.empty_ocr_aggregate()
    for record in records:
        pred_path = prediction_dir(output_root, record["sample_id"]) / "prediction.txt"
        prediction = common.normalize_text_for_evaluation(read_prediction(pred_path))
        reference = common.normalize_text_for_evaluation(record.get("text") or "")
        scores = common.score_line_ocr(reference, prediction)
        common.add_ocr_score(aggregate, scores, missing=not pred_path.is_file())
    summary = common.finalize_ocr_aggregate(aggregate)
    return {
        "test_lines": summary["evaluated_lines"],
        "missing_predictions": summary["missing_predictions"],
        "letter_accuracy": summary["letter_accuracy"],
        "word_accuracy": summary["word_accuracy"],
        "exact_line_accuracy": summary["line_accuracy"],
        "letter_accuracy_pct": pct(float(summary["letter_accuracy"])),
        "word_accuracy_pct": pct(float(summary["word_accuracy"])),
        "exact_line_accuracy_pct": pct(float(summary["line_accuracy"])),
        "exact_lines_correct": summary["exact_lines"],
        "total_reference_chars": summary["reference_chars"],
        "total_char_distance": summary["char_distance"],
        "total_reference_words": summary["reference_words"],
        "total_word_distance": summary["word_distance"],
    }


def load_experiment_config(model_dir: Path) -> dict:
    config_path = model_dir / "experiment_config.json"
    if not config_path.is_file():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def write_evaluation_result(
    model_dir: Path,
    output_root: Path,
    test_manifest: Path,
    metrics: dict,
    model_label: str = "TrOCR",
) -> Path:
    experiment = load_experiment_config(model_dir)
    report_dir = common.PROJECT_ROOT / "thesis" / "reports" / "trocr_experiments"
    report_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_label,
        "checkpoint": str(model_dir),
        "prediction_root": str(output_root),
        "test_split": str(test_manifest),
        "khatt_enabled": experiment.get("khatt_enabled", False),
        "khatt_samples_requested": experiment.get("khatt_samples_requested", 0),
        "khatt_samples_used": experiment.get("khatt_samples_used", 0),
        "khatt_dataset_name": experiment.get("khatt_dataset_name", ""),
        "own_training_lines": experiment.get("own_training_samples"),
        "combined_training_lines": experiment.get("combined_training_samples"),
        "random_seed": experiment.get("random_seed"),
        **metrics,
    }
    result_path = report_dir / f"{model_dir.name}_evaluation.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result_path


def main() -> None:
    args = parse_args()
    args.test_manifest = project_path(args.test_manifest)
    args.model = project_path(args.model)
    args.output_root = project_path(args.output_root)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.max_length < 1:
        raise SystemExit("--max-length must be at least 1")

    records = read_manifest(args.test_manifest)
    if args.dry_run:
        for record in records:
            print(f"{record['sample_id']}: {record['image_path']}")
        return
    if not args.model.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model}")

    try:
        import torch
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise SystemExit(f"Model dependencies are missing: {exc}") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    skipped = 0
    started = time.monotonic()
    for offset in range(0, len(records), args.batch_size):
        batch = records[offset : offset + args.batch_size]
        pending = []
        for record in batch:
            output_dir = prediction_dir(args.output_root, record["sample_id"])
            prediction_path = output_dir / "prediction.txt"
            if prediction_path.is_file() and not args.force:
                skipped += 1
                continue
            image_path = project_path(record["image_path"])
            pending.append((record, image_path, output_dir, prediction_path))
        if not pending:
            continue

        pil_images = [Image.open(image_path).convert("RGB") for _, image_path, _, _ in pending]
        pixels = processor(images=pil_images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated = model.generate(pixels, max_length=args.max_length)
        texts = processor.batch_decode(generated, skip_special_tokens=True)

        for (record, image_path, output_dir, prediction_path), text in zip(pending, texts):
            row = {
                "sample_id": record["sample_id"],
                "page_id": record["page_id"],
                "line_index": record["line_index"],
                "image_path": str(image_path.relative_to(common.PROJECT_ROOT)),
                "prediction_path": str(prediction_path.relative_to(common.PROJECT_ROOT)),
                "model": "my_TrOCR",
            }
            write_prediction(output_dir, text, row)
            rows.append(row)
            print(f"{record['sample_id']}: {text.strip()}", flush=True)

    with (args.output_root / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_root / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["sample_id", "page_id", "line_index", "image_path", "prediction_path", "model"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": "my_TrOCR",
        "test_samples": len(records),
        "completed": len(rows),
        "skipped": skipped,
        "device": str(device),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    metrics = evaluate_predictions(records, args.output_root)
    result_path = write_evaluation_result(args.model, args.output_root, args.test_manifest, metrics)
    (args.output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("TrOCR evaluation:")
    print(f"  Letter accuracy:     {metrics['letter_accuracy_pct']}%")
    print(f"  Word accuracy:       {metrics['word_accuracy_pct']}%")
    print(f"  Exact line accuracy: {metrics['exact_line_accuracy_pct']}%")
    print(f"  Exact lines correct: {metrics['exact_lines_correct']} / {metrics['test_lines']}")
    print(f"Wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
