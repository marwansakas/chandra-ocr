"""Train several my_TrOCR augmentation experiments on the same split."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import common
import train_line_ocr_model as base


DEFAULT_MODEL_ROOT = common.PROJECT_ROOT / "thesis" / "models" / "augmentation_trocr"
DEFAULT_PREDICTION_ROOT = common.PROJECT_ROOT / "thesis" / "predictions" / "augmentation_trocr"
DEFAULT_REPORT = common.PROJECT_ROOT / "thesis" / "reports" / "trocr_augmentation_experiments.csv"


def parse_profiles(raw: str) -> list[str]:
    profiles = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(profiles) - set(base.AUGMENTATION_PROFILES))
    if invalid:
        raise SystemExit(f"Unknown augmentation profile(s): {', '.join(invalid)}")
    if not profiles:
        raise SystemExit("At least one augmentation profile is required")
    return profiles


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else common.PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate multiple my_TrOCR augmentation profiles.")
    parser.add_argument(
        "--profiles",
        default="none,light,scan,geometry,stroke,mixed",
        help=f"Comma-separated profiles. Choices: {', '.join(base.AUGMENTATION_PROFILES)}",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dataset-root", type=Path, default=base.DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-root", type=Path, default=base.DEFAULT_SPLIT_ROOT)
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-strategy", choices=("student", "line"), default="student")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--khatt-samples", type=int, default=0)
    parser.add_argument("--khatt-cache-root", type=Path, default=base.DEFAULT_KHATT_CACHE_ROOT)
    parser.add_argument("--khatt-dataset-name", default=base.DEFAULT_KHATT_DATASET)
    parser.add_argument("--refresh-khatt-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-label-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--views-per-line", type=int, default=3)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def add_common_training_args(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--dataset-root",
            str(project_path(args.dataset_root)),
            "--split-root",
            str(project_path(args.split_root)),
            "--model-name",
            args.model_name,
            "--train-ratio",
            str(args.train_ratio),
            "--split-strategy",
            args.split_strategy,
            "--seed",
            str(args.seed),
            "--khatt-samples",
            str(args.khatt_samples),
            "--khatt-cache-root",
            str(project_path(args.khatt_cache_root)),
            "--khatt-dataset-name",
            args.khatt_dataset_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--max-label-length",
            str(args.max_label_length),
            "--num-workers",
            str(args.num_workers),
            "--views-per-line",
            str(args.views_per_line),
        ]
    )
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.eval_limit is not None:
        command.extend(["--eval-limit", str(args.eval_limit)])
    if args.refresh_khatt_cache:
        command.append("--refresh-khatt-cache")
    if args.no_cuda:
        command.append("--no-cuda")


def collect_evaluation_rows(profiles: list[str], model_root: Path) -> list[dict]:
    rows = []
    report_dir = common.PROJECT_ROOT / "thesis" / "reports" / "trocr_experiments"
    for profile in profiles:
        result_path = report_dir / f"{profile}_evaluation.json"
        if not result_path.is_file():
            continue
        row = json.loads(result_path.read_text(encoding="utf-8"))
        experiment_path = model_root / profile / "experiment_config.json"
        experiment = json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path.is_file() else {}
        rows.append(
            {
                "augmentation_profile": profile,
                "checkpoint": row.get("checkpoint", ""),
                "prediction_root": row.get("prediction_root", ""),
                "letter_accuracy_pct": row.get("letter_accuracy_pct", ""),
                "word_accuracy_pct": row.get("word_accuracy_pct", ""),
                "exact_line_accuracy_pct": row.get("exact_line_accuracy_pct", ""),
                "exact_lines_correct": row.get("exact_lines_correct", ""),
                "test_lines": row.get("test_lines", ""),
                "views_per_line": experiment.get("views_per_line", ""),
                "khatt_samples_used": row.get("khatt_samples_used", 0),
            }
        )
    return rows


def write_report(path: Path, rows: list[dict]) -> None:
    fields = [
        "augmentation_profile",
        "checkpoint",
        "prediction_root",
        "letter_accuracy_pct",
        "word_accuracy_pct",
        "exact_line_accuracy_pct",
        "exact_lines_correct",
        "test_lines",
        "views_per_line",
        "khatt_samples_used",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


def main() -> None:
    args = parse_args()
    profiles = parse_profiles(args.profiles)
    model_root = project_path(args.model_root)
    prediction_root = project_path(args.prediction_root)
    train_script = common.PROJECT_ROOT / "thesis" / "scripts" / "train_line_ocr_model.py"
    predict_script = common.PROJECT_ROOT / "thesis" / "scripts" / "predict_trocr_test_samples.py"

    for profile in profiles:
        model_dir = model_root / profile
        command = [
            sys.executable,
            str(train_script),
            "--output-dir",
            str(model_dir),
            "--augmentation-profile",
            profile,
        ]
        add_common_training_args(command, args)
        run_command(command)

        if args.skip_predict:
            continue
        output_root = prediction_root / profile
        predict_command = [
            sys.executable,
            str(predict_script),
            "--model",
            str(model_dir),
            "--output-root",
            str(output_root),
            "--test-manifest",
            str(project_path(args.split_root) / "test.jsonl"),
            "--batch-size",
            str(args.eval_batch_size),
            "--max-length",
            str(args.max_label_length),
            "--device",
            args.device,
        ]
        if args.force_predict:
            predict_command.append("--force")
        run_command(predict_command)

    if not args.skip_predict:
        write_report(project_path(args.report), collect_evaluation_rows(profiles, model_root))


if __name__ == "__main__":
    main()
