"""Validate, run and compare the five pretrained line OCR baselines."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import time

from . import __version__
from .dataset import check_output_path, read_dataset, select_samples
from .metrics import aggregate, score
from .models import load_backend
from .registry import MODELS

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT / "thesis/datasets/ground_truth_lines"
DEFAULT_OUTPUT = PROJECT / "thesis/predictions/pretrained"


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(data):
    return sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def packages():
    result = {"python": sys.version.split()[0], "pretrained_ocr": __version__}
    for name in ("torch", "torchvision", "transformers", "pillow", "paddleocr", "paddlepaddle", "paddlepaddle-gpu"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def load_selection(args):
    samples, skipped = read_dataset(args.dataset)
    selected = select_samples(samples, args.start_line, args.limit)
    print(f"Dataset: {args.dataset.resolve()}\nEligible: {len(samples)}; selected: {len(selected)}; skipped: {len(skipped)}")
    return selected, skipped


def validate(args):
    samples, skipped = load_selection(args)
    counts = {}
    for item in skipped:
        counts[item["reason"]] = counts.get(item["reason"], 0) + 1
    print(json.dumps({"skipped": counts, "first_sample": samples[0].sample_id}, ensure_ascii=True))
    if args.check_images:
        from PIL import Image
        for sample in samples:
            with Image.open(sample.image) as image:
                image.verify()
        print(f"Verified {len(samples)} image files.")
    return 0


def write_reports(output, rows, context):
    # Failures are empty predictions in the primary score so coverage cannot inflate accuracy.
    successful = [row for row in rows if row["status"] == "ok"]
    summary = dict(context)
    summary.update(
        completed_samples=len(rows), successful_samples=len(successful),
        failed_samples=len(rows) - len(successful),
        token_limit_samples=sum(row["hit_token_limit"] for row in rows),
        metrics=aggregate(rows), successful_only_metrics=aggregate(successful),
    )
    dump_json(output / "run_summary.json", summary)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    columns = ["sample_id", "status", "reference", "prediction", "cer", "wer", "exact_match",
               "reference_chars", "reference_words", "char_edits", "word_edits",
               "seconds", "hit_token_limit", "error"]
    with (output / "scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary


def run(args):
    spec = MODELS[args.model]
    output = (args.output or DEFAULT_OUTPUT / spec.key).resolve()
    check_output_path(args.dataset, output)
    samples, skipped = load_selection(args)
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.dry_run:
        print(f"Model: {spec.name}\nCheckpoint: {spec.checkpoint}\nOutput: {output}\nNo weights loaded.")
        return 0
    config = {"model": asdict(spec), "device": args.device, "dtype": args.dtype,
              "attention": args.attention, "max_new_tokens": args.max_new_tokens,
              "normalization": args.normalization, "dataset": str(args.dataset.resolve()),
              "implementation_version": __version__}
    config_path = output / "run_config.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise ValueError(f"Output already contains files. Use --resume or a new --output: {output}")
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing run configuration differs; use a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    dump_json(config_path, config)
    dump_json(output / "skipped_samples.json", skipped)
    fingerprints = {sample.sample_id: sample.fingerprint() for sample in samples}
    context = {
        "model": spec.key, "model_name": spec.name, "config": config,
        "dataset_fingerprint": digest(fingerprints), "selected_samples": len(samples),
        "skipped_samples": len(skipped), "normalization": args.normalization,
        "started_utc": datetime.now(timezone.utc).isoformat(), "packages": packages(),
        "status": "running", "failure_policy": "score_failed_samples_as_empty_predictions",
    }
    rows, backend = [], None
    # Write a non-complete status before loading weights; crashes must not leave a completed report.
    write_reports(output, rows, context)
    try:
        for index, sample in enumerate(samples, 1):
            sample_output = output / "samples" / sample.sample_id
            record_path = sample_output / "metadata.json"
            cached = None
            if args.resume and record_path.is_file():
                cached = json.loads(record_path.read_text(encoding="utf-8"))
                if cached["input_fingerprint"] != fingerprints[sample.sample_id]:
                    raise ValueError(f"Sample changed since the saved prediction: {sample.sample_id}. Use a new --output.")
            if cached and cached["status"] == "ok":
                row = cached
            else:
                if backend is None:
                    print(f"Loading {spec.name}: {spec.checkpoint}", flush=True)
                    backend = load_backend(spec, args)
                    context["backend"] = backend.info
                start = time.perf_counter()
                try:
                    prediction = backend.predict(sample.image)
                    row = {"prediction": prediction.text, "status": "ok", "error": "",
                           "generated_tokens": prediction.generated_tokens,
                           "hit_token_limit": prediction.hit_token_limit,
                           "confidence": prediction.confidence}
                except Exception as error:
                    row = {"prediction": "", "status": "error", "error": f"{type(error).__name__}: {error}",
                           "generated_tokens": None, "hit_token_limit": False, "confidence": None}
                    print(f"Failed {sample.sample_id}: {row['error']}", file=sys.stderr, flush=True)
                row.update(sample_id=sample.sample_id, image_path=str(sample.image), reference=sample.reference,
                           input_fingerprint=fingerprints[sample.sample_id], seconds=time.perf_counter() - start,
                           backend=backend.info)
                row.update(score(sample.reference, row["prediction"], args.normalization))
                sample_output.mkdir(parents=True, exist_ok=True)
                (sample_output / "prediction.txt").write_text(row["prediction"], encoding="utf-8")
                dump_json(record_path, row)
            rows.append(row)
            print(f"[{index}/{len(samples)}] {sample.sample_id}: {row['status']}" +
                  (" (token limit reached)" if row["hit_token_limit"] else ""), flush=True)
        context["status"] = "complete"
    except BaseException:
        context["status"] = "interrupted"
        raise
    finally:
        context["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary = write_reports(output, rows, context)
    metrics = summary["metrics"]
    print(f"CER: {percent(metrics['cer'])}; WER: {percent(metrics['wer'])}; exact lines: {percent(metrics['exact_match_rate'])}")
    print(f"Successful: {summary['successful_samples']}; failed: {summary['failed_samples']}\nReports: {output}")
    return 1 if summary["failed_samples"] else 0


def percent(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def compare(args):
    paths = sorted(args.results.glob("*/run_summary.json"))
    if not paths:
        raise ValueError(f"No */run_summary.json files in {args.results}")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(item["status"] != "complete" for item in summaries):
        raise ValueError("Some runs are incomplete; resume them before comparing")
    identities = {(item["dataset_fingerprint"], item["normalization"]) for item in summaries}
    if len(identities) != 1:
        raise ValueError("Runs used different samples, input contents, or normalization. Compare matching runs only.")
    summaries.sort(key=lambda item: item["metrics"]["cer"] if item["metrics"]["cer"] is not None else float("inf"))
    lines = ["| Model | Lines | Failed | CER | WER | Exact lines | Token limit |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in summaries:
        metric = item["metrics"]
        lines.append(f"| {item['model_name']} | {metric['samples']} | {item['failed_samples']} | "
                     f"{percent(metric['cer'])} | {percent(metric['wer'])} | {percent(metric['exact_match_rate'])} | "
                     f"{item['token_limit_samples']} |")
    report = "\n".join(lines) + "\n"
    (args.results / "comparison.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for command in ("validate", "run"):
        child = sub.add_parser(command)
        child.add_argument("--dataset", "--input-root", type=Path, default=DEFAULT_DATASET)
        child.add_argument("--start-line", type=int, default=1, help="1-based index after deterministic sorting/filtering")
        child.add_argument("--limit", type=int)
        if command == "validate":
            child.add_argument("--check-images", action="store_true")
            child.set_defaults(func=validate)
        else:
            child.add_argument("--model", choices=MODELS, required=True)
            child.add_argument("--output", type=Path)
            child.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N; Paddle also accepts gpu:N")
            child.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
            child.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
            child.add_argument("--max-new-tokens", type=int, default=512)
            child.add_argument("--normalization", choices=("raw", "standard", "arabic-light"), default="standard")
            child.add_argument("--resume", action="store_true")
            child.add_argument("--dry-run", action="store_true")
            child.set_defaults(func=run)
    child = sub.add_parser("compare")
    child.add_argument("--results", type=Path, default=DEFAULT_OUTPUT)
    child.set_defaults(func=compare)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
