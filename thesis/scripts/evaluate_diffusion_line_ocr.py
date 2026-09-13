"""Compare saved OCR outputs on one frozen partition, with literal and historical metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thesis.diffusion.data import file_hash, json_bytes, load_frozen, resolve_relative
from thesis.diffusion.metrics import aggregate, correction_changes, score_pair
from thesis.diffusion.tokenizer import read_literal


def read_prediction(root, sample_id):
    path = resolve_relative(root, sample_id + "/prediction.txt")
    metadata_path = path.parent / "metadata.json"
    metadata = {}
    try:
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "failed":
            return "", "failed", metadata
        if not path.is_file():
            return "", "missing", metadata
        return read_literal(path), "completed", metadata
    except (OSError, ValueError, UnicodeError) as exc:
        return "", "failed", {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--predictions", action="append", required=True, metavar="NAME=PATH", help="Repeat for TrOCR, diffusion and other saved outputs")
    parser.add_argument("--before-root", type=Path, help="Optional earlier complete predictions for aligned correction/damage counts")
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW evaluation report directory")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    model_roots = {}
    for value in args.predictions:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in model_roots:
            parser.error("Each --predictions must have a unique NAME=PATH")
        model_roots[name] = Path(path)
    partitions, lock, tokenizer = load_frozen(args.split_root, args.dataset_root, verify_data=True)
    records = partitions[args.partition]
    rows, summaries = [], {}
    for model, root in model_roots.items():
        pairs, complete_pairs, damage = [], [], []
        timings, memories = [], []
        model_rows = []
        for record in records:
            text, status, metadata = read_prediction(root, record["sample_id"])
            pair = (record["text"], text)
            pairs.append(pair)
            if status == "completed":
                complete_pairs.append(pair)
                if metadata.get("preprocess_and_inference_seconds") is not None:
                    timings.append(metadata["preprocess_and_inference_seconds"])
                if metadata.get("peak_cuda_memory_bytes") is not None:
                    memories.append(metadata["peak_cuda_memory_bytes"])
            row = {"model": model, "sample_id": record["sample_id"], "status": status, "reference": record["text"], "prediction": text,
                   "strict": score_pair(*pair), "historical_normalized": score_pair(*pair, historical=True),
                   "has_eos": metadata.get("has_eos"), "error": metadata.get("error"),
                   "capacity_overflow": len(record["text"]) + 1 > lock["capacity"],
                   "oov_codepoints": [f"U+{ord(c):04X}" for c in sorted(set(record["text"]) - set(tokenizer.characters))]}
            if args.before_root:
                before, before_status, _ = read_prediction(args.before_root, record["sample_id"])
                if before_status == "completed" and status == "completed":
                    row["correction"] = correction_changes(record["text"], before, text)
                    damage.append(row["correction"])
            model_rows.append(row)
        summaries[model] = {"expected": len(records), "completed": len(complete_pairs),
                            "missing": sum(r["status"] == "missing" for r in model_rows),
                            "failed": sum(r["status"] == "failed" for r in model_rows),
                            "no_eos": sum(r["has_eos"] is False for r in model_rows),
                            "capacity_overflow": sum(r["capacity_overflow"] for r in model_rows),
                            "oov_lines": sum(bool(r["oov_codepoints"]) for r in model_rows),
                            "strict_all_expected": aggregate(pairs), "historical_normalized_all_expected": aggregate(pairs, historical=True),
                            "strict_completed_only": aggregate(complete_pairs),
                            "latency_seconds": {"measured_lines": len(timings), "mean": statistics.mean(timings),
                                                "median": statistics.median(timings), "p95": sorted(timings)[math.ceil(0.95 * len(timings)) - 1]} if timings else None,
                            "peak_cuda_memory_bytes": max(memories) if memories else None,
                            "runtime_source": str(root / "run_summary.json") if (root / "run_summary.json").is_file() else None,
                            "correction": {"compared_lines": len(damage), "counts": {key: sum(d[key] for d in damage) for key in damage[0]} if damage else {}}}
        rows.extend(model_rows)
    report = {"split_lock_sha256": file_hash(args.split_root / "split_lock.json"), "dataset_fingerprint": lock["dataset_fingerprint"],
              "partition": args.partition, "models": summaries,
              "policy": {"strict_cer": "literal code points including spaces; no normalization",
                         "strict_wer": "Python str.split() whitespace words; includes digits and punctuation",
                         "historical": "common.normalize_text_for_evaluation + common.tokenize_words (omits digit words)",
                         "coverage": "all-expected metrics count missing/failed predictions as empty; completed-only also reported",
                         "correction": "aligned reference character matches before/after; ties diagonal/delete/insert; insertions measured by char_error_delta"}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "evaluation.json").write_bytes(json_bytes(report))
    (args.output_dir / "samples.jsonl").write_bytes(b"".join(json_bytes(row) for row in rows))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
