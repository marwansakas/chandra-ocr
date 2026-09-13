"""Evaluate named prediction folders against one manifest with identical rules."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pretrained_ocr.metrics import aggregate, normalize, score


def evaluate(manifest: Path, predictions: dict[str, Path], output: Path):
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    ids = [row["sample_id"] for row in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Test manifest must contain unique samples and cannot be empty.")
    for sample_id in ids:
        if not isinstance(sample_id, str) or sample_id in {".", ".."} or "/" in sample_id or "\\" in sample_id:
            raise ValueError(f"Invalid sample ID: {sample_id!r}")
    # Require a complete common test set rather than silently excluding failures.
    for name, root in predictions.items():
        missing = [sid for sid in ids if not (root / sid / "prediction.txt").is_file()]
        if missing:
            raise ValueError(f"{name}: missing {len(missing)}/{len(ids)} predictions; first: {missing[0]}")
    details, summaries = [], []
    for name, root in predictions.items():
        rows = []
        for record in records:
            prediction = (root / record["sample_id"] / "prediction.txt").read_text(encoding="utf-8")
            result = score(record["text"], prediction, "arabic-light")
            rows.append(result)
            details.append(dict(model=name, sample_id=record["sample_id"],
                                reference=normalize(record["text"]), prediction=normalize(prediction), **result))
        summary = aggregate(rows)
        summary["character_accuracy"] = max(0., 1 - summary["cer"]) if summary["cer"] is not None else None
        summary["word_accuracy"] = max(0., 1 - summary["wer"]) if summary["wer"] is not None else None
        summaries.append(dict(model=name, **summary))
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("summary.csv", summaries), ("per_line.csv", details)):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "test_manifest": str(manifest.resolve()),
        "test_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "normalization": "arabic-light: NFKC, remove Arabic combining marks and tatweel, collapse whitespace; preserve hamza letters",
        "alignment": "character alignment: match +2, mismatch -1, gap -1; divide by twice the longer line; clip to [0,1]",
        "aggregation": "corpus CER/WER (may exceed 1), mean per-line alignment and exact match; rates are fractions",
        "prediction_roots": {name: str(root.resolve()) for name, root in predictions.items()},
        "models": summaries,
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=PATH",
                        help="Repeat per model; PATH contains sample_id/prediction.txt. For pretrained adapters use their samples subfolder.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    predictions = {}
    for value in args.prediction:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in predictions:
            parser.error("Each --prediction must have a unique NAME=PATH.")
        predictions[name] = Path(path)
    try:
        report = evaluate(args.test_manifest, predictions, args.output_dir)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    for row in report["models"]:
        print(f"{row['model']}: CER={row['cer']}, WER={row['wer']}, exact={row['exact_match_rate']}")
    print(f"Reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
