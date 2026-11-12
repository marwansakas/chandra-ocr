"""Bind PaddleOCR observations to one immutable diffusion split partition.

Paddle text is an input feature only. Ground-truth text remains the sole target.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .data import file_hash, resolve_relative
from .tokenizer import HARAKAT_V1


def _input_fingerprint(dataset_root: Path, record: dict) -> str:
    image = resolve_relative(dataset_root, record["image_path"])
    label = resolve_relative(dataset_root, record["label_path"])
    reference = label.read_bytes().decode("utf-8-sig", errors="strict")
    digest = hashlib.sha256(image.read_bytes())
    digest.update(b"\x00")
    digest.update(reference.encode("utf-8"))
    return digest.hexdigest()


def bind_paddle_predictions(
    prediction_root: Path,
    records: list[dict],
    dataset_root: Path,
    split_manifest: Path,
) -> tuple[list[dict], dict]:
    """Return records with verified Paddle text/confidence attached."""
    prediction_root = prediction_root.resolve()
    dataset_root = dataset_root.resolve()
    required = {
        name: prediction_root / name
        for name in ("run_config.json", "run_summary.json", "predictions.jsonl")
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError("Missing Paddle conditioning files: " + ", ".join(missing))
    config = json.loads(required["run_config.json"].read_text(encoding="utf-8"))
    summary = json.loads(required["run_summary.json"].read_text(encoding="utf-8"))
    model = config.get("model") or {}
    if model.get("key") != "paddle-arabic" or model.get("checkpoint") != "arabic_PP-OCRv5_mobile_rec":
        raise ValueError("Conditioning source is not the expected PaddleOCR Arabic checkpoint")
    if config.get("normalization") != "no-harakat-v1":
        raise ValueError("Paddle conditioning must use no-harakat-v1")
    configured_dataset = Path(config.get("dataset", "")).resolve()
    if configured_dataset != dataset_root:
        raise ValueError("Paddle conditioning dataset root differs from the frozen dataset")
    manifest = config.get("test_manifest") or {}
    expected_manifest_hash = file_hash(split_manifest)
    if manifest.get("sha256") != expected_manifest_hash:
        raise ValueError("Paddle conditioning manifest differs from the frozen partition")
    if summary.get("status") != "complete" or summary.get("failed_samples") != 0:
        raise ValueError("Paddle conditioning run is incomplete or contains failures")
    if summary.get("selected_samples") != len(records) or summary.get("completed_samples") != len(records):
        raise ValueError("Paddle conditioning coverage differs from the frozen partition")

    rows = {}
    for line_number, line in enumerate(required["predictions.jsonl"].read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in rows:
            raise ValueError(f"Invalid or duplicate Paddle sample at line {line_number}")
        rows[sample_id] = row
    expected_ids = {record["sample_id"] for record in records}
    if set(rows) != expected_ids:
        raise ValueError("Paddle conditioning sample IDs differ from the frozen partition")

    bound = []
    for record in records:
        row = rows[record["sample_id"]]
        if row.get("status") != "ok" or not isinstance(row.get("prediction"), str):
            raise ValueError(f"Invalid Paddle prediction for {record['sample_id']}")
        confidence = row.get("confidence")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid Paddle confidence for {record['sample_id']}")
        if row.get("input_fingerprint") != _input_fingerprint(dataset_root, record):
            raise ValueError(f"Paddle input fingerprint mismatch for {record['sample_id']}")
        bound.append({**record, "conditioning_text": row["prediction"],
                      "conditioning_confidence": float(confidence)})

    provenance = {
        "kind": "paddle_text_v1",
        "model": model,
        "normalization": config["normalization"],
        "partition_manifest_sha256": expected_manifest_hash,
        "selected_samples": len(bound),
        "files": {name: file_hash(path) for name, path in required.items()},
        "input_binding": "SHA256(image bytes + NUL + literal ground-truth file text)",
        "target_use": "Paddle prediction and confidence are conditioning only; Paddle reference/score fields are ignored",
    }
    return bound, provenance


def conditioning_encoding_report(records: list[dict], tokenizer, capacity: int) -> dict:
    """Describe bounded/OOV observation encoding without exposing target text."""
    oov = set()
    oov_lines = truncated = 0
    for record in records:
        text = "".join(c for c in record["conditioning_text"] if c not in HARAKAT_V1)
        unknown = set(text) - set(tokenizer.characters)
        if unknown:
            oov_lines += 1
            oov.update(unknown)
        truncated += len(text) + 1 > capacity
    return {
        "capacity": capacity,
        "truncated_observations": truncated,
        "oov_observation_lines": oov_lines,
        "oov_codepoints": [f"U+{ord(c):04X}" for c in sorted(oov, key=ord)],
        "oov_policy": "map to MASK observation token",
        "overflow_policy": "truncate Paddle observation only to capacity-1 then append EOS; ground truth is unchanged",
    }
