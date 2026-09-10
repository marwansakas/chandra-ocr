"""Validate generated line pairs and attach them to the training partition only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from generate_synthetic_lines import clean_text, text_hash


def load_synthetic_records(root: Path, train, test, record_type, limit=None):
    root = root.resolve()
    if limit is not None and limit < 1:
        raise ValueError("--synthetic-limit must be positive.")
    config = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if config.get("complete") is not True or config.get("split") != "train":
        raise ValueError("Synthetic generation is incomplete or not marked training-only.")
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported synthetic dataset format.")
    own = {row.sample_id: row for row in train}
    held_out = {row.sample_id for row in test}
    seen = set(own) | held_out
    records = []
    original_hashes = {}
    total = 0
    for raw in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        total += 1
        row = json.loads(raw)
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or sample_id in seen:
            raise ValueError(f"Duplicate synthetic/real sample ID: {sample_id}")
        seen.add(sample_id)
        image = (root / row["image_path"]).resolve()
        if root not in image.parents or image.name != "image.png":
            raise ValueError("Synthetic image paths must stay inside the generated dataset.")
        label = image.parent / "ground_truth.txt"
        metadata = json.loads((image.parent / "metadata.json").read_text(encoding="utf-8"))
        for key in ("sample_id", "synthetic", "split", "generation_method", "source_sample_id",
                    "source_text_sha256", "text_sha256", "source_image_sha256"):
            if metadata.get(key) != row.get(key):
                raise ValueError(f"Metadata disagrees with the manifest for {sample_id}: {key}")
        if row.get("synthetic") is not True or row.get("split") != "train":
            raise ValueError(f"Sample is not marked synthetic training data: {sample_id}")
        if not image.is_file():
            raise ValueError(f"Missing synthetic image: {image}")
        text = label.read_text(encoding="utf-8").strip()
        if text != row["text"] or text_hash(text) != row["text_sha256"]:
            raise ValueError(f"Image label/manifest mismatch: {sample_id}")
        source_id = row.get("source_sample_id")
        if row["generation_method"] == "augment" and not source_id:
            raise ValueError("Augmented handwriting must identify its source training sample.")
        if source_id:
            if source_id in held_out or source_id not in own:
                raise ValueError(
                    f"{sample_id} derives from {source_id}, which is not in the current training split. "
                    "Use the original split settings or regenerate from the current train.jsonl."
                )
            if text_hash(clean_text(own[source_id].text)) != row["source_text_sha256"]:
                raise ValueError(f"Source transcription changed: {source_id}")
            if row["generation_method"] == "augment":
                if source_id not in original_hashes:
                    original_hashes[source_id] = hashlib.sha256(
                        Path(own[source_id].image_path).read_bytes()).hexdigest()
                if original_hashes[source_id] != row.get("source_image_sha256"):
                    raise ValueError(f"Source image changed: {source_id}")
        if limit is None or len(records) < limit:
            records.append(record_type(
                sample_id=sample_id, page_id="synthetic/train", page_dir=str(image.parent),
                rank=2_000_000 + total, line_index=total, image_path=str(image), text=text,
                bbox=[], polygon=[], confidence=None, status="synthetic_train",
            ))
    if total != config["count"] or not records:
        raise ValueError("Synthetic dataset is empty or its sample count does not match run.json.")
    return records
