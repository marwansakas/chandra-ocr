"""Frozen TRAIN-only membership for fitting diagnostics, independent of file order."""
import json
import random
import re
from collections import Counter

from .data import file_hash, resolve_relative
from .tokenizer import HARAKAT_V1, training_text


def page_id(record):
    match = re.match(r"^\d{6}_(\d{4})_", record["sample_id"])
    if not match:
        raise ValueError("Page-balanced selection requires the prepared Drive sample-ID format")
    return match.group(1)


def subset_stats(records):
    lengths = [len(training_text(r["text"], "no_harakat_v1")) for r in records]
    return {"samples": len(records), "pages": len({page_id(r) for r in records}),
            "tatweel_crops": sum("ـ" in r["text"] for r in records),
            "multiline_crops": sum("\n" in r["text"] for r in records),
            "harakat_crops": sum(any(c in HARAKAT_V1 for c in r["text"]) for r in records),
            "raw_characters": sum(len(r["text"]) for r in records),
            "derived_characters": sum(lengths), "min_length": min(lengths), "max_length": max(lengths),
            "length_bins": {"0_5": sum(n <= 5 for n in lengths), "6_15": sum(6 <= n <= 15 for n in lengths),
                            "16_30": sum(16 <= n <= 30 for n in lengths), "31_plus": sum(n > 30 for n in lengths)}}


def make_subset(train_records, count, seed, fingerprint, split_hash):
    if not 1 <= count <= len(train_records):
        raise ValueError("Subset count must be within the TRAIN partition size")
    records = sorted(train_records, key=lambda r: r["sample_id"])
    rng = random.Random(seed)
    candidates = records.copy()
    rng.shuffle(candidates)
    chosen, seen, page_counts = [], set(), Counter()

    def add(record):
        if record["sample_id"] not in seen and len(chosen) < count:
            chosen.append(record)
            seen.add(record["sample_id"])
            page_counts[page_id(record)] += 1

    # Reserve extremes and annotation cases, using TRAIN labels only.
    add(min(records, key=lambda r: len(r["text"])))
    add(max(records, key=lambda r: len(r["text"])))
    for predicate in (lambda s: "ـ" in s, lambda s: "\n" in s, lambda s: any(c in HARAKAT_V1 for c in s)):
        example = next((r for r in candidates if predicate(r["text"])), None)
        if example is not None:
            add(example)
    while len(chosen) < count:
        remaining = [r for r in candidates if r["sample_id"] not in seen]
        add(min(remaining, key=lambda r: page_counts[page_id(r)]))
    return {"schema_version": 1, "partition": "train", "algorithm": "page_balanced_reserved_cases_v1",
            "seed": seed, "dataset_fingerprint": fingerprint, "split_lock_sha256": split_hash,
            "sample_ids": [r["sample_id"] for r in chosen], "stats": subset_stats(chosen),
            "notes": "Fitting diagnostic: reserved shortest/longest, tatweel, multiline and harakat examples, then seeded least-represented-page selection. Pages are not writer identities. Not a population-weighted test sample."}


def read_subset(path, train_records, fingerprint, split_hash, count):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("partition") != "train":
        raise ValueError("Unsupported TRAIN subset manifest")
    if manifest.get("dataset_fingerprint") != fingerprint or manifest.get("split_lock_sha256") != split_hash:
        raise ValueError("Subset belongs to different frozen data/split")
    ids = manifest["sample_ids"]
    if len(ids) != count or len(set(ids)) != len(ids):
        raise ValueError("Subset count or unique membership differs from --overfit-samples")
    lookup = {r["sample_id"]: r for r in train_records}
    if any(sample_id not in lookup for sample_id in ids):
        raise ValueError("Subset contains a sample outside TRAIN")
    return [lookup[sample_id] for sample_id in ids]


def run_training_records(run_root, run, train_records):
    """Restore exact fitted membership; legacy first-N runs retain their behavior."""
    if file_hash(run_root / "splits" / "split_lock.json") != run["split_lock_sha256"]:
        raise ValueError("Run's frozen split differs from its training provenance")
    count = run["training_options"].get("overfit_samples")
    expected_hash = run["training_options"].get("training_subset_sha256")
    if expected_hash:
        path = resolve_relative(run_root, "training_subset.json")
        if file_hash(path) != expected_hash:
            raise ValueError("Saved training subset changed")
        return read_subset(path, train_records, run["dataset_fingerprint"], run["split_lock_sha256"], count)
    return train_records[:count] if count else train_records
