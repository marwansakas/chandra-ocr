"""Read-only audit and content-verified, portable immutable partitions."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import unicodedata

from PIL import Image

from .tokenizer import CharacterTokenizer, POLICY, read_literal

PARTITIONS = ("train", "validation", "test")


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_relative(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if not value or "\\" in value or ":" in value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected portable relative path: {value!r}")
    path = (root / Path(*relative.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset root: {value}")
    return path


def normalize_exclusion_ids(values) -> list[str]:
    """Accept only portable relative sample IDs; never resolve them for deletion."""
    result = []
    for raw in values or ():
        value = raw.strip() if isinstance(raw, str) else None
        path = PurePosixPath(value) if value else None
        if (not value or "\\" in value or ":" in value or path.is_absolute() or ".." in path.parts
                or not path.parts or any(part in ("", ".") for part in path.parts)):
            raise ValueError(f"Invalid exclusion sample ID: {raw!r}")
        result.append(value)
    if len(result) != len(set(result)):
        raise ValueError("Duplicate exclusion sample ID")
    return sorted(result)


def image_info(path: Path) -> dict:
    with Image.open(path) as original:
        image = original.convert("RGB")
        width, height = image.size
        pixels = digest_bytes(json_bytes([width, height, "RGB"]) + image.tobytes())
    return {"width": width, "height": height, "image_sha256": file_hash(path), "pixels_sha256": pixels}


def distribution(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {"count": len(values), "min": ordered[0], "max": ordered[-1],
            **{f"p{p}": ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)] for p in (50, 90, 95, 99)}}


def audit_dataset(root: Path, writer_map: dict[str, str] | None = None, *, excluded_sample_ids=(),
                  allowed_sample_ids: set[str] | None = None) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    writer_map = writer_map or {}
    excluded_ids = set(normalize_exclusion_ids(excluded_sample_ids))
    folders = {p.parent for pattern in ("*/image.png", "*/line_*/image.png", "*/ground_truth.txt", "*/line_*/ground_truth.txt") for p in root.glob(pattern)}
    records, issues, accepted_exclusions = [], [], []
    chars = Counter()
    for folder in sorted(folders):
        sample_id = folder.relative_to(root).as_posix()
        try:
            if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
                raise ValueError("folder is absent from the verified source inventory")
            image = resolve_relative(root, sample_id + "/image.png")
            label = resolve_relative(root, sample_id + "/ground_truth.txt")
            if not image.is_file() or not label.is_file():
                raise ValueError("missing image.png or ground_truth.txt")
            text = read_literal(label)
            if sample_id in excluded_ids:
                if text.strip() != "≠":
                    raise ValueError("exclusion is allowed only for a standalone ≠ label")
                accepted_exclusions.append({"sample_id": sample_id, "reason": "standalone_not_equal"})
                continue
            if not text.strip():
                raise ValueError("empty label")
            if "ignore" in text.casefold():
                raise ValueError("label contains ignore (existing dataset exclusion convention)")
            metadata_path = folder / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object")
            writer = writer_map.get(sample_id)
            if writer is not None and (not isinstance(writer, str) or not writer.strip()):
                raise ValueError("writer mapping must contain nonempty string IDs")
            record = {"sample_id": sample_id, "image_path": image.relative_to(root.resolve()).as_posix(),
                      "label_path": label.relative_to(root.resolve()).as_posix(), "text": text,
                      "label_sha256": file_hash(label), "writer_id": writer,
                      "metadata_writer_candidate": metadata.get("writer_id", metadata.get("student_id")),
                      "metadata_sha256": file_hash(metadata_path) if metadata_path.is_file() else None,
                      **image_info(image)}
            records.append(record)
            chars.update(text)
        except (ValueError, OSError, UnicodeError) as exc:
            issues.append({"sample_id": sample_id, "reason": str(exc)})
    duplicate_map = defaultdict(list)
    for record in records:
        duplicate_map[record["pixels_sha256"]].append(record)
    duplicates = [{"pixels_sha256": key, "sample_ids": [r["sample_id"] for r in group],
                   "conflicting_labels": len({r["text"] for r in group}) > 1}
                  for key, group in duplicate_map.items() if len(group) > 1]
    fingerprint = digest_bytes(json_bytes({"policy": POLICY, "records": records, "issues": issues}))
    return {"schema_version": 1, "policy": POLICY, "fingerprint": fingerprint,
            "discovered": len(folders), "labeled_records": len(records), "issues": issues,
            "records": records, "duplicates": duplicates,
            "excluded_records": accepted_exclusions,
            "missing_exclusion_ids": sorted(excluded_ids - {folder.relative_to(root).as_posix() for folder in folders}),
            "unmapped_writers": [r["sample_id"] for r in records if not r["writer_id"]],
            "unused_writer_map_keys": sorted(set(writer_map) - {r["sample_id"] for r in records}),
            "writer_counts": dict(Counter(r["writer_id"] for r in records if r["writer_id"])),
            "lengths_codepoints": distribution([len(r["text"]) for r in records]),
            "widths": distribution([r["width"] for r in records]),
            "heights": distribution([r["height"] for r in records]),
            "characters": [{"character": c, "codepoint": f"U+{ord(c):04X}", "name": unicodedata.name(c, "UNNAMED"), "count": n}
                           for c, n in sorted(chars.items(), key=lambda pair: ord(pair[0]))]}


def check_partitions(partitions: dict[str, list[dict]]):
    owners = {key: {} for key in ("sample_id", "writer_id", "pixels_sha256")}
    for split, records in partitions.items():
        if not records:
            raise ValueError(f"Empty partition: {split}")
        local_ids = set()
        for record in records:
            if record["sample_id"] in local_ids:
                raise ValueError(f"Duplicate sample ID in {split}: {record['sample_id']}")
            local_ids.add(record["sample_id"])
            for key, seen in owners.items():
                value = record[key]
                if not value:
                    raise ValueError(f"Missing {key}: {record['sample_id']}")
                if value in seen and seen[value] != split:
                    raise ValueError(f"Partition leakage by {key}: {value}")
                seen[value] = split


def freeze_splits(audit: dict, output: Path, *, seed=42, train_ratio=0.8, validation_ratio=0.1,
                  capacity: int | None = None, extra_inventory="", allow_exclusions=False):
    if output.exists():
        raise FileExistsError(f"Immutable split directory already exists: {output}")
    if not (0 < train_ratio < 1 and 0 < validation_ratio < 1 and train_ratio + validation_ratio < 1):
        raise ValueError("Ratios must leave nonzero train, validation and test partitions")
    if audit["issues"] and not allow_exclusions:
        raise ValueError("Audit has excluded records; review audit and explicitly use --allow-exclusions")
    if audit["unmapped_writers"]:
        raise ValueError("Explicit writer map required for every usable sample; filename/notebook guesses are not verified writers")
    if any(d["conflicting_labels"] for d in audit["duplicates"]):
        raise ValueError("Identical images have conflicting labels; resolve annotations before freezing")
    # Union writers connected by identical pixels, keeping every component together.
    writers = sorted(audit["writer_counts"])
    parent = {w: w for w in writers}

    def find(w):
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w

    seen_pixels = {}
    for record in audit["records"]:
        writer = record["writer_id"]
        pixel = record["pixels_sha256"]
        if pixel in seen_pixels:
            parent[find(writer)] = find(seen_pixels[pixel])
        seen_pixels[pixel] = writer
    groups = defaultdict(list)
    for record in audit["records"]:
        groups[find(record["writer_id"])].append(record)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("At least three independent writer/duplicate groups are required")
    random.Random(seed).shuffle(keys)
    n_train = min(len(keys) - 2, max(1, round(len(keys) * train_ratio)))
    n_val = min(len(keys) - n_train - 1, max(1, round(len(keys) * validation_ratio)))
    split_keys = (keys[:n_train], keys[n_train:n_train + n_val], keys[n_train + n_val:])
    partitions = {name: sorted([r for key in group for r in groups[key]], key=lambda r: r["sample_id"])
                  for name, group in zip(PARTITIONS, split_keys)}
    check_partitions(partitions)
    # Capacity selection never inspects test lengths. Test overflow is reported by evaluation.
    needed = max(len(r["text"]) + 1 for name in ("train", "validation") for r in partitions[name])
    capacity = needed if capacity is None else capacity
    if capacity < needed:
        raise ValueError(f"Capacity overflow in train/validation: need at least {needed}, got {capacity}")
    tokenizer = CharacterTokenizer.build((r["text"] for r in partitions["train"]), extra_inventory)
    for name in ("train", "validation"):
        for record in partitions[name]:
            try:
                tokenizer.encode(record["text"], capacity)
            except ValueError as exc:
                raise ValueError(f"{name}/{record['sample_id']}: {exc}") from exc
    files = {f"{name}.jsonl": b"".join(json_bytes(r) for r in records) for name, records in partitions.items()}
    files["tokenizer.json"] = json_bytes(tokenizer.as_dict())
    files["audit.json"] = json_bytes(audit)
    lock = {"schema_version": 1, "dataset_fingerprint": audit["fingerprint"], "policy": POLICY,
            "seed": seed, "train_ratio": train_ratio, "validation_ratio": validation_ratio,
            "split_unit": "explicit writer IDs joined by exact RGB pixel duplicates",
            "capacity": capacity, "counts": {k: len(v) for k, v in partitions.items()},
            "hashes": {name: digest_bytes(data) for name, data in files.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (output / name).write_bytes(data)
    (output / "split_lock.json").write_bytes(json_bytes(lock))
    return lock


def check_line_pilot(partitions):
    """The explicitly requested pilot splits individual lines, without writer isolation."""
    if not partitions["train"] or not partitions["test"] or partitions["validation"]:
        raise ValueError("A line pilot requires nonempty train/test and an empty validation partition")
    seen_ids, seen_paths = set(), set()
    for name in ("train", "test"):
        for record in partitions[name]:
            if record["sample_id"] in seen_ids or record["image_path"] in seen_paths:
                raise ValueError("The same sample/path occurs more than once in the line pilot")
            seen_ids.add(record["sample_id"])
            seen_paths.add(record["image_path"])


def check_line_split(partitions):
    """Validate a crop-level split with an optional validation partition."""
    if not partitions["train"] or not partitions["validation"] or not partitions["test"]:
        raise ValueError("A line split requires nonempty train, validation and test partitions")
    seen_ids, seen_paths, pixel_owners = set(), set(), {}
    for name in PARTITIONS:
        for record in partitions[name]:
            if record["sample_id"] in seen_ids or record["image_path"] in seen_paths:
                raise ValueError("The same sample/path occurs more than once in the line split")
            seen_ids.add(record["sample_id"])
            seen_paths.add(record["image_path"])
            pixel = record["pixels_sha256"]
            if pixel in pixel_owners and pixel_owners[pixel] != name:
                raise ValueError(f"Cross-partition duplicate pixels: {pixel}")
            pixel_owners[pixel] = name


def split_report(partitions: dict[str, list[dict]]) -> dict:
    """Describe observable crop/page overlap without treating pages as writers."""
    pixel_owners, page_owners = defaultdict(set), defaultdict(set)
    for name, records in partitions.items():
        for record in records:
            pixel_owners[record["pixels_sha256"]].add(name)
            match = re.match(r"^\d{6}_(\d{4})_", record["sample_id"])
            if match:
                page_owners[match.group(1)].add(name)
    cross_pixels = sorted(pixel for pixel, owners in pixel_owners.items() if len(owners) > 1)
    overlapping_pages = {page: sorted(owners) for page, owners in page_owners.items() if len(owners) > 1}
    return {
        "cross_split_duplicate_pixel_hashes": cross_pixels,
        "cross_split_duplicate_pixel_count": len(cross_pixels),
        "pages_per_partition": {name: len({match.group(1) for r in records
                                             if (match := re.match(r"^\d{6}_(\d{4})_", r["sample_id"]))})
                                for name, records in partitions.items()},
        "pages_in_multiple_partitions": len(overlapping_pages),
        "page_overlap": overlapping_pages,
        "length_codepoints_by_partition": {name: distribution([len(r["text"]) for r in records])
                                             for name, records in partitions.items()},
        "page_note": "Page IDs are source-folder identifiers, not verified child or writer identities.",
    }


def freeze_line_split(audit: dict, output: Path, *, train_ratio=0.7, validation_ratio=0.1,
                      seed=42, capacity=None, extra_inventory=""):
    """Freeze a shuffled crop-level split; validation is held out from training."""
    if output.exists():
        raise FileExistsError(f"Immutable split directory already exists: {output}")
    if audit["issues"]:
        raise ValueError("Audit has excluded records; resolve them before freezing")
    if not (0 < train_ratio < 1 and 0 < validation_ratio < 1 and train_ratio + validation_ratio < 1):
        raise ValueError("Ratios must leave nonzero train, validation and test partitions")
    if any(row["conflicting_labels"] for row in audit["duplicates"]):
        raise ValueError("Identical images have conflicting labels; resolve annotations before freezing")
    records = sorted(audit["records"], key=lambda r: r["sample_id"])
    pixel_groups = defaultdict(list)
    for record in records:
        pixel_groups[record["pixels_sha256"]].append(record)
    groups = [sorted(group, key=lambda r: r["sample_id"]) for _, group in sorted(pixel_groups.items())]
    random.Random(seed).shuffle(groups)
    if len(groups) < 3:
        raise ValueError("At least three distinct pixel groups are required")
    n_train = max(1, min(len(records) - 2, math.floor(len(records) * train_ratio)))
    n_val = max(1, min(len(records) - n_train - 1, math.floor(len(records) * validation_ratio)))
    targets = {"train": n_train, "validation": n_val, "test": len(records) - n_train - n_val}
    partitions = {name: [] for name in PARTITIONS}
    for group in groups:
        # A duplicate group is indivisible.  With unique images this produces the
        # exact requested crop counts; otherwise the small documented deviation is
        # preferable to leaking a pixel across partitions.
        name = max(PARTITIONS, key=lambda candidate: (targets[candidate] - len(partitions[candidate]), -PARTITIONS.index(candidate)))
        partitions[name].extend(group)
    partitions = {name: sorted(rows, key=lambda r: r["sample_id"]) for name, rows in partitions.items()}
    check_line_split(partitions)
    needed = max(len(r["text"]) + 1 for name in ("train", "validation") for r in partitions[name])
    capacity = needed if capacity is None else capacity
    tokenizer = CharacterTokenizer.build((r["text"] for r in partitions["train"]), extra_inventory)
    for name in ("train", "validation"):
        for record in partitions[name]:
            tokenizer.encode(record["text"], capacity)
    files = {f"{name}.jsonl": b"".join(json_bytes(r) for r in rows) for name, rows in partitions.items()}
    files["tokenizer.json"] = json_bytes(tokenizer.as_dict())
    files["audit.json"] = json_bytes(audit)
    lock = {"schema_version": 1, "dataset_fingerprint": audit["fingerprint"], "policy": POLICY,
            "seed": seed, "train_ratio": train_ratio, "validation_ratio": validation_ratio,
            "split_unit": "line_split", "evaluation_policy": "validation_loss_and_final_test",
            "writer_isolation": False, "rounding": "floor(N * ratio), remainder to test",
            **split_report(partitions), "capacity": capacity,
            "counts": {name: len(rows) for name, rows in partitions.items()},
            "hashes": {name: digest_bytes(data) for name, data in files.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (output / name).write_bytes(data)
    (output / "split_lock.json").write_bytes(json_bytes(lock))
    return lock


def freeze_line_pilot(audit: dict, output: Path, *, train_ratio=0.9, seed=42,
                      capacity=None, extra_inventory=""):
    """Freeze the user-requested two-way line split; the test set never selects checkpoints."""
    if output.exists():
        raise FileExistsError(f"Immutable split directory already exists: {output}")
    if not 0 < train_ratio < 1 or len(audit["records"]) < 2:
        raise ValueError("Need at least two usable lines and a ratio strictly between 0 and 1")
    records = sorted(audit["records"], key=lambda row: row["sample_id"])
    random.Random(seed).shuffle(records)
    cut = max(1, min(len(records) - 1, math.floor(len(records) * train_ratio)))
    partitions = {"train": sorted(records[:cut], key=lambda r: r["sample_id"]), "validation": [],
                  "test": sorted(records[cut:], key=lambda r: r["sample_id"])}
    check_line_pilot(partitions)
    needed = max(len(r["text"]) + 1 for r in partitions["train"])
    capacity = needed if capacity is None else capacity
    if capacity < needed:
        raise ValueError(f"Training label overflow: need capacity {needed}, got {capacity}")
    tokenizer = CharacterTokenizer.build((r["text"] for r in partitions["train"]), extra_inventory)
    for record in partitions["train"]:
        tokenizer.encode(record["text"], capacity)
    train_pixels = {r["pixels_sha256"] for r in partitions["train"]}
    test_pixels = {r["pixels_sha256"] for r in partitions["test"]}
    files = {f"{name}.jsonl": b"".join(json_bytes(r) for r in rows) for name, rows in partitions.items()}
    files["tokenizer.json"] = json_bytes(tokenizer.as_dict())
    files["audit.json"] = json_bytes(audit)
    lock = {"schema_version": 1, "dataset_fingerprint": audit["fingerprint"], "policy": POLICY,
            "seed": seed, "train_ratio": train_ratio, "validation_ratio": 0,
            "split_unit": "line_pilot", "evaluation_policy": "fixed_epochs_final_checkpoint_no_validation",
            "writer_isolation": False, "rounding": "floor(N * train_ratio), remainder to test",
            "cross_split_duplicate_pixel_hashes": sorted(train_pixels & test_pixels),
            "capacity": capacity, "counts": {name: len(rows) for name, rows in partitions.items()},
            "hashes": {name: digest_bytes(data) for name, data in files.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (output / name).write_bytes(data)
    (output / "split_lock.json").write_bytes(json_bytes(lock))
    return lock


def load_frozen(split_root: Path, dataset_root: Path, *, verify_data=True):
    lock = json.loads((split_root / "split_lock.json").read_text(encoding="utf-8"))
    if lock["schema_version"] != 1 or lock["policy"] != POLICY:
        raise ValueError("Unsupported split policy/version")
    for name in [*(f"{p}.jsonl" for p in PARTITIONS), "tokenizer.json", "audit.json"]:
        if file_hash(split_root / name) != lock["hashes"][name]:
            raise ValueError(f"Frozen file changed: {name}")
    partitions = {name: [json.loads(line) for line in (split_root / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                  for name in PARTITIONS}
    if lock.get("split_unit") == "line_pilot":
        if lock.get("evaluation_policy") != "fixed_epochs_final_checkpoint_no_validation":
            raise ValueError("Unsupported line-pilot evaluation policy")
        check_line_pilot(partitions)
    elif lock.get("split_unit") == "line_split":
        if lock.get("evaluation_policy") != "validation_loss_and_final_test":
            raise ValueError("Unsupported line-split evaluation policy")
        check_line_split(partitions)
    else:
        check_partitions(partitions)
    if verify_data:
        for records in partitions.values():
            for record in records:
                for field, hash_field in (("image_path", "image_sha256"), ("label_path", "label_sha256")):
                    path = resolve_relative(dataset_root, record[field])
                    if not path.is_file() or file_hash(path) != record[hash_field]:
                        raise ValueError(f"Missing or changed frozen sample: {record['sample_id']} ({field})")
                if read_literal(resolve_relative(dataset_root, record["label_path"])) != record["text"]:
                    raise ValueError(f"Frozen text mismatch: {record['sample_id']}")
                metadata = resolve_relative(dataset_root, record["sample_id"] + "/metadata.json")
                actual_metadata_hash = file_hash(metadata) if metadata.is_file() else None
                if actual_metadata_hash != record.get("metadata_sha256"):
                    raise ValueError(f"Metadata changed since freeze: {record['sample_id']}")
    return partitions, lock, CharacterTokenizer.load(split_root / "tokenizer.json")
