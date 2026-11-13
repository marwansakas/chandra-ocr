"""Immutable, portable image/text snapshots for the two training stages."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re

from PIL import Image

from thesis.diffusion.data import file_hash
from thesis.diffusion.tokenizer import CharacterTokenizer, read_literal, training_text

DATASET = "johnlockejrr/KHATT_v1.0_dataset"
POLICY = "no_harakat_v1"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pixel_hash(image):
    image = image.convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def add_image(root, sample_id, image, text, *, reuse=False, **extra):
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Empty or invalid transcription: {sample_id}")
    path = root / "images" / (hashlib.sha256(sample_id.encode()).hexdigest() + ".png")
    path.parent.mkdir(exist_ok=True)
    image = image.convert("RGB")
    cached = False
    if reuse and path.exists():
        try:
            with Image.open(path) as previous:
                cached = pixel_hash(previous) == pixel_hash(image)
        except OSError:
            pass  # An uncommitted image write from the old preparer may be incomplete.
        else:
            if not cached:
                raise ValueError("Cached KHATT image differs from the selected source: " + sample_id)
    if not cached:
        temporary = path.with_suffix(".png.tmp")
        image.save(temporary, format="PNG")
        temporary.replace(path)
    return dict(sample_id=sample_id, image_path=path.relative_to(root).as_posix(),
                image_sha256=file_hash(path), pixels_sha256=pixel_hash(image),
                width=image.width, height=image.height, source_text=text,
                text=training_text(text, POLICY), **extra)


def freeze(root, parts, *, stage, source, tokenizer=None, inventory=""):
    """Never silently drop duplicates or move samples across supplied splits."""
    seen_ids, pixels = set(), {}
    for partition, rows in parts.items():
        if not rows:
            raise ValueError(f"Empty {partition} split")
        for row in rows:
            if row["sample_id"] in seen_ids:
                raise ValueError("Duplicate sample ID: " + row["sample_id"])
            seen_ids.add(row["sample_id"])
            previous = pixels.get(row["pixels_sha256"])
            if previous and (previous[0] != partition or previous[1] != row["text"]):
                raise ValueError("Duplicate image crosses splits or has conflicting labels: " + row["sample_id"])
            pixels[row["pixels_sha256"]] = (partition, row["text"])
    tokenizer = tokenizer or CharacterTokenizer.build((r["text"] for r in parts["train"]), "\r\n\t≠" + inventory)
    tokenizer.save(root / "tokenizer.json")
    hashes = {"tokenizer.json": file_hash(root / "tokenizer.json")}
    for name, rows in parts.items():
        path = root / (name + ".jsonl")
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        hashes[path.name] = file_hash(path)
    metadata = dict(version=1, stage=stage, text_policy=POLICY, source=source,
                    hashes=hashes, counts={k: len(v) for k, v in parts.items()},
                    train_max_characters=max(len(r["text"]) for r in parts["train"]))
    # This is the completion marker: it must never be visible half-written.
    temporary = root / "prepared.json.tmp"
    write_json(temporary, metadata)
    temporary.replace(root / "prepared.json")
    return metadata


def prepare_khatt(args):
    from .locking import run_lock
    with run_lock(args.output.resolve()):
        _prepare_khatt(args)


def progress_rows(root, partition):
    """Recover only complete journal records; never skip missing/tampered images."""
    path = root / (partition + ".progress.jsonl")
    if not path.exists():
        return []
    rows, committed = [], 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            if row["sample_id"] != f"khatt:{partition}:{len(rows):07d}":
                raise ValueError("KHATT progress order/identity differs")
            image = (root / row["image_path"]).resolve()
            if not image.is_relative_to(root) or file_hash(image) != row["image_sha256"]:
                raise ValueError("Committed KHATT progress image is missing or changed")
            rows.append(row)
            committed = handle.tell()
    if path.stat().st_size != committed:
        # A killed append can leave one incomplete final record, not a committed sample.
        with path.open("r+b") as handle:
            handle.truncate(committed)
    return rows


def _prepare_khatt(args):
    root = args.output.resolve()
    if (root / "prepared.json").exists():
        load_prepared(root)
        print("Completed KHATT snapshot verified; no preparation needed.", flush=True)
        return
    binding_path = root / "preparation.json"
    previous = read_json(binding_path) if binding_path.exists() else None
    if root.exists() and previous is None and any(p.name != "images" for p in root.iterdir()):
        raise ValueError("Unrecognized incomplete preparation directory; preserve it and choose another --output")
    root.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset, load_from_disk
    inventory = args.inventory_file.read_text(encoding="utf-8") if args.inventory_file else ""
    options = dict(text_policy=POLICY, inventory_sha256=hashlib.sha256(inventory.encode()).hexdigest())
    if args.hf_root:
        source = dict(kind="saved_huggingface_splits", path=str(args.hf_root.resolve()),
                      label_order="mirror reversed; restore once", dataset=DATASET)
        def get_dataset(partition):
            return load_from_disk(str(args.hf_root / ("khatt_hf_" + partition)))
    else:
        from huggingface_hub import HfApi
        requested = args.revision
        if requested is None and previous and previous["source"]["kind"] == "huggingface":
            requested = previous["source"]["revision"]
        revision = HfApi().dataset_info(DATASET, revision=requested).sha
        source = dict(kind="huggingface", dataset=DATASET, revision=revision,
                      label_order="mirror reversed; restore once")
        def get_dataset(partition):
            return load_dataset(DATASET, split=partition, revision=revision, streaming=True)
    binding = dict(version=1, source=source, options=options)
    if previous and previous != binding:
        raise ValueError("Preparation source/revision/inventory changed; retain the original settings")
    if not previous:
        temporary = binding_path.with_suffix(".tmp")
        write_json(temporary, binding)
        temporary.replace(binding_path)
    parts = {}
    for partition in ("train", "validation"):
        rows = progress_rows(root, partition)
        print(f"Loading KHATT {partition}; {len(rows)} committed images available to resume.", flush=True)
        dataset = get_dataset(partition)
        done = len(rows)
        dataset = dataset.skip(done) if hasattr(dataset, "skip") else itertools.islice(dataset, done, None)
        journal = root / (partition + ".progress.jsonl")
        for index, sample in enumerate(dataset, start=done):
            # This conversion is ONLY for this explicitly identified HF mirror.
            raw = sample["text"]
            if not isinstance(raw, str):
                raise ValueError("KHATT transcription must be a string")
            row = add_image(root, f"khatt:{partition}:{index:07d}", sample["image"], raw[::-1],
                            reuse=True, original_mirror_text=raw)
            with journal.open("ab") as handle:
                handle.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())
                handle.flush()
                os.fsync(handle.fileno())
            rows.append(row)
            if (index + 1) % 100 == 0 or index == done:
                print(f"Prepared {partition}: {index + 1}", flush=True)
        parts[partition] = rows
    result = freeze(root, parts, stage="khatt", source=source, inventory=inventory)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def notebook_group(sample_id):
    # Flat exports may add a global ID before one or two PDF/page rank prefixes.
    name = re.sub(r"^(?:\d+_)+", "", sample_id)
    if not re.search(r"_Page_\d+", name, flags=re.I):
        raise ValueError(f"Cannot identify notebook from {sample_id!r}; supply --writer-map or explicitly use --split-strategy line")
    return re.split(r"_Page_\d+", name, flags=re.I)[0]


def split_children(rows, strategy, writer_map, seed, validation_ratio, test_ratio, assignments=None):
    if validation_ratio <= 0 or test_ratio <= 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("Validation/test ratios must be positive and sum to less than 1")
    # Union requested groups AND exact pixel duplicates before assigning splits.
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        parents[find(a)] = find(b)
    groups, pixels = {}, {}
    for index, row in enumerate(rows):
        sid = row["sample_id"]
        group = writer_map[sid] if writer_map is not None else notebook_group(sid) if strategy == "notebook" else sid
        if not isinstance(group, str) or not group:
            raise ValueError("Writer/group IDs must be nonempty strings")
        if group in groups:
            union(index, groups[group])
        groups[group] = index
        key = row["pixels_sha256"]
        if key in pixels:
            if rows[pixels[key]]["text"] != row["text"]:
                raise ValueError("Conflicting duplicate image labels: " + sid)
            union(index, pixels[key])
        pixels[key] = index
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[find(index)].append(row)
    units = sorted(buckets.values(), key=lambda batch: min(r["sample_id"] for r in batch))
    random.Random(seed).shuffle(units)
    if len(units) < 3:
        raise ValueError("Need at least three independent groups for train/validation/test")
    if assignments is not None:
        if set(assignments) != {r["sample_id"] for r in rows} or set(assignments.values()) != {"train", "validation", "test"}:
            raise ValueError("Assignments must cover every usable sample exactly, with train/validation/test")
        parts = {name: [] for name in ("train", "validation", "test")}
        for unit in units:
            names = {assignments[r["sample_id"]] for r in unit}
            if len(names) != 1:
                raise ValueError("Existing assignments split a writer/notebook/duplicate group")
            parts[names.pop()].extend(unit)
        return {k: sorted(v, key=lambda r: r["sample_id"]) for k, v in parts.items()}
    # Greedy whole-group assignment approximates requested line counts.
    targets = dict(train=len(rows)*(1-validation_ratio-test_ratio),
                   validation=len(rows)*validation_ratio, test=len(rows)*test_ratio)
    parts = {name: [] for name in targets}
    for index, unit in enumerate(units):
        empty = [k for k, v in parts.items() if not v]
        choices = empty if len(units)-index == len(empty) else list(parts)
        name = max(choices, key=lambda k: (targets[k]-len(parts[k])) / targets[k])
        parts[name].extend(unit)
    return {k: sorted(v, key=lambda r: r["sample_id"]) for k, v in parts.items()}


def prepare_children(args):
    from thesis.khatt_transfer.training import load_checkpoint
    checkpoint = load_checkpoint(args.init_from)
    tokenizer = CharacterTokenizer(checkpoint["tokenizer"]["characters"])
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows, exclusions = [], []
    for image in sorted(args.dataset_root.rglob("image.png")):
        label = image.with_name("ground_truth.txt")
        sid = image.parent.relative_to(args.dataset_root).as_posix()
        if not label.is_file():
            raise ValueError("Missing label: " + str(label))
        text = read_literal(label)
        if not text.strip() or text.strip().casefold() in {"ignore", "≠"}:
            exclusions.append(dict(sample_id=sid, reason="blank/ignore/non-transcription marker"))
            continue
        with Image.open(image) as source:
            rows.append(add_image(root, sid, source, text, source_label_sha256=file_hash(label)))
    if not rows:
        raise ValueError("No usable children image/text pairs")
    # Unknown characters and length/width problems are errors, never exclusions.
    from thesis.khatt_transfer.training import check_records
    ancestors = set(checkpoint["seen_pixels"])
    if any(r["pixels_sha256"] in ancestors for r in rows):
        raise ValueError("Children images overlap a parent stage's training/validation data")
    writer_map = read_json(args.writer_map) if args.writer_map else None
    assignments = read_json(args.assignments) if args.assignments else None
    parts = split_children(rows, args.split_strategy, writer_map, args.seed,
                           args.validation_ratio, args.test_ratio, assignments)
    for name in ("train", "validation"):
        check_records(parts[name], tokenizer, checkpoint["config"])
    write_json(root / "exclusions.json", exclusions)
    source = dict(kind="children_snapshot", dataset_root=str(args.dataset_root.resolve()),
                  split_strategy="writer-map" if writer_map is not None else args.split_strategy,
                  writer_map_sha256=file_hash(args.writer_map) if args.writer_map else None,
                  assignments_sha256=file_hash(args.assignments) if args.assignments else None,
                  seed=args.seed, validation_ratio=args.validation_ratio, test_ratio=args.test_ratio,
                  init_checkpoint_sha256=file_hash(args.init_from),
                  interpretation="verified writer identity depends on supplied map; notebook and line splits do not establish unseen-writer performance")
    result = freeze(root, parts, stage="children", source=source, tokenizer=tokenizer)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def load_prepared(root, partitions=("train", "validation")):
    root = Path(root).resolve()
    if not (root / "prepared.json").is_file():
        raise ValueError(f"Prepared dataset is incomplete: {root}. Finish prepare-khatt (or prepare-children) successfully before training.")
    metadata = read_json(root / "prepared.json")
    if metadata["version"] != 1 or metadata["text_policy"] != POLICY or metadata["stage"] not in {"khatt", "children"}:
        raise ValueError("Unsupported prepared-data policy")
    tokenizer = CharacterTokenizer.load(root / "tokenizer.json")
    if file_hash(root / "tokenizer.json") != metadata["hashes"]["tokenizer.json"]:
        raise ValueError("Tokenizer hash mismatch")
    parts, ids, seen_pixels = {}, set(), {}
    for name in partitions:
        path = root / (name + ".jsonl")
        if file_hash(path) != metadata["hashes"][path.name]:
            raise ValueError("Manifest hash mismatch: " + name)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not rows or len(rows) != metadata["counts"][name]:
            raise ValueError("Manifest count mismatch")
        for row in rows:
            image = (root / row["image_path"]).resolve()
            if not image.is_relative_to(root) or file_hash(image) != row["image_sha256"]:
                raise ValueError("Image path/hash mismatch: " + row["sample_id"])
            with Image.open(image) as source:
                if source.size != (row["width"], row["height"]) or pixel_hash(source) != row["pixels_sha256"]:
                    raise ValueError("Image dimensions/pixels mismatch")
            if row["text"] != training_text(row["source_text"], POLICY) or row["sample_id"] in ids:
                raise ValueError("Target/ID mismatch")
            ids.add(row["sample_id"])
            previous = seen_pixels.get(row["pixels_sha256"])
            if previous and previous != name:
                raise ValueError("Cross-partition duplicate pixels")
            seen_pixels[row["pixels_sha256"]] = name
        parts[name] = rows
    return metadata, tokenizer, parts
