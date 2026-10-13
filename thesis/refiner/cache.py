"""One-time frozen TrOCR inference, portable caches, and split/source validation."""

import json
import importlib.metadata
from pathlib import Path
import re
import time

from PIL import Image
import torch

from thesis.diffusion.data import digest_bytes, file_hash, json_bytes, load_frozen, resolve_relative
from .model import encode_draft

DEFAULT_BASE = "microsoft/trocr-base-handwritten"
DEFAULT_REVISION = "eaacaf452b06415df8f10bb6fad3a4c11e609406"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def frozen_data(dataset_root, split_root):
    parts, lock, tokenizer = load_frozen(split_root, dataset_root)
    if any(not parts[name] for name in ("train", "validation", "test")):
        raise ValueError("Refiner selection requires a three-way split with nonempty validation; a two-way pilot is insufficient")
    return parts, lock, tokenizer


def base_source(model_name, revision, manifest, external_base, partitions):
    if bool(manifest) == bool(external_base):
        raise ValueError("Specify exactly one: --base-training-manifest or --external-base (no project-data training)")
    provenance = {"kind": "external_pretrained_no_project_training"}
    if manifest:
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        train = {r["sample_id"]: r for r in partitions["train"]}
        if not rows or len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Base training manifest must have unique, nonempty training records")
        for row in rows:
            if row["sample_id"] not in train:
                raise ValueError("Base training manifest includes a held-out or unknown sample: " + row["sample_id"])
            original = train[row["sample_id"]]
            for key in ("text", "image_sha256"):
                if key not in row or row[key] != original[key]:
                    raise ValueError(f"Base training manifest {key} differs or is absent: {row['sample_id']}")
        provenance = {"kind": "project_train_only", "manifest_sha256": file_hash(manifest), "samples": len(rows),
                      "training_drafts": "in-sample; whole-draft dropout mitigates but does not eliminate train/inference mismatch"}
    path = Path(model_name)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in (".json", ".txt", ".safetensors", ".bin", ".model"))
        if not any(p.suffix in (".safetensors", ".bin") for p in files):
            raise ValueError("Local TrOCR directory has no weights; choose an actual saved checkpoint")
        identity = {"kind": "local", "files": {p.name: file_hash(p) for p in files}}
    else:
        if revision is None and model_name == DEFAULT_BASE:
            revision = DEFAULT_REVISION
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("A remote base requires an immutable 40-character --base-revision")
        identity = {"kind": "huggingface", "model": model_name, "revision": revision}
    return {"identity": identity, "provenance": provenance}, revision


@torch.inference_mode()
def extract_batch(model, processor, images, *, device, max_new_tokens):
    """Image-only inference: no reference strings/lengths enter either HF model."""
    from transformers.modeling_outputs import BaseModelOutput

    model.eval()
    started = time.perf_counter()
    pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
    memory = model.encoder(pixel_values=pixels).last_hidden_state
    tokens = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                            return_dict_in_generate=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - started) / len(images)
    drafts = processor.batch_decode(tokens, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else (eos or [])
    # Ignore decoder start token, which can itself equal EOS.
    ended = [any(int(token) in eos_ids for token in row[1:]) for row in tokens]
    return memory.detach().cpu().to(torch.float16), drafts, ended, seconds


def prepare_cache(args):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from thesis.diffusion.runtime import get_device

    parts, lock, _ = frozen_data(args.dataset_root, args.split_root)
    source, revision = base_source(args.base_model, args.base_revision, args.base_training_manifest,
                                   args.external_base, parts)
    base = {"version": 1, "split_lock_sha256": file_hash(args.split_root / "split_lock.json"),
            "dataset_fingerprint": lock["dataset_fingerprint"], "source": source,
            "extraction_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "Pillow")},
            "max_new_tokens": args.max_new_tokens, "preprocessing": "saved TrOCR processor; RGB; no image augmentation"}
    base["signature"] = digest_bytes(json_bytes(base))
    args.cache_root.mkdir(parents=True, exist_ok=True)
    base_path = args.cache_root / "base.json"
    if base_path.exists():
        if read_json(base_path) != base:
            raise ValueError("Cache base/split/settings changed; use a new cache root")
    else:
        save_json(base_path, base)
    for partition in args.partition:
        if (args.cache_root / partition).exists() and not args.resume:
            raise FileExistsError(f"Cache partition exists: {partition}; use --resume to verify/reuse it")
    device = get_device(args.device)
    options = {"local_files_only": Path(args.base_model).is_dir(), "trust_remote_code": False}
    if revision and not Path(args.base_model).is_dir():
        options["revision"] = revision
    processor = TrOCRProcessor.from_pretrained(args.base_model, **options)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model, **options).to(device).eval()
    model.requires_grad_(False)
    for partition in args.partition:
        root = args.cache_root / partition
        root.mkdir(exist_ok=True)
        entries = []
        for start in range(0, len(parts[partition]), args.batch_size):
            batch = parts[partition][start:start + args.batch_size]
            pending = []
            for record in batch:
                key = digest_bytes(record["sample_id"].encode())
                sidecar = root / (key + ".json")
                if args.resume and sidecar.exists():
                    entry = read_json(sidecar)
                    if (entry["sample_id"] != record["sample_id"] or entry["image_sha256"] != record["image_sha256"]
                            or entry["base_signature"] != base["signature"]
                            or file_hash(resolve_relative(root, entry["file"])) != entry["sha256"]):
                        raise ValueError("Resume cache integrity mismatch: " + record["sample_id"])
                    entries.append(entry)
                else:
                    pending.append((record, key))
            if pending:
                images = []
                for record, _ in pending:
                    with Image.open(resolve_relative(args.dataset_root, record["image_path"])) as image:
                        images.append(image.convert("RGB"))
                features, drafts, ended, seconds = extract_batch(model, processor, images, device=device,
                                                                 max_new_tokens=args.max_new_tokens)
                for (record, key), feature, draft, has_eos in zip(pending, features, drafts, ended):
                    if not torch.isfinite(feature).all():
                        raise ValueError("Nonfinite cached visual features")
                    target = root / (key + ".pt")
                    temporary = target.with_suffix(".tmp")
                    torch.save({"visual": feature.clone(), "draft": draft, "base_has_eos": has_eos,
                                "base_seconds": seconds}, temporary)
                    temporary.replace(target)
                    entry = {"sample_id": record["sample_id"], "image_sha256": record["image_sha256"],
                             "base_signature": base["signature"], "file": target.name, "sha256": file_hash(target),
                             "visual_dim": feature.shape[-1]}
                    save_json(root / (key + ".json"), entry)
                    entries.append(entry)
            print(f"Cached {partition}: {min(start + args.batch_size, len(parts[partition]))}/{len(parts[partition])}", flush=True)
        entries.sort(key=lambda x: x["sample_id"])
        save_json(root / "manifest.json", {"base_signature": base["signature"], "partition": partition, "entries": entries})


class CachedLines(torch.utils.data.Dataset):
    def __init__(self, cache_root, partition, records, split_sha, *, expected_signature=None):
        self.root = cache_root / partition
        self.base = read_json(cache_root / "base.json")
        canonical = {k: v for k, v in self.base.items() if k != "signature"}
        if digest_bytes(json_bytes(canonical)) != self.base["signature"]:
            raise ValueError("Cache base signature is invalid")
        if self.base["split_lock_sha256"] != split_sha:
            raise ValueError("Cache split mismatch")
        if expected_signature and self.base["signature"] != expected_signature:
            raise ValueError("Training/evaluation base mismatch")
        manifest = read_json(self.root / "manifest.json")
        if manifest["base_signature"] != self.base["signature"] or manifest["partition"] != partition:
            raise ValueError("Cache manifest mismatch")
        entries = manifest["entries"]
        lookup = {r["sample_id"]: r for r in records}
        if len(entries) != len(lookup) or {e["sample_id"] for e in entries} != set(lookup):
            raise ValueError("Cache coverage differs from the complete frozen partition")
        self.items = []
        for entry in entries:
            record = lookup[entry["sample_id"]]
            if entry["image_sha256"] != record["image_sha256"] or entry["base_signature"] != self.base["signature"]:
                raise ValueError("Cached image/source mismatch")
            path = resolve_relative(self.root, entry["file"])
            if file_hash(path) != entry["sha256"]:
                raise ValueError("Cache tensor integrity mismatch")
            self.items.append((path, record))
        dimensions = {entry["visual_dim"] for entry in entries}
        if len(dimensions) != 1:
            raise ValueError("Inconsistent visual feature dimensions")
        self.visual_dim = dimensions.pop()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, record = self.items[index]
        value = torch.load(path, map_location="cpu", weights_only=True)
        visual = value["visual"].float()
        if visual.ndim != 2 or visual.shape[-1] != self.visual_dim or not torch.isfinite(visual).all():
            raise ValueError("Invalid cached visual features")
        return {**value, "visual": visual, "record": record}


class CachedCollator:
    def __init__(self, tokenizer, capacity, *, targets=True):
        self.tokenizer, self.capacity, self.targets = tokenizer, capacity, targets

    def __call__(self, rows):
        visual = torch.nn.utils.rnn.pad_sequence([row["visual"] for row in rows], batch_first=True)
        lengths = torch.tensor([len(row["visual"]) for row in rows])
        mask = torch.arange(visual.shape[1])[None] >= lengths[:, None]
        drafts = torch.tensor([encode_draft(self.tokenizer, row["draft"], self.capacity) for row in rows])
        targets = (torch.tensor([self.tokenizer.encode(row["record"]["text"], self.capacity, eos_fill=False) for row in rows])
                   if self.targets else None)
        return visual, mask, drafts, targets, rows
