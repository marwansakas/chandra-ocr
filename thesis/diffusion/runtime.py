"""Checkpoint integrity, reproducibility and shared training/inference adapters."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
from PIL import Image
import torch

from .data import file_hash, json_bytes, resolve_relative
from .images import (AugmentationConfig, ImageConfig, augmentation_seed, augmentation_view_index,
                     augment_image, prepare_images)
from .model import DiffusionRecognizer, ModelConfig, Tokens, diffusion_loss
from .tokenizer import CharacterTokenizer, HARAKAT_V1

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def get_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def runtime_info():
    versions = {}
    for package in ("torch", "numpy", "Pillow", "transformers", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    sources = list((ROOT / "thesis" / "diffusion").glob("*.py"))
    sources += list((ROOT / "thesis" / "scripts").glob("*diffusion*.py"))
    sources += [ROOT / "pipelines" / "diffusion_line_run.py", ROOT / "pipelines" / "line_runner_common.py", ROOT / "thesis" / "scripts" / "common.py"]
    sources += [ROOT / "requirements-diffusion.txt", ROOT / "requirements-diffusion-vit.txt", ROOT / "thesis" / "DIFFUSION.md"]
    return {"python": platform.python_version(), "platform": platform.platform(), "versions": versions,
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "source_revision": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--short"),
            "source_hashes": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources if p.is_file()},
            "determinism": "seeded Python/NumPy/torch/workers; cuDNN deterministic; bitwise equality across hardware is not promised"}


def rng_state():
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(path, model, tokenizer, image_config, *, metadata, training_state=None):
    path.mkdir(parents=True, exist_ok=False)
    config = {"format_version": 1, "model": model.config.as_dict(), "image": image_config.as_dict()}
    (path / "config.json").write_bytes(json_bytes(config))
    tokenizer.save(path / "tokenizer.json")
    torch.save(model.state_dict(), path / "model.pt")
    files = ["config.json", "tokenizer.json", "model.pt"]
    if training_state is not None:
        torch.save(training_state, path / "training_state.pt")
        files.append("training_state.pt")
    (path / "checkpoint.json").write_bytes(json_bytes({"metadata": metadata, "hashes": {name: file_hash(path / name) for name in files}}))


def checkpoint_path(path: Path) -> Path:
    for pointer in ("best.json", "last.json"):
        if (path / pointer).is_file():
            name = json.loads((path / pointer).read_text(encoding="utf-8"))["checkpoint"]
            return resolve_relative(path, name)
    return path


def load_checkpoint(path: Path, device="cpu"):
    path = checkpoint_path(path)
    info = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    for name in ("config.json", "tokenizer.json", "model.pt"):
        if file_hash(path / name) != info["hashes"][name]:
            raise ValueError(f"Checkpoint integrity mismatch: {name}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    tokenizer = CharacterTokenizer.load(path / "tokenizer.json")
    model_config = ModelConfig(**config["model"])
    if model_config.vocab_size != len(tokenizer):
        raise ValueError("Vocabulary does not match model head")
    expected_exclusions = tuple(sorted(tokenizer.ids[c] for c in HARAKAT_V1 if c in tokenizer.ids)) if model_config.text_policy == "no_harakat_v1" else ()
    if model_config.excluded_token_ids != expected_exclusions:
        raise ValueError("Output exclusions do not match saved text policy and tokenizer")
    model = DiffusionRecognizer(model_config)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True), strict=True)
    return model.to(device), tokenizer, ImageConfig(**config["image"]), info["metadata"]


def load_training_state(path: Path):
    path = checkpoint_path(path)
    info = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if file_hash(path / "training_state.pt") != info["hashes"]["training_state.pt"]:
        raise ValueError("Training state integrity mismatch")
    return torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)


def verify_checkpoint_provenance(metadata, tokenizer, frozen_tokenizer, lock, split_lock_sha256: str):
    """Reject evaluation of a checkpoint against a different frozen split."""
    required = ("dataset_fingerprint", "split_lock_sha256")
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(f"Checkpoint has no frozen-run provenance: missing {', '.join(missing)}")
    if metadata["dataset_fingerprint"] != lock["dataset_fingerprint"]:
        raise ValueError("Checkpoint dataset fingerprint differs from the requested frozen split")
    if metadata["split_lock_sha256"] != split_lock_sha256:
        raise ValueError("Checkpoint split hash differs from the requested frozen split")
    if tokenizer.as_dict() != frozen_tokenizer.as_dict():
        raise ValueError("Checkpoint tokenizer differs from the requested frozen split")


class LineDataset(torch.utils.data.Dataset):
    def __init__(self, root, records, views_per_sample: int = 1):
        if views_per_sample < 1:
            raise ValueError("views_per_sample must be positive")
        self.root, self.records, self.views_per_sample = root, records, views_per_sample

    def __len__(self):
        return len(self.records) * self.views_per_sample

    def __getitem__(self, index):
        record_index, view_index = divmod(index, self.views_per_sample)
        record = self.records[record_index]
        with Image.open(resolve_relative(self.root, record["image_path"])) as image:
            sample = image.convert("RGB"), record
        return (*sample, view_index) if self.views_per_sample > 1 else sample


@dataclass
class Collator:
    image_config: ImageConfig
    tokenizer: CharacterTokenizer
    capacity: int
    eos_target: str = "repeat"
    augmentation: AugmentationConfig = AugmentationConfig()
    augmentation_seed_base: int = 42
    epoch: int = 0

    def __call__(self, samples):
        if len(samples[0]) == 3:
            images, records, supplied_view_indices = zip(*samples)
        else:
            images, records = zip(*samples)
            supplied_view_indices = None
        if self.augmentation.policy != "none":
            if self.augmentation.policy == "trocr_all_views_v1":
                if supplied_view_indices is None:
                    raise ValueError("trocr_all_views_v1 requires a multi-view training dataset")
                view_indices = supplied_view_indices
            else:
                view_index = augmentation_view_index(self.epoch) if self.augmentation.policy == "trocr_views_v1" else None
                view_indices = (view_index,) * len(records)
            images = tuple(augment_image(image, self.augmentation,
                                         augmentation_seed(record["sample_id"], self.epoch, self.augmentation_seed_base),
                                         view_index=view_index)[0]
                           for image, record, view_index in zip(images, records, view_indices))
        pixels, widths = prepare_images(list(images), self.image_config)
        if self.eos_target not in ("repeat", "single"):
            raise ValueError("Unknown EOS target policy")
        targets = torch.tensor([self.tokenizer.encode(r["text"], self.capacity, eos_fill=self.eos_target == "repeat") for r in records], dtype=torch.long)
        return pixels, widths, targets, list(records)


@torch.no_grad()
def fixed_corruption_loss(model, batches, device, seed: int):
    """Evaluation-mode denoising loss under a repeatable corruption schedule."""
    if model.training:
        raise ValueError("Fixed corruption loss requires eval mode")
    generator = torch.Generator(device=device).manual_seed(seed)
    total, seen = 0.0, 0
    for pixels, widths, targets, records in batches:
        loss = model(pixels.to(device), widths.to(device), targets.to(device), generator=generator)
        total += float(loss) * len(records)
        seen += len(records)
    return total / seen if seen else None


@torch.no_grad()
def all_mask_loss(model, batches, device):
    """Target-free-input loss at t=1; targets select loss positions only."""
    if model.training:
        raise ValueError("All-mask loss requires eval mode")
    total, seen = 0.0, 0
    for pixels, widths, targets, records in batches:
        pixels, widths, targets = pixels.to(device), widths.to(device), targets.to(device)
        memory, visual_mask = model.encoder(pixels, widths)
        noisy = torch.full_like(targets, Tokens.MASK)
        logits = model.denoise(noisy, memory, visual_mask, pixels.new_ones(pixels.shape[0]))
        loss = diffusion_loss(logits, targets, targets.ne(Tokens.PAD), pixels.new_ones(pixels.shape[0]))
        total += float(loss) * len(records)
        seen += len(records)
    return total / seen if seen else None


@torch.no_grad()
def full_mask_diagnostics(model, batches, device):
    """Separate character, first-EOS and tail-EOS accuracy without target input.

    Targets select scoring positions only. Every decoder input is MASK, at the
    configured fixed capacity; there is no true-length attention mask.
    """
    if model.training:
        raise ValueError("Diagnostics require eval mode")
    totals = {name: {"positions": 0, "correct": 0, "cross_entropy_sum": 0.0}
              for name in ("characters", "first_eos", "tail_eos")}
    for pixels, widths, targets, _ in batches:
        pixels, widths, targets = pixels.to(device), widths.to(device), targets.to(device)
        memory, visual_mask = model.encoder(pixels, widths)
        noisy = torch.full_like(targets, CharacterTokenizer.MASK)
        logits = model.denoise(noisy, memory, visual_mask, pixels.new_ones(pixels.shape[0]))
        ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none", ignore_index=CharacterTokenizer.PAD)
        correct = logits.argmax(-1).eq(targets)
        eos = targets.eq(CharacterTokenizer.EOS)
        first = eos & eos.cumsum(1).eq(1)
        groups = {"characters": targets.ge(3), "first_eos": first, "tail_eos": eos & ~first}
        for name, mask in groups.items():
            totals[name]["positions"] += int(mask.sum())
            totals[name]["correct"] += int((correct & mask).sum())
            totals[name]["cross_entropy_sum"] += float(ce[mask].sum())
    for values in totals.values():
        count = values["positions"]
        values["accuracy"] = values["correct"] / count if count else None
        values["cross_entropy"] = values["cross_entropy_sum"] / count if count else None
    return totals
