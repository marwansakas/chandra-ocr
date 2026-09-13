"""Aspect-preserving line resizing with explicit width limits and padding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch


@dataclass
class ImageConfig:
    height: int = 64
    max_width: int = 2048
    patch_size: int = 8
    mean: tuple = (0.5, 0.5, 0.5)
    std: tuple = (0.5, 0.5, 0.5)
    ink_mode: str = "original"

    def __post_init__(self):
        self.mean, self.std = tuple(self.mean), tuple(self.std)
        if self.patch_size < 1 or self.height < 1 or self.max_width < 1:
            raise ValueError("Image dimensions and patch size must be positive")
        if self.height % self.patch_size or self.max_width % self.patch_size:
            raise ValueError("Image height and max width must be multiples of patch size")
        if len(self.mean) != 3 or len(self.std) != 3 or any(s <= 0 for s in self.std):
            raise ValueError("RGB normalization requires three means and positive standard deviations")
        if self.ink_mode not in ("original", "dominant_soft"):
            raise ValueError("Unknown ink mode")

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AugmentationConfig:
    """Conservative training-only transforms with whitespace safety margins."""
    policy: str = "none"

    def __post_init__(self):
        if self.policy not in ("none", "mild_v1", "trocr_views_v1", "trocr_all_views_v1"):
            raise ValueError("Unknown augmentation policy")

    def as_dict(self):
        if self.policy == "none":
            return {"policy": "none"}
        if self.policy in ("trocr_views_v1", "trocr_all_views_v1"):
            all_views = self.policy == "trocr_all_views_v1"
            return {
                "policy": self.policy,
                "view_schedule": "all three views every epoch" if all_views else "(epoch - 1) modulo 3",
                "views": ["original", "photometric", "geometry"],
                "photometric": {
                    "brightness_dark": [0.90, 0.97],
                    "brightness_light": [1.03, 1.10],
                    "contrast": [0.88, 1.12],
                },
                "geometry": {
                    "rotation_degrees": [-1.8, 1.8],
                    "scale": [0.96, 1.0],
                    "translation_fraction_of_margin": [-0.5, 0.5],
                    "canvas_margin_fraction": 0.08,
                },
                "views_per_sample_per_epoch": 3 if all_views else 1,
                "optimizer_updates_per_epoch": "three times the one-view dataset" if all_views else "unchanged; one view per sample per epoch",
                "guarantee": "original view is pixel-preserving; geometry uses a white safety canvas; no flips, destructive crop, blur, noise, or stroke filters",
            }
        return {"policy": "mild_v1", "rotation_degrees": [-2.0, 2.0], "scale": [0.96, 1.0],
                "translation_fraction_of_margin": [-0.5, 0.5], "brightness": [0.9, 1.1],
                "contrast": [0.9, 1.1], "blur_probability": 0.10, "blur_radius": 0.25,
                "noise_probability": 0.10, "noise_stddev_8bit": 2.0,
                "canvas_margin_fraction": 0.08,
                "guarantee": "original crop is placed inside a white margin before geometry; no flips or destructive crop"}


def augmentation_view_index(epoch: int) -> int:
    """Map one-based training epochs to original/photo/geometry views."""
    if epoch < 1:
        raise ValueError("Augmentation epoch must be one-based")
    return (epoch - 1) % 3


def _safe_geometry(source: Image.Image, rng: random.Random, rotation_limit: float):
    """Apply bounded geometry after adding whitespace so crop content remains."""
    width, height = source.size
    margin_x, margin_y = max(2, math.ceil(width * 0.08)), max(2, math.ceil(height * 0.08))
    canvas = Image.new("RGB", (width + 2 * margin_x, height + 2 * margin_y), "white")
    scale = rng.uniform(0.96, 1.0)
    scaled = source.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    max_x, max_y = canvas.width - scaled.width, canvas.height - scaled.height
    offset_x = max(0, min(max_x, round(max_x / 2 + rng.uniform(-0.5, 0.5) * margin_x)))
    offset_y = max(0, min(max_y, round(max_y / 2 + rng.uniform(-0.5, 0.5) * margin_y)))
    canvas.paste(scaled, (offset_x, offset_y))
    rotation = rng.uniform(-rotation_limit, rotation_limit)
    result = canvas.rotate(rotation, resample=Image.Resampling.BICUBIC, fillcolor="white")
    details = {"margin_pixels": [margin_x, margin_y], "scale": scale, "offset": [offset_x, offset_y],
               "rotation_degrees": rotation}
    return result, details


def augment_image(image: Image.Image, config: AugmentationConfig, seed: int, *, view_index: int | None = None):
    """Return a deterministic, label-preserving augmented RGB image and record."""
    source = image.convert("RGB")
    if config.policy == "none":
        return source, {"policy": "none", "applied": False}
    rng = random.Random(seed)
    if config.policy in ("trocr_views_v1", "trocr_all_views_v1"):
        if view_index not in (0, 1, 2):
            raise ValueError(f"{config.policy} requires view_index 0 (original), 1 (photometric), or 2 (geometry)")
        view = ("original", "photometric", "geometry")[view_index]
        if view_index == 0:
            return source, {"policy": config.policy, "view": view, "view_index": view_index, "applied": False}
        if view_index == 1:
            brightness_range = (0.90, 0.97) if rng.random() < 0.5 else (1.03, 1.10)
            brightness = rng.uniform(*brightness_range)
            contrast = rng.uniform(0.88, 1.12)
            result = ImageEnhance.Brightness(source).enhance(brightness)
            result = ImageEnhance.Contrast(result).enhance(contrast)
            return result, {"policy": config.policy, "view": view, "view_index": view_index, "applied": True,
                            "seed": seed, "brightness": brightness, "contrast": contrast}
        result, details = _safe_geometry(source, rng, 1.8)
        return result, {"policy": config.policy, "view": view, "view_index": view_index, "applied": True,
                        "seed": seed, **details}

    result, geometry = _safe_geometry(source, rng, 2.0)
    brightness, contrast = rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)
    result = ImageEnhance.Brightness(result).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    blur = rng.random() < 0.10
    if blur:
        result = result.filter(ImageFilter.GaussianBlur(radius=0.25))
    noise = rng.random() < 0.10
    if noise:
        np_rng = np.random.default_rng(seed)
        array = np.asarray(result, dtype=np.float32) + np_rng.normal(0, 2.0, size=(result.height, result.width, 1))
        result = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), "RGB")
    return result, {"policy": config.policy, "applied": True, "seed": seed, **geometry,
                    "brightness": brightness, "contrast": contrast, "blur": blur, "noise": noise}


def augmentation_seed(sample_id: str, epoch: int, seed: int) -> int:
    data = f"{sample_id}\0{epoch}\0{seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def resized_width(width: int, height: int, config: ImageConfig) -> int:
    result = max(1, round(width * config.height / height))
    if result > config.max_width:
        raise ValueError(f"Image width overflow: aspect-preserving resize needs {result}, limit {config.max_width}; increase --max-width")
    return result


def emphasize_dominant_ink(image: Image.Image):
    """Conservative color heuristic, not semantic handwriting segmentation.

    Estimate a broad dominant hue from darkness-weighted pixels. Mixed ink falls
    back to RGB. Otherwise retain at least 65% of every non-dominant stroke's
    original contrast. No component removal, thresholding, or source-file edit.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    high, low = rgb.max(2), rgb.min(2)
    chroma = high - low
    darkness = 1 - rgb.mean(2)
    foreground = darkness > 0.15
    # PIL hue is stable and deterministic; neutral ink has its own category.
    hue = np.asarray(image.convert("RGB").convert("HSV"), dtype=np.uint8)[:, :, 0]
    bins = ((hue.astype(np.int32) + 16) % 256) // 32
    bins = np.where(chroma < 0.12, 8, bins)
    weights = darkness**2 * foreground
    mass = np.bincount(bins.ravel(), weights=weights.ravel(), minlength=9)
    total = float(mass.sum())
    dominant = int(mass.argmax())
    share = float(mass[dominant] / total) if total else 0.0
    info = {"dominant_bin": dominant, "dominant_share": share,
            "applied": bool(total >= 5 and share >= 0.85), "minimum_retained_contrast": 0.65}
    if not info["applied"]:
        return image.convert("RGB"), info
    keep = np.where(bins == dominant, 1.0, 0.65).astype(np.float32)
    result = 1 - (1 - rgb) * keep[:, :, None]
    return Image.fromarray(np.rint(result * 255).astype(np.uint8)), info


def prepare_images(images: list[Image.Image], config: ImageConfig):
    if not images:
        raise ValueError("Empty image batch")
    widths = [resized_width(*image.size, config) for image in images]
    padded_width = math.ceil(max(widths) / config.patch_size) * config.patch_size
    array = np.ones((len(images), 3, config.height, padded_width), dtype=np.float32)
    for i, (image, width) in enumerate(zip(images, widths)):
        if config.ink_mode == "dominant_soft":
            image, _ = emphasize_dominant_ink(image)
        resized = image.convert("RGB").resize((width, config.height), Image.Resampling.LANCZOS)
        array[i, :, :, :width] = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255
    pixels = torch.from_numpy(array)
    mean = torch.tensor(config.mean).reshape(1, 3, 1, 1)
    std = torch.tensor(config.std).reshape(1, 3, 1, 1)
    return (pixels - mean) / std, torch.tensor(widths, dtype=torch.long)
