"""Behavioral checks for the controlled memorization experiments."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thesis.diffusion.images import (AugmentationConfig, ImageConfig, augmentation_view_index,
                                     augment_image, emphasize_dominant_ink)
from thesis.diffusion.model import DiffusionRecognizer, ModelConfig
from thesis.diffusion.runtime import (Collator, LineDataset, all_mask_loss, fixed_corruption_loss,
                                      full_mask_diagnostics, load_checkpoint, save_checkpoint)
from thesis.diffusion.tokenizer import CharacterTokenizer, HARAKAT_V1, training_text


def test_derived_text_preserves_semantic_characters_and_source():
    source = "أَ إِ آ ؤ ئ ا\u0654 ـعبت\nمدرصة ١٢ ة ى شّ"
    target = training_text(source, "no_harakat_v1")
    assert target == "أ إ آ ؤ ئ ا\u0654 ـعبت\nمدرصة ١٢ ة ى ش"
    assert training_text(source) == source
    assert "َ" in source


def test_single_eos_padding_is_not_a_decoder_length_hint(monkeypatch):
    tokenizer = CharacterTokenizer.build(["اب"])
    config = ModelConfig(len(tokenizer), 8, d_model=16, heads=4, encoder_layers=1,
                         decoder_layers=1, patch_size=4, dropout=0, eos_target="single", full_mask_probability=1)
    model = DiffusionRecognizer(config)
    collator = Collator(ImageConfig(16, 128, 4), tokenizer, 8, "single")
    batch = collator([(Image.new("RGB", (24, 16), "black"), {"text": "اب"})])
    assert batch[2].tolist() == [[tokenizer.ids["ا"], tokenizer.ids["ب"], tokenizer.EOS, 0, 0, 0, 0, 0]]
    seen = []
    original = model.denoise

    def capture(noisy, memory, visual_mask, t, **kwargs):
        seen.append((noisy.detach().clone(), t.detach().clone()))
        return original(noisy, memory, visual_mask, t, **kwargs)

    monkeypatch.setattr(model, "denoise", capture)
    loss = model(*batch[:3], generator=torch.Generator().manual_seed(9))
    loss.backward()
    assert torch.isfinite(loss)
    assert seen[0][0].tolist() == [[tokenizer.MASK] * 8]
    assert seen[0][1].tolist() == [1.0]
    assert model.encoder.patch.weight.grad.abs().sum() > 0
    model.eval()
    diag = full_mask_diagnostics(model, [batch], "cpu")
    assert diag["characters"]["positions"] == 2
    assert diag["first_eos"]["positions"] == 1
    assert diag["tail_eos"]["positions"] == 0 and diag["tail_eos"]["accuracy"] is None


def test_color_fallback_and_retained_strokes():
    mixed = np.full((20, 20, 3), 255, dtype=np.uint8)
    mixed[:10] = [0, 0, 0]
    mixed[10:] = [255, 0, 0]
    source = Image.fromarray(mixed)
    result, info = emphasize_dominant_ink(source)
    assert not info["applied"] and np.array_equal(np.asarray(result), mixed)
    dominant = mixed.copy()
    dominant[:] = 0
    dominant[-1, -1] = [255, 0, 0]
    result, info = emphasize_dominant_ink(Image.fromarray(dominant))
    assert info["applied"]
    # Minority red survives; source RGB is not mutated. Quantization tolerance.
    assert np.asarray(result)[-1, -1, 1] <= 90
    assert np.array_equal(dominant[-1, -1], [255, 0, 0])
    for value in (255, 254):
        blank = Image.new("RGB", (20, 20), (value,) * 3)
        result, info = emphasize_dominant_ink(blank)
        assert not info["applied"] and np.array_equal(result, blank)


def test_new_checkpoint_restores_policy_and_old_defaults(tmp_path):
    from thesis.diffusion.data import file_hash, json_bytes
    tokenizer = CharacterTokenizer.build(["بَـ"])
    excluded = tuple(sorted(tokenizer.ids[c] for c in HARAKAT_V1 if c in tokenizer.ids))
    config = ModelConfig(len(tokenizer), 8, d_model=16, heads=4, encoder_layers=1, decoder_layers=1,
                         patch_size=4, dropout=0, eos_target="single", full_mask_probability=.25,
                         text_policy="no_harakat_v1", excluded_token_ids=excluded)
    model = DiffusionRecognizer(config).eval()
    image_config = ImageConfig(16, 128, 4, ink_mode="dominant_soft")
    save_checkpoint(tmp_path / "new", model, tokenizer, image_config, metadata={})
    loaded, _, restored_image, _ = load_checkpoint(tmp_path / "new")
    assert loaded.config == config and restored_image == image_config
    pixels, widths, targets, _ = Collator(image_config, tokenizer, 8, "single")([(Image.new("RGB", (24, 16)), {"text": "بـ"})])
    loaded.train()
    loss = loaded(pixels, widths, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(loaded.head.weight.grad[[i - 2 for i in excluded]], torch.zeros_like(loaded.head.weight.grad[[i - 2 for i in excluded]]))
    loaded.eval()
    ids, _ = loaded.sample(pixels, widths, steps=2)
    assert not any(i in excluded for i in ids[0].tolist())
    # Simulate a genuine pre-feature checkpoint config, keeping integrity valid.
    old_config = ModelConfig(len(tokenizer), 8, d_model=16, heads=4, encoder_layers=1, decoder_layers=1, patch_size=4)
    save_checkpoint(tmp_path / "old", DiffusionRecognizer(old_config), tokenizer, ImageConfig(16, 128, 4), metadata={})
    path = tmp_path / "old" / "config.json"
    saved = json.loads(path.read_text())
    for key in ("eos_target", "full_mask_probability", "text_policy", "excluded_token_ids"):
        saved["model"].pop(key)
    saved["image"].pop("ink_mode")
    path.write_bytes(json_bytes(saved))
    manifest_path = path.parent / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"]["config.json"] = file_hash(path)
    manifest_path.write_bytes(json_bytes(manifest))
    old, _, old_image, _ = load_checkpoint(path.parent)
    assert old.config.eos_target == "repeat" and old.config.full_mask_probability == 0
    assert old_image.ink_mode == "original" and old.config.text_policy == "literal"


def test_invalid_objective_configuration():
    with pytest.raises(ValueError, match="probability"):
        ModelConfig(20, 8, full_mask_probability=1.1)
    with pytest.raises(ValueError, match="only to diffusion"):
        ModelConfig(20, 8, decoder_kind="autoregressive", full_mask_probability=.25)


def test_mild_augmentation_is_deterministic_and_all_mask_loss_is_fixed():
    source = Image.new("RGB", (40, 20), "white")
    source.putpixel((20, 10), (0, 0, 0))  # A dot-like mark must remain in the augmented canvas.
    first, details = augment_image(source, AugmentationConfig("mild_v1"), 77)
    second, _ = augment_image(source, AugmentationConfig("mild_v1"), 77)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert first.width > source.width and first.height > source.height and min(details["margin_pixels"]) >= 2
    tokenizer = CharacterTokenizer.build(["بت"])
    model = DiffusionRecognizer(ModelConfig(len(tokenizer), 8, d_model=16, heads=4, encoder_layers=1, decoder_layers=1,
                                             patch_size=4, dropout=0)).eval()
    batch = Collator(ImageConfig(16, 128, 4), tokenizer, 8)([(source, {"text": "بت", "sample_id": "sample"})])
    before = torch.get_rng_state().clone()
    first_loss = fixed_corruption_loss(model, [batch], "cpu", 300042)
    second_loss = fixed_corruption_loss(model, [batch], "cpu", 300042)
    assert first_loss == second_loss and torch.equal(before, torch.get_rng_state())
    assert all_mask_loss(model, [batch], "cpu") >= 0


def test_trocr_style_views_cycle_without_losing_the_original():
    source_array = np.full((20, 40, 3), 255, dtype=np.uint8)
    source_array[8:12, 18:22] = 0  # small ink mark must remain visible
    source = Image.fromarray(source_array)
    assert [augmentation_view_index(epoch) for epoch in range(1, 7)] == [0, 1, 2, 0, 1, 2]

    original, original_details = augment_image(source, AugmentationConfig("trocr_views_v1"), 1, view_index=0)
    photo_a, photo_details = augment_image(source, AugmentationConfig("trocr_views_v1"), 2, view_index=1)
    photo_b, _ = augment_image(source, AugmentationConfig("trocr_views_v1"), 2, view_index=1)
    geometry, geometry_details = augment_image(source, AugmentationConfig("trocr_views_v1"), 3, view_index=2)

    assert np.array_equal(np.asarray(original), source_array)
    assert original_details["view"] == "original" and not original_details["applied"]
    assert photo_a.size == source.size and np.array_equal(np.asarray(photo_a), np.asarray(photo_b))
    assert photo_details["brightness"] < 0.97 or photo_details["brightness"] > 1.03
    assert geometry.width > source.width and geometry.height > source.height
    assert np.asarray(geometry).min() < 128 and geometry_details["view"] == "geometry"
    with pytest.raises(ValueError, match="requires view_index"):
        augment_image(source, AugmentationConfig("trocr_views_v1"), 4)


def test_all_trocr_views_are_present_in_each_training_epoch(tmp_path):
    image_path = tmp_path / "line.png"
    array = np.full((20, 40, 3), 240, dtype=np.uint8)
    array[7:13, 15:25] = 60
    Image.fromarray(array).save(image_path)
    records = [{"sample_id": "line-1", "image_path": "line.png", "text": "بت"}]
    dataset = LineDataset(tmp_path, records, views_per_sample=3)
    assert len(dataset) == 3
    samples = [dataset[index] for index in range(3)]
    assert [sample[2] for sample in samples] == [0, 1, 2]
    tokenizer = CharacterTokenizer.build(["بت"])
    collator = Collator(ImageConfig(16, 128, 4), tokenizer, 8,
                        augmentation=AugmentationConfig("trocr_all_views_v1"), epoch=7)
    pixels, widths, targets, returned = collator(samples)
    assert pixels.shape[0] == widths.shape[0] == targets.shape[0] == len(returned) == 3
    assert not torch.equal(pixels[0], pixels[1])
    assert not torch.equal(pixels[0], pixels[2])


def test_compact_epoch_progress_contains_only_requested_fields():
    from thesis.scripts.train_diffusion_line_ocr_model import format_epoch_progress

    line = format_epoch_progress({
        "epoch": 7,
        "train_loss": 1.23456789,
        "validation_loss": 2.34567891,
        "metrics": {"strict": {"cer": 0.81234, "exact_lines": 3, "no_eos": 9}},
        "elapsed_seconds": 65.43,
        "peak_cuda_memory_bytes": 123,
    })
    assert line == (
        "Epoch 007 | Train loss 1.234568 | Validation loss 2.345679 | "
        "CER 81.234% | Exact lines 3 | Time 65.4s"
    )
    assert "CUDA" not in line and "no_eos" not in line
