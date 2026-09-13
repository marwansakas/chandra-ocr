from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from thesis.diffusion.data import audit_dataset, check_partitions, freeze_line_split, freeze_splits, load_frozen, normalize_exclusion_ids, resolve_relative
from thesis.diffusion.images import ImageConfig, prepare_images
from thesis.diffusion.metrics import aggregate, correction_changes, score_pair
from thesis.diffusion.model import DiffusionRecognizer, ModelConfig, corrupt_tokens, diffusion_loss, reverse_step
from thesis.diffusion.runtime import load_checkpoint, restore_rng, rng_state, save_checkpoint, seed_everything, verify_checkpoint_provenance
from thesis.diffusion.tokenizer import CharacterTokenizer, read_literal

torch.set_num_threads(2)


def fixture_data(root, count=12):
    writers = {}
    for index in range(count):
        sample = f"page_{index // 2}/line_{index:03d}"
        folder = root / sample
        folder.mkdir(parents=True)
        Image.new("RGB", (20 + index * 3, 16), (index * 13 % 255, 120, 210)).save(folder / "image.png")
        (folder / "ground_truth.txt").write_bytes(" أَب ١، مدرصة  \n".encode("utf-8"))
        writers[sample] = f"child_{index // 2}"
    return writers


def tiny_model(tokenizer, capacity=16):
    return DiffusionRecognizer(ModelConfig(len(tokenizer), capacity, d_model=32, heads=4,
                                           encoder_layers=1, decoder_layers=1, patch_size=4, dropout=0))


def test_literal_unicode_roundtrip_and_overflow(tmp_path):
    text = " أَب ١٢ 12، مدرصة ـ ﻻ e\u0301 \t"
    tokenizer = CharacterTokenizer.build([text])
    ids = tokenizer.encode(text, len(text) + 4)
    assert tokenizer.decode(ids) == (text, True)
    assert ids[-3:] == [tokenizer.EOS] * 3
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    assert CharacterTokenizer.load(path).tokens == tokenizer.tokens
    with pytest.raises(ValueError, match="overflow"):
        tokenizer.encode(text, len(text))
    with pytest.raises(ValueError, match="Out-of-vocabulary"):
        tokenizer.encode("☃", 3)
    label = tmp_path / "label.txt"
    label.write_bytes((text + "\r\n").encode("utf-8"))
    assert read_literal(label) == text
    label.write_bytes(b"a\n\n")
    assert read_literal(label) == "a\n"
    stacked = "ماضي\nزار المريض\r\nطبيب الاسنان"
    label.write_bytes((stacked + "\n").encode("utf-8"))
    assert read_literal(label) == stacked
    stacked_tokenizer = CharacterTokenizer.build([stacked])
    assert stacked_tokenizer.decode(stacked_tokenizer.encode(stacked, len(stacked) + 1)) == (stacked, True)
    assert score_pair(stacked, stacked)["char_distance"] == 0
    assert score_pair(stacked, stacked.replace("\n", " "))["char_distance"] > 0


def test_line_split_keeps_duplicate_pixels_together_and_validates_exclusions(tmp_path):
    root = tmp_path / "data"
    fixture_data(root, count=12)
    samples = sorted(path.parent.relative_to(root).as_posix() for path in root.rglob("image.png"))
    (root / samples[1] / "image.png").write_bytes((root / samples[0] / "image.png").read_bytes())
    audit = audit_dataset(root)
    split = tmp_path / "split"
    freeze_line_split(audit, split)
    parts, lock, _ = load_frozen(split, root)
    locations = {row["sample_id"]: name for name, rows in parts.items() for row in rows}
    assert locations[samples[0]] == locations[samples[1]]
    assert lock["cross_split_duplicate_pixel_hashes"] == []
    (root / samples[1] / "ground_truth.txt").write_text("other", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting labels"):
        freeze_line_split(audit_dataset(root), tmp_path / "conflict")
    with pytest.raises(ValueError, match="Invalid exclusion"):
        normalize_exclusion_ids(["../escape"])
    (root / samples[2] / "ground_truth.txt").write_text(" ≠ \n", encoding="utf-8")
    excluded = audit_dataset(root, excluded_sample_ids=[samples[2]])
    assert samples[2] not in {row["sample_id"] for row in excluded["records"]}
    assert excluded["excluded_records"] == [{"sample_id": samples[2], "reason": "standalone_not_equal"}]


def test_checkpoint_provenance_rejects_a_different_frozen_split(tmp_path):
    root = tmp_path / "data"
    writers = fixture_data(root, count=12)
    split = tmp_path / "split"
    freeze_splits(audit_dataset(root, writers), split)
    _, lock, tokenizer = load_frozen(split, root)
    with pytest.raises(ValueError, match="dataset fingerprint"):
        verify_checkpoint_provenance({"dataset_fingerprint": "other", "split_lock_sha256": "hash"}, tokenizer, tokenizer, lock, "hash")
    with pytest.raises(ValueError, match="no frozen-run provenance"):
        verify_checkpoint_provenance({}, tokenizer, tokenizer, lock, "hash")


def test_eos_no_eos_unresolved():
    tokenizer = CharacterTokenizer.build(["اب"])
    assert tokenizer.decode([tokenizer.EOS]) == ("", True)
    assert tokenizer.decode([tokenizer.ids["ا"]]) == ("ا", False)
    with pytest.raises(ValueError, match="Unresolved"):
        tokenizer.decode([tokenizer.MASK])


def test_audit_freeze_portability_and_integrity(tmp_path):
    root = tmp_path / "data"
    writers = fixture_data(root)
    audit = audit_dataset(root, writers)
    assert audit["labeled_records"] == 12 and not audit["issues"]
    assert audit["lengths_codepoints"]["max"] == len(" أَب ١، مدرصة  ")
    split = tmp_path / "split"
    freeze_splits(audit, split)
    partitions, lock, tokenizer = load_frozen(split, root)
    check_partitions(partitions)
    assert sum(lock["counts"].values()) == 12
    assert lock["capacity"] == len(" أَب ١، مدرصة  ") + 1
    assert all(not Path(r["image_path"]).is_absolute() for rows in partitions.values() for r in rows)
    import shutil
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert audit_dataset(relocated, writers)["fingerprint"] == audit["fingerprint"]
    load_frozen(split, relocated)
    with pytest.raises(FileExistsError):
        freeze_splits(audit, split)
    sample = partitions["train"][0]
    (root / sample["label_path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed frozen sample"):
        load_frozen(split, root)


def test_missing_writer_duplicate_leakage_and_conflicting_labels(tmp_path):
    root = tmp_path / "data"
    writers = fixture_data(root)
    with pytest.raises(ValueError, match="writer map"):
        freeze_splits(audit_dataset(root), tmp_path / "no_writer")
    samples = sorted(writers)
    (root / samples[2] / "image.png").write_bytes((root / samples[0] / "image.png").read_bytes())
    audit = audit_dataset(root, writers)
    assert len(audit["duplicates"]) == 1
    freeze_splits(audit, tmp_path / "split")
    partitions, _, _ = load_frozen(tmp_path / "split", root)
    locations = {r["sample_id"]: name for name, rows in partitions.items() for r in rows}
    assert locations[samples[0]] == locations[samples[2]]
    (root / samples[2] / "ground_truth.txt").write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting labels"):
        freeze_splits(audit_dataset(root, writers), tmp_path / "conflict")
    record = {"sample_id": "x", "writer_id": "a", "pixels_sha256": "x"}
    other = {"sample_id": "y", "writer_id": "a", "pixels_sha256": "y"}
    with pytest.raises(ValueError, match="writer_id"):
        check_partitions({"train": [record], "test": [other]})


def test_missing_labels_and_manifest_changes_are_explicit(tmp_path):
    root = tmp_path / "data"
    writers = fixture_data(root)
    sample = next(iter(writers))
    (root / sample / "ground_truth.txt").unlink()
    audit = audit_dataset(root, writers)
    assert audit["labeled_records"] == 11 and len(audit["issues"]) == 1
    with pytest.raises(ValueError, match="excluded records"):
        freeze_splits(audit, tmp_path / "split")
    freeze_splits(audit, tmp_path / "split", allow_exclusions=True)
    with (tmp_path / "split" / "train.jsonl").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="Frozen file changed"):
        load_frozen(tmp_path / "split", root)


def test_test_labels_cannot_expand_capacity_or_vocabulary(tmp_path):
    root = tmp_path / "data"
    writers = fixture_data(root)
    freeze_splits(audit_dataset(root, writers), tmp_path / "original")
    partitions, first_lock, _ = load_frozen(tmp_path / "original", root)
    for record in partitions["test"]:
        (root / record["label_path"]).write_text("☃" * 100, encoding="utf-8")
    freeze_splits(audit_dataset(root, writers), tmp_path / "changed_test")
    _, next_lock, tokenizer = load_frozen(tmp_path / "changed_test", root)
    assert first_lock["capacity"] == next_lock["capacity"]
    assert "☃" not in tokenizer.characters


@pytest.mark.parametrize("path", ["../x", "/x", "C:/x", "foo\\bar"])
def test_portable_path_rejects_escape(tmp_path, path):
    with pytest.raises(ValueError):
        resolve_relative(tmp_path, path)


def test_corruption_endpoints_and_pad_exclusion():
    targets = torch.tensor([[3, 4, 2, 0]])
    clean, mask = corrupt_tokens(targets, torch.zeros(1))
    assert torch.equal(clean, targets) and not mask.any()
    noisy, mask = corrupt_tokens(targets, torch.ones(1))
    assert noisy.tolist() == [[1, 1, 1, 0]]
    assert mask.tolist() == [[True, True, True, False]]
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    loss = diffusion_loss(logits, targets, mask, torch.ones(1))
    loss.backward()
    assert torch.equal(logits.grad[:, -1], torch.zeros(1, 5))
    assert loss.item() == pytest.approx(torch.log(torch.tensor(5.)).item())
    assert diffusion_loss(logits, targets, torch.zeros_like(mask), torch.ones(1)).item() == 0


def test_weighted_loss_exact_enumeration():
    # At T=2, enumerate mask events for a single token and both time levels.
    # E[masked CE/t] = CE; the zero-mask event must not be forced nonempty.
    logits = torch.tensor([[[0., 0., 0., 2.]]])
    target = torch.tensor([[3]])
    ce = torch.nn.functional.cross_entropy(logits[:, 0], target[:, 0])
    expected = 0.
    for t in (0.5, 1.):
        for flag in (False, True):
            probability = t if flag else 1 - t
            expected += 0.5 * probability * diffusion_loss(logits, target, torch.tensor([[flag]]), torch.tensor([t])).item()
    assert expected == pytest.approx(ce.item())


def test_reverse_kernel_carry_over_endpoints_and_frequency():
    tokens = torch.ones((1, 10000), dtype=torch.long)
    tokens[0, 0] = 3
    logits = torch.full((1, 10000, 5), -torch.inf)
    logits[:, :, 4] = 0
    result = reverse_step(tokens, logits, 1, 0.5, generator=torch.Generator().manual_seed(19))
    assert result[0, 0] == 3
    assert 0.48 < (result == 4).float().mean() < 0.52
    final = reverse_step(result, logits, 0.5, 0, generator=torch.Generator().manual_seed(19))
    assert not (final == 1).any() and final[0, 0] == 3


def test_mixed_width_padding_does_not_change_valid_features():
    seed_everything(7)
    tokenizer = CharacterTokenizer.build(["اب"])
    model = tiny_model(tokenizer).eval()
    config = ImageConfig(16, 128, 4)
    short, long = Image.new("RGB", (19, 16), "black"), Image.new("RGB", (63, 16), "blue")
    alone, aw = prepare_images([short], config)
    mixed, mw = prepare_images([short, long], config)
    with torch.no_grad():
        a, am = model.encoder(alone, aw)
        b, bm = model.encoder(mixed, mw)
        assert bm[0].sum() > 0 and not am.any()
        assert torch.allclose(a[0], b[0, ~bm[0]], atol=1e-5)
        noisy = torch.ones(1, model.config.capacity, dtype=torch.long)
        first = model.denoise(noisy, a, am, torch.ones(1))
        second = model.denoise(noisy, b[:1], bm[:1], torch.ones(1))
        assert torch.allclose(first[:, :, 2:], second[:, :, 2:], atol=1e-5)
    with pytest.raises(ValueError, match="width overflow"):
        prepare_images([Image.new("RGB", (500, 16))], config)


def test_cpu_forward_gradients_target_free_sampling_and_checkpoint(tmp_path):
    seed_everything(3)
    tokenizer = CharacterTokenizer.build(["مدرصة ١", "أ ب"])
    model = tiny_model(tokenizer)
    config = ImageConfig(16, 128, 4)
    pixels, widths = prepare_images([Image.new("RGB", (41, 16), "black"), Image.new("RGB", (23, 16), "white")], config)
    targets = torch.tensor([tokenizer.encode(text, 16) for text in ["مدرصة ١", "أ ب"]])
    loss = model(pixels, widths, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.encoder.patch.weight.grad.abs().sum() > 0
    assert model.head.weight.grad.abs().sum() > 0
    model.eval()
    ids, trace = model.sample(pixels, widths, steps=4, generator=torch.Generator().manual_seed(1), return_trace=True)
    assert len(trace) == 5 and set(trace[0][0]) == {tokenizer.MASK}
    assert not ((ids == tokenizer.MASK) | (ids == tokenizer.PAD)).any()
    for a, b in zip(trace, trace[1:]):
        a, b = torch.tensor(a), torch.tensor(b)
        assert torch.equal(a[a != tokenizer.MASK], b[a != tokenizer.MASK])
    save_checkpoint(tmp_path / "checkpoint", model, tokenizer, config, metadata={"test": True})
    loaded, loaded_tokenizer, loaded_config, _ = load_checkpoint(tmp_path / "checkpoint")
    loaded.eval()
    same, _ = loaded.sample(pixels, widths, steps=4, generator=torch.Generator().manual_seed(1))
    assert torch.equal(ids, same)
    assert loaded_tokenizer.tokens == tokenizer.tokens and loaded_config == config
    # Same-size reordered vocabulary must be rejected, not silently loaded.
    path = tmp_path / "checkpoint" / "tokenizer.json"
    altered = json.loads(path.read_text(encoding="utf-8"))
    altered["characters"].reverse()
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_checkpoint(tmp_path / "checkpoint")


def test_rng_restore_and_strict_metrics():
    import random
    import numpy as np
    seed_everything(9)
    state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    restore_rng(state)
    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])
    assert score_pair("أَ ١", "ا ٢")["char_distance"] == 3
    assert score_pair("أَ ١", "ا ٢", historical=True)["word_distance"] == 0
    assert score_pair("أَ ١", "ا ٢")["word_distance"] == 2
    assert score_pair("ا ب", "اب")["space_deletions"] == 1
    assert score_pair("ا", "ا  ")["space_insertions"] == 2
    assert correction_changes("مدرصة", "مدرصة", "مدرسة")["damaged_reference_chars"] == 1
    assert aggregate([("abc", "")])["cer"] == 1


def test_autoregressive_control_is_causal_and_has_matching_encoder():
    from dataclasses import replace
    tokenizer = CharacterTokenizer.build(["اب"])
    seed_everything(12)
    diffusion = tiny_model(tokenizer)
    seed_everything(12)
    control = DiffusionRecognizer(replace(diffusion.config, decoder_kind="autoregressive"))
    for key, value in diffusion.encoder.state_dict().items():
        assert torch.equal(value, control.encoder.state_dict()[key])
    pixels, widths = prepare_images([Image.new("RGB", (24, 16), "white")], ImageConfig(16, 128, 4))
    targets = torch.tensor([tokenizer.encode("اب", 16)])
    loss = control(pixels, widths, targets)
    loss.backward()
    assert torch.isfinite(loss) and control.encoder.patch.weight.grad.abs().sum() > 0
    control.eval()
    with torch.no_grad():
        memory, visual_mask = control.encoder(pixels, widths)
        inputs = torch.ones((1, 16), dtype=torch.long)
        before = control.denoise(inputs, memory, visual_mask, torch.zeros(1), causal=True)
        inputs[:, 5:] = tokenizer.ids["ا"]
        after = control.denoise(inputs, memory, visual_mask, torch.zeros(1), causal=True)
        assert torch.allclose(before[:, :5, 2:], after[:, :5, 2:], atol=1e-5)
    tokens, trace = control.sample(pixels, widths, return_trace=True)
    assert not tokens.eq(tokenizer.MASK).any()
    assert 2 <= len(trace) <= 17


def test_local_vit_adapter_with_tiny_random_weights(tmp_path):
    transformers = pytest.importorskip("transformers")
    vit_config = transformers.ViTConfig(image_size=16, patch_size=4, hidden_size=32, num_hidden_layers=1,
                                        num_attention_heads=4, intermediate_size=64)
    tokenizer = CharacterTokenizer.build(["اب"])
    config = ModelConfig(len(tokenizer), 8, d_model=32, heads=4, encoder_layers=1, decoder_layers=1,
                         patch_size=4, dropout=0, encoder_kind="vit", vit_config=vit_config.to_dict())
    seed_everything(3)
    model = DiffusionRecognizer(config)
    image_config = ImageConfig(16, 128, 4)
    short, long = Image.new("RGB", (19, 16), "blue"), Image.new("RGB", (61, 16), "black")
    pixels, widths = prepare_images([short, long], image_config)
    targets = torch.tensor([tokenizer.encode("اب", 8)] * 2)
    loss = model(pixels, widths, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.vit.embeddings.patch_embeddings.projection.weight.grad.abs().sum() > 0
    model.eval()
    alone, aw = prepare_images([short], image_config)
    with torch.no_grad():
        a, am = model.encoder(alone, aw)
        b, bm = model.encoder(pixels, widths)
        assert torch.allclose(a[0], b[0, ~bm[0]], atol=1e-5)
    save_checkpoint(tmp_path / "vit", model, tokenizer, image_config, metadata={})
    restored, _, _, _ = load_checkpoint(tmp_path / "vit")
    restored.eval()
    first, _ = model.sample(pixels, widths, steps=2, generator=torch.Generator().manual_seed(5))
    second, _ = restored.sample(pixels, widths, steps=2, generator=torch.Generator().manual_seed(5))
    assert torch.equal(first, second)


def run_cli(*args):
    import os
    import subprocess
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_epoch_resume_matches_uninterrupted_cpu_training(tmp_path):
    root = tmp_path / "data"
    writers = fixture_data(root)
    split = tmp_path / "split"
    freeze_splits(audit_dataset(root, writers), split)
    common = ["thesis/scripts/train_diffusion_line_ocr_model.py", "--dataset-root", root, "--split-root", split,
              "--batch-size", "2", "--height", "16", "--max-width", "128", "--patch-size", "4", "--d-model", "16",
              "--heads", "4", "--encoder-layers", "1", "--decoder-layers", "1", "--sampling-steps", "2", "--device", "cpu"]
    run_cli(*common, "--output-dir", tmp_path / "check", "--check-only")
    assert not (tmp_path / "check").exists()
    run_cli(*common, "--output-dir", tmp_path / "whole", "--epochs", "2")
    run_cli(*common, "--output-dir", tmp_path / "first", "--epochs", "1")
    run_cli(*common, "--output-dir", tmp_path / "resumed", "--epochs", "2", "--resume", tmp_path / "first" / "checkpoint-0001")
    a = torch.load(tmp_path / "whole" / "checkpoint-0002" / "model.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "checkpoint-0002" / "model.pt", weights_only=True)
    assert a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def test_runner_never_reads_labels_and_preserves_edge_spaces(tmp_path):
    root = tmp_path / "data"
    writers = fixture_data(root)
    split = tmp_path / "split"
    freeze_splits(audit_dataset(root, writers), split)
    _, lock, tokenizer = load_frozen(split, root)
    model = tiny_model(tokenizer, capacity=lock["capacity"])
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(-100)
        model.head.bias[tokenizer.ids[" "] - 2] = 100
    save_checkpoint(tmp_path / "model", model, tokenizer, ImageConfig(16, 128, 4), metadata={})
    labels = {}
    for path in root.rglob("ground_truth.txt"):
        labels[path] = path.read_bytes()
        path.unlink()
    output = tmp_path / "predictions"
    run_cli("pipelines/diffusion_line_run.py", "--model", tmp_path / "model", "--input-root", root,
            "--split-root", split, "--output-root", output, "--device", "cpu", "--steps", "2", "--trace")
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["failed"] == 0 and summary["completed"] == summary["selected"]
    assert summary["no_eos"] == summary["selected"]
    for path in output.rglob("prediction.txt"):
        assert path.read_bytes() == b" " * lock["capacity"] + b"\n"
    for path, content in labels.items():
        path.write_bytes(content)
    # Deliberately leave one prediction missing and verify the denominator remains fixed.
    next(output.rglob("prediction.txt")).unlink()
    run_cli("thesis/scripts/evaluate_diffusion_line_ocr.py", "--dataset-root", root, "--split-root", split,
            "--predictions", "fixture=" + str(output), "--output-dir", tmp_path / "evaluation")
    report = json.loads((tmp_path / "evaluation" / "evaluation.json").read_text(encoding="utf-8"))
    scores = report["models"]["fixture"]
    assert scores["missing"] == 1 and scores["strict_all_expected"]["lines"] == scores["expected"]
