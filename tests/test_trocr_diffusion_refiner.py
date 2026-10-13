"""Offline mathematical, leakage, artifact, and real tiny-HF integration checks."""

import json
from pathlib import Path
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thesis.diffusion.data import audit_dataset, file_hash, freeze_splits, load_frozen
from thesis.diffusion.tokenizer import CharacterTokenizer
from thesis.refiner.cache import CachedLines, base_source, save_json
from thesis.refiner.cli import choose_strategy, load_run, main
from thesis.refiner.model import DiffusionRefiner, RefinerConfig, encode_draft
from thesis.refiner.processor import load_processor

torch.set_num_threads(2)


def fixture_split(tmp_path):
    root = tmp_path / "data"
    writers = {}
    for i in range(12):
        sample = f"line_{i:03d}"
        folder = root / sample
        folder.mkdir(parents=True)
        Image.new("RGB", (20 + i, 16), (i * 19, 70, 120)).save(folder / "image.png")
        (folder / "ground_truth.txt").write_text("أب" if i % 2 else "بت", encoding="utf-8")
        writers[sample] = f"child_{i // 2}"
    split = tmp_path / "split"
    freeze_splits(audit_dataset(root, writers), split)
    return root, split


def test_diffusion_learns_with_frozen_features_and_reproducible_sampling():
    torch.manual_seed(7)
    tokenizer = CharacterTokenizer(["أ", "ب", "ت"])
    config = RefinerConfig(len(tokenizer), 4, 8, d_model=16, heads=2, layers=1,
                           dropout=0, draft_dropout=0, full_mask_probability=1)
    model = DiffusionRefiner(config)
    visual = torch.randn(2, 5, 8)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    draft = torch.tensor([encode_draft(tokenizer, "بت", 4)] * 2)
    targets = torch.tensor([tokenizer.encode("أب", 4)] * 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    initial = float(model(visual, mask, draft, targets).detach())
    for _ in range(60):
        optimizer.zero_grad()
        loss = model(visual, mask, draft, targets)
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial * 0.1
    assert visual.grad is None
    model.eval()
    a = model.sample(visual, mask, draft, steps=4, generator=torch.Generator().manual_seed(21))
    b = model.sample(visual, mask, draft, steps=4, generator=torch.Generator().manual_seed(21))
    assert torch.equal(a, b)
    assert (a >= tokenizer.EOS).all()
    assert tokenizer.decode(model.sample(visual, mask, draft, mode="one_pass")[0])[0] == "أب"


def test_draft_noise_never_changes_strict_target_contract():
    tokenizer = CharacterTokenizer(["ب"])
    assert encode_draft(tokenizer, "☃بببب", 3) == [tokenizer.MASK, tokenizer.ids["ب"], tokenizer.EOS]
    with pytest.raises(ValueError, match="Out-of-vocabulary"):
        tokenizer.encode("☃", 3)
    with pytest.raises(ValueError, match="overflow"):
        tokenizer.encode("ببب", 3)
    assert choose_strategy(0.1, 0.1, "diffusion") == "baseline"
    assert choose_strategy(0.1, 0.11, "diffusion") == "baseline"
    assert choose_strategy(0.1, 0.09, "diffusion") == "diffusion"


def test_base_training_leakage_rejected(tmp_path):
    root, split = fixture_split(tmp_path)
    parts, _, _ = load_frozen(split, root)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(parts["test"][0], ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="held-out"):
        base_source("model/name", "a" * 40, manifest, False, parts)
    manifest.write_text(json.dumps(parts["train"][0], ensure_ascii=False) + "\n", encoding="utf-8")
    source, _ = base_source("model/name", "a" * 40, manifest, False, parts)
    assert source["provenance"]["kind"] == "project_train_only"
    with pytest.raises(ValueError, match="immutable"):
        base_source("model/name", "main", None, True, parts)


def test_prepare_cli_and_validation_requirement(tmp_path):
    root, original = fixture_split(tmp_path)
    parts, _, _ = load_frozen(original, root)
    mapping = {row["sample_id"]: row["writer_id"] for records in parts.values() for row in records}
    writer_map = tmp_path / "writers.json"
    save_json(writer_map, mapping)
    writer_split = tmp_path / "writer_split"
    main(["prepare", "--dataset-root", str(root), "--split-root", str(writer_split), "--writer-map", str(writer_map)])
    _, writer_lock, _ = load_frozen(writer_split, root)
    assert writer_lock["split_unit"].startswith("explicit writer")
    line_split = tmp_path / "line_split"
    main(["prepare", "--dataset-root", str(root), "--split-root", str(line_split), "--split-kind", "line"])
    _, line_lock, _ = load_frozen(line_split, root)
    assert line_lock["writer_isolation"] is False
    with pytest.raises(ValueError, match="writer-map"):
        main(["prepare", "--dataset-root", str(root), "--split-root", str(tmp_path / "missing_map")])
    from thesis.diffusion.data import freeze_line_pilot
    from thesis.refiner.cache import frozen_data
    pilot = tmp_path / "pilot"
    freeze_line_pilot(audit_dataset(root), pilot)
    with pytest.raises(ValueError, match="three-way"):
        frozen_data(root, pilot)


def tiny_hf_checkpoint(path):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import (RobertaTokenizerFast, TrOCRConfig, TrOCRProcessor,
                              ViTConfig, ViTImageProcessor, VisionEncoderDecoderConfig,
                              VisionEncoderDecoderModel)

    path.mkdir()
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(["أب بت school handwriting"], vocab_size=270,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.save_model(str(path))
    bpe.save(str(path / "tokenizer.json"))
    tokenizer = RobertaTokenizerFast(tokenizer_file=str(path / "tokenizer.json"),
                                     bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")
    processor = TrOCRProcessor(image_processor=ViTImageProcessor(size={"height": 16, "width": 16}), tokenizer=tokenizer)
    encoder = ViTConfig(image_size=16, patch_size=4, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32)
    decoder = TrOCRConfig(vocab_size=len(tokenizer), d_model=16, decoder_layers=1,
                          decoder_attention_heads=2, decoder_ffn_dim=32, cross_attention_hidden_size=16,
                          max_position_embeddings=32, pad_token_id=1, eos_token_id=2, decoder_start_token_id=0)
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
    config.decoder_start_token_id, config.pad_token_id, config.eos_token_id = 0, 1, 2
    model = VisionEncoderDecoderModel(config)
    model.save_pretrained(path)
    processor.save_pretrained(path)


@pytest.mark.parametrize("legacy_tokenizer", [False, True], ids=["modern_tokenizer_json", "legacy_vocab_merges"])
def test_actual_hf_cache_train_test_pipeline(tmp_path, legacy_tokenizer):
    """Runs actual tiny TrOCR encoder/generation, not a mocked cache extractor."""
    root, split = fixture_split(tmp_path)
    base, cache, run, report = [tmp_path / name for name in ("base", "cache", "run", "report")]
    tiny_hf_checkpoint(base)
    if legacy_tokenizer:
        # Match the released microsoft/trocr-base-handwritten layout: no
        # tokenizer.json, and an explicit RobertaTokenizer in the metadata.
        (base / "tokenizer.json").unlink()
        tokenizer_config = json.loads((base / "tokenizer_config.json").read_text(encoding="utf-8"))
        tokenizer_config["tokenizer_class"] = "RobertaTokenizer"
        save_json(base / "tokenizer_config.json", tokenizer_config)
        from transformers import RobertaTokenizer
        expected = RobertaTokenizer.from_pretrained(base, local_files_only=True)
        loaded = load_processor(str(base), local_files_only=True, trust_remote_code=False)
        for text in ("hello", "أب بت", "  punctuation, 123!"):
            assert loaded.tokenizer.encode(text) == expected.encode(text)
        assert loaded.tokenizer.get_vocab() == expected.get_vocab()
    weight_hash = file_hash(base / "model.safetensors")
    common = ["--dataset-root", str(root), "--split-root", str(split), "--cache-root", str(cache), "--device", "cpu"]
    cache_options = ["--base-model", str(base), "--external-base", "--max-new-tokens", "3", "--batch-size", "2"]
    main(["cache", *common, *cache_options])
    assert not (cache / "test").exists()
    # Check resumability without changing frozen features.
    before = file_hash(cache / "train" / "manifest.json")
    main(["cache", *common, *cache_options, "--resume"])
    assert file_hash(cache / "train" / "manifest.json") == before
    main(["train", *common, "--output-dir", str(run), "--epochs", "2", "--batch-size", "2",
          "--d-model", "16", "--heads", "2", "--layers", "1", "--sampling-steps", "2"])
    assert file_hash(base / "model.safetensors") == weight_hash
    assert (run / "completed.json").is_file()
    main(["cache", *common, *cache_options, "--partition", "test"])
    main(["evaluate", *common, "--run-dir", str(run), "--output-dir", str(report)])
    result = json.loads((report / "evaluation.json").read_text())
    assert result["partition"] == "test"
    assert set(result["metrics"]) == {"baseline", "one_pass", "diffusion", "selected"}
    assert result["lines"] == 2
    assert result["metrics"]["selected"] == result["metrics"][result["selection"]["strategy"]]
    with pytest.raises(FileExistsError):
        main(["evaluate", *common, "--run-dir", str(run), "--output-dir", str(report)])
    # Inference does not require test targets to fit capacity or vocabulary.
    parts, _, _ = load_frozen(split, root)
    changed_records = [{**r, "text": "☃" * 30} for r in parts["test"]]
    data = CachedLines(cache, "test", changed_records, file_hash(split / "split_lock.json"))
    from thesis.refiner.cli import infer
    model, tokenizer, _, selection = load_run(run, torch.device("cpu"))
    predictions = infer(model, data, tokenizer, torch.device("cpu"), batch_size=2, steps=2, seed=selection["seed"])
    assert all(row["capacity_overflow"] for row in predictions)
    assert all(row["target_unknown_codepoints"] == ["☃"] for row in predictions)
    manifest = json.loads((cache / "test" / "manifest.json").read_text())
    tensor = cache / "test" / manifest["entries"][0]["file"]
    with tensor.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        CachedLines(cache, "test", parts["test"], file_hash(split / "split_lock.json"))
    save_json(run / "selection.json", {"strategy": "tampered"})
    with pytest.raises(ValueError, match="integrity"):
        load_run(run, torch.device("cpu"))


def test_processor_does_not_hide_unrelated_loading_errors(monkeypatch):
    from transformers import TrOCRProcessor

    def fail(*args, **kwargs):
        raise ValueError("unrelated processor configuration error")

    monkeypatch.setattr(TrOCRProcessor, "from_pretrained", fail)
    with pytest.raises(ValueError, match="unrelated processor"):
        load_processor("unused", local_files_only=True)


def test_processor_never_substitutes_unknown_tokenizers(tmp_path, monkeypatch):
    from transformers import TrOCRProcessor
    from transformers.utils import hub

    config = tmp_path / "tokenizer_config.json"
    save_json(config, {"tokenizer_class": "CustomArabicTokenizer"})

    def fail(*args, **kwargs):
        raise ValueError("Couldn't instantiate the backend tokenizer")

    monkeypatch.setattr(TrOCRProcessor, "from_pretrained", fail)
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(config))
    with pytest.raises(ValueError, match="backend tokenizer"):
        load_processor("unused", local_files_only=True)
