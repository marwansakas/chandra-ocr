from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "thesis/scripts"))
import generate_synthetic_lines as generator
import synthetic_training
import train_line_ocr_model as trainer


def make_real(root, index):
    name = f"{index:06d}_0001_0001_School__Notebook_{index}_Page_001_line_001"
    sample = root / name
    sample.mkdir(parents=True)
    image = Image.new("RGB", (200, 70), "white")
    ImageDraw.Draw(image).line((25, 35, 160, 35), fill="black", width=3)
    image.save(sample / "image.png")
    (sample / "ground_truth.txt").write_text("هذا كتاب جميل\n", encoding="utf-8")
    return sample


def make_training(tmp_path):
    real = tmp_path / "real"
    make_real(real, 1)
    rows = trainer.read_line_records(real)
    manifest = tmp_path / "train.jsonl"
    trainer.write_jsonl(manifest, rows)
    return rows, manifest


def test_label_whitespace_is_normalized():
    assert generator.clean_text("  هَذا\tكتاب\r\nجميل  ") == "هَذا كتاب جميل"


def test_invalid_control_identifies_source(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"sample_id": "bad-label", "text": "word\x00word",
                                    "image_path": "unused.png"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad-label.*U\\+0000"):
        generator.read_training_sources(manifest)


def test_normalized_source_still_passes_training_validation(tmp_path):
    rows, manifest = make_training(tmp_path)
    rows[0].text = "هذا\tكتاب\nجميل"
    trainer.write_jsonl(manifest, rows)
    output = tmp_path / "generated"
    generator.generate(generator.parse_args(["--mode", "augment", "--train-manifest", str(manifest),
                                             "--output-root", str(output), "--count", "1"]))
    loaded = synthetic_training.load_synthetic_records(output, rows, [], trainer.LineRecord)
    assert loaded[0].text == "هذا كتاب جميل"


def generate_augmented(tmp_path):
    rows, manifest = make_training(tmp_path)
    output = tmp_path / "generated"
    args = generator.parse_args(["--mode", "augment", "--train-manifest", str(manifest),
                                 "--output-root", str(output), "--count", "3"])
    generator.generate(args)
    return rows, output


def test_render_preserves_logical_labels_and_reproduces_pixels(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("ذَهَبَ أحمد إلى المدرسة ١٢.\nهذه حديقة جميلة\n", encoding="utf-8")
    outputs = []
    for name in ("one", "two"):
        root = tmp_path / name
        generator.generate(generator.parse_args(["--text-file", str(corpus), "--count", "4",
                                                   "--output-root", str(root), "--shaping", "fallback"]))
        outputs.append(root)
    first_rows = [json.loads(line) for line in (outputs[0] / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["text"] for r in first_rows} == set(corpus.read_text(encoding="utf-8").splitlines())
    for row in first_rows:
        left = outputs[0] / row["image_path"]
        right = outputs[1] / row["image_path"]
        assert left.read_bytes() == right.read_bytes()
        assert left.parent.joinpath("ground_truth.txt").read_text(encoding="utf-8").strip() == row["text"]
        assert row["synthetic"] and row["split"] == "train"
        assert row["shaping"] == "reshaper-bidi"
        with Image.open(left) as image:
            assert image.width > 50 and image.height > 20
            assert image.convert("L").getextrema()[0] < 100
    assert (outputs[0] / "preview.jpg").is_file()


def test_arabic_fallback_joins_letters_without_reversing_labels():
    renderer = generator.ArabicRenderer(generator.DEFAULT_FONTS, "fallback")
    logical = "هذا كتاب"
    assert renderer.display_text(logical) != logical
    assert any(0xFB50 <= ord(c) <= 0xFEFF for c in renderer.display_text(logical))
    assert logical == "هذا كتاب"


@pytest.mark.skipif(not features.check_feature("raqm"), reason="RAQM unavailable in this Pillow build")
def test_raqm_preserves_arabic_and_renders():
    renderer = generator.ArabicRenderer(generator.DEFAULT_FONTS, "raqm")
    text = "ذهبت مريم إلى المدرسة"
    assert renderer.display_text(text) == text
    image, info = renderer.render(text, random.Random(8))
    assert image.width > 50 and info["shaping"] == "raqm"


def test_augmentation_tracks_source_and_is_training_only(tmp_path):
    own, output = generate_augmented(tmp_path)
    records = synthetic_training.load_synthetic_records(output, own, [], trainer.LineRecord, limit=2)
    assert len(records) == 2
    assert all(r.status == "synthetic_train" and r.text == own[0].text for r in records)
    with pytest.raises(ValueError, match="--synthetic-root"):
        trainer.read_line_records(output)


def test_held_out_source_is_rejected(tmp_path):
    own, output = generate_augmented(tmp_path)
    with pytest.raises(ValueError, match="current training split"):
        synthetic_training.load_synthetic_records(output, [], own, trainer.LineRecord)


def test_changed_source_label_is_rejected(tmp_path):
    own, output = generate_augmented(tmp_path)
    own[0].text = "نص آخر"
    with pytest.raises(ValueError, match="transcription changed"):
        synthetic_training.load_synthetic_records(output, own, [], trainer.LineRecord)


def test_changed_source_image_is_rejected(tmp_path):
    own, output = generate_augmented(tmp_path)
    Image.new("RGB", (100, 20), "black").save(own[0].image_path)
    with pytest.raises(ValueError, match="Source image changed"):
        synthetic_training.load_synthetic_records(output, own, [], trainer.LineRecord)


def test_label_tampering_is_rejected(tmp_path):
    own, output = generate_augmented(tmp_path)
    next(output.glob("synthetic_*/ground_truth.txt")).write_text("خطأ", encoding="utf-8")
    with pytest.raises(ValueError, match="label/manifest mismatch"):
        synthetic_training.load_synthetic_records(output, own, [], trainer.LineRecord)


def test_incomplete_generation_cannot_be_used(tmp_path):
    own, output = generate_augmented(tmp_path)
    config = json.loads((output / "run.json").read_text())
    config["complete"] = False
    generator.save_json(output / "run.json", config)
    with pytest.raises(ValueError, match="incomplete"):
        synthetic_training.load_synthetic_records(output, own, [], trainer.LineRecord)


def test_existing_output_not_overwritten(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "important.txt"
    marker.write_text("keep me")
    with pytest.raises(ValueError, match="not empty"):
        generator.generate(generator.parse_args(["--output-root", str(output)]))
    assert marker.read_text() == "keep me"


@pytest.mark.parametrize("name", ["test.jsonl", "validation.jsonl", "val.jsonl"])
def test_test_manifest_rejected(tmp_path, name):
    manifest = tmp_path / name
    manifest.write_text("")
    with pytest.raises(ValueError, match="test/validation"):
        generator.generate(generator.parse_args(["--mode", "augment", "--train-manifest", str(manifest),
                                                  "--output-root", str(tmp_path / "output")]))


@pytest.mark.parametrize("script", ["train_line_ocr_model.py", "train_cnn_trocr_model.py"])
def test_training_cli_adds_generated_samples_only_to_train(tmp_path, script):
    real = tmp_path / "real"
    for index in range(1, 7):
        make_real(real, index)
    records = trainer.read_line_records(real)
    train, held_out = trainer.split_records(records, 0.8, 42, "student")
    manifest = tmp_path / "train.jsonl"
    trainer.write_jsonl(manifest, train)
    generated = tmp_path / "generated"
    generator.generate(generator.parse_args(["--mode", "augment", "--train-manifest", str(manifest),
                                              "--output-root", str(generated), "--count", "7"]))
    splits = tmp_path / "splits"
    result = subprocess.run([sys.executable, "-B", str(ROOT / "thesis/scripts" / script),
                             "--prepare-only", "--dataset-root", str(real),
                             "--split-root", str(splits), "--synthetic-root", str(generated)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    combined = [json.loads(s) for s in (splits / "train_with_synthetic.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [json.loads(s) for s in (splits / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(combined) == len(train) + 7
    assert {r["sample_id"] for r in test} == {r.sample_id for r in held_out}
    assert not any(r["status"] == "synthetic_train" for r in test)


def test_font_sources_match_recorded_hashes():
    manifest = json.loads((generator.DEFAULT_FONTS / "sources.json").read_text())
    for record in manifest["files"]:
        assert hashlib.sha256((generator.DEFAULT_FONTS / record["file"]).read_bytes()).hexdigest() == record["sha256"]
