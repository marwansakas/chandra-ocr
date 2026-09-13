"""The explicit two-way pilot keeps image/text pairs intact and test labels out of fitting."""

import json
import csv
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thesis.diffusion.data import audit_dataset, freeze_line_pilot, load_frozen
from thesis.diffusion.runtime import checkpoint_path
from thesis.scripts.prepare_diffusion_pilot import prepare


def pilot_fixture(root, count=20):
    dataset = root / "ground_truth_lines"
    inventory = {"selection": [], "ground_folders": [], "images": [], "labels": [], "surya_line_folders": []}
    for page in range(2):
        inventory["selection"].append({"id": f"page{page}", "title": f"{page + 1:04d}_notebook"})
    for index in range(count):
        page = index % 2
        name = f"{index + 1:06d}_{page + 1:04d}_notebook_line_{index + 1:03d}"
        folder = dataset / name
        folder.mkdir(parents=True)
        Image.new("RGB", (20 + index, 16), (index * 10, 120, 210)).save(folder / "image.png")
        (folder / "ground_truth.txt").write_bytes("أ ب ١\n".encode("utf-8"))
        inventory["ground_folders"].append({"id": name, "title": name})
        inventory["surya_line_folders"].append({"id": name, "parent_ids": [f"page{page}"]})
        for key, filename in (("images", "image.png"), ("labels", "ground_truth.txt")):
            inventory[key].append({"id": name + filename, "title": filename, "parent_ids": [name],
                                   "size": str((folder / filename).stat().st_size)})
    (root / "drive_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    return dataset, inventory


def cli(*args):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_prepare_counts_pairs_and_seed_reproducibility(tmp_path):
    dataset, _ = pilot_fixture(tmp_path)
    stats = prepare(tmp_path, tmp_path / "split", tmp_path / "stats")
    assert (stats["selected_pages"], stats["usable_labeled_lines"], stats["training_lines"], stats["test_lines"]) == (2, 20, 18, 2)
    assert stats["writer_isolation"] is False and stats["pages_in_both_partitions"] >= 1
    parts, lock, _ = load_frozen(tmp_path / "split", dataset)
    assert parts["validation"] == [] and lock["evaluation_policy"] == "fixed_epochs_final_checkpoint_no_validation"
    for rows in parts.values():
        for record in rows:
            assert Path(record["image_path"]).parent == Path(record["label_path"]).parent
    freeze_line_pilot(audit_dataset(dataset), tmp_path / "second")
    assert (tmp_path / "split" / "train.jsonl").read_bytes() == (tmp_path / "second" / "train.jsonl").read_bytes()
    cli("thesis/scripts/report_diffusion_stats.py", "--pilot-root", tmp_path)


def test_all_views_training_epoch_uses_three_examples_per_crop_and_compact_log(tmp_path):
    dataset, _ = pilot_fixture(tmp_path, count=4)
    split, run = tmp_path / "split", tmp_path / "run"
    freeze_line_pilot(audit_dataset(dataset), split)
    parts, _, _ = load_frozen(split, dataset)
    result = cli(
        "thesis/scripts/train_diffusion_line_ocr_model.py",
        "--dataset-root", dataset, "--split-root", split, "--output-dir", run,
        "--epochs", 1, "--batch-size", 2, "--height", 16, "--max-width", 64,
        "--patch-size", 4, "--d-model", 16, "--heads", 4, "--encoder-layers", 1,
        "--decoder-layers", 1, "--device", "cpu", "--augmentation", "trocr_all_views_v1",
        "--early-stopping-patience", 0, "--epoch-log-format", "compact",
    )
    row = json.loads((run / "training_metrics.jsonl").read_text(encoding="utf-8").strip())
    expected_examples = 3 * len(parts["train"])
    assert row["optimizer_updates"] == (expected_examples + 1) // 2
    assert result.stdout.startswith("Epoch 001 | Train loss ")
    assert "| Validation loss n/a | CER n/a | Exact lines n/a | Time " in result.stdout
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert metadata["training_options"]["training_views_per_sample"] == 3
    assert metadata["training_options"]["training_examples_per_epoch"] == expected_examples


def test_prepare_rejects_missing_image(tmp_path):
    dataset, inventory = pilot_fixture(tmp_path)
    (dataset / inventory["ground_folders"][0]["title"] / "image.png").unlink()
    with pytest.raises(ValueError, match="Download incomplete"):
        prepare(tmp_path, tmp_path / "split", tmp_path / "stats")
    assert not (tmp_path / "split").exists()


def test_prepare_preserves_stacked_crop_as_one_pair(tmp_path):
    dataset, inventory = pilot_fixture(tmp_path)
    text = "ماضي\nزار المريض"
    for index, folder in enumerate(inventory["ground_folders"]):
        label = dataset / folder["title"] / "ground_truth.txt"
        label.write_bytes(text.encode("utf-8"))
        inventory["labels"][index]["size"] = str(label.stat().st_size)
    (tmp_path / "drive_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    stats = prepare(tmp_path, tmp_path / "split", tmp_path / "stats")
    assert stats["labels_with_internal_line_breaks"] == 20
    assert (stats["usable_labeled_lines"], stats["training_lines"], stats["test_lines"]) == (20, 18, 2)
    parts, lock, tokenizer = load_frozen(tmp_path / "split", dataset)
    assert all(record["text"] == text for rows in parts.values() for record in rows)
    assert tokenizer.decode(tokenizer.encode(text, lock["capacity"])) == (text, True)
    with (tmp_path / "stats/image_text_pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 20 and all(pair["text"] == text for pair in pairs)


def test_prepare_excludes_ignore_and_empty_labels(tmp_path):
    dataset, inventory = pilot_fixture(tmp_path)
    for index, text in enumerate(("IGNORE", "   \n")):
        name = inventory["ground_folders"][index]["title"]
        path = dataset / name / "ground_truth.txt"
        path.write_bytes(text.encode("utf-8"))
        inventory["labels"][index]["size"] = str(path.stat().st_size)
    (tmp_path / "drive_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    stats = prepare(tmp_path, tmp_path / "split", tmp_path / "stats")
    assert stats["ground_truth_line_folders"] == 20 and stats["usable_labeled_lines"] == 18
    assert (stats["training_lines"], stats["test_lines"]) == (16, 2)
    assert len(stats["excluded_empty_or_ignore"]) == 2


def test_pilot_test_labels_do_not_change_training_tokenizer_or_capacity(tmp_path):
    dataset, _ = pilot_fixture(tmp_path)
    freeze_line_pilot(audit_dataset(dataset), tmp_path / "first")
    parts, first_lock, first_tokenizer = load_frozen(tmp_path / "first", dataset)
    for record in parts["test"]:
        (dataset / record["label_path"]).write_text("☃" * 100, encoding="utf-8")
    freeze_line_pilot(audit_dataset(dataset), tmp_path / "second")
    _, second_lock, second_tokenizer = load_frozen(tmp_path / "second", dataset)
    assert first_lock["capacity"] == second_lock["capacity"]
    assert first_tokenizer.as_dict() == second_tokenizer.as_dict()


@pytest.mark.parametrize("monitor_train", [False, True])
def test_fixed_epoch_training_then_test_and_stats(tmp_path, monitor_train):
    dataset, inventory = pilot_fixture(tmp_path)
    # Real curated crops can include stacked writing: preserve LF through training and evaluation.
    for folder in inventory["ground_folders"]:
        (dataset / folder["title"] / "ground_truth.txt").write_bytes("ماضي\nزار المريض\n".encode("utf-8"))
    split, run = tmp_path / "split", tmp_path / "run"
    freeze_line_pilot(audit_dataset(dataset), split)
    cli("thesis/scripts/train_diffusion_line_ocr_model.py", "--dataset-root", dataset, "--split-root", split,
        "--output-dir", run, "--epochs", 2, "--batch-size", 2, "--height", 16, "--max-width", 64,
        "--patch-size", 4, "--d-model", 16, "--heads", 4, "--encoder-layers", 1, "--decoder-layers", 1,
        "--device", "cpu", *(["--monitor-train", "--evaluation-decoding", "greedy_one_pass", "--diagnostics-every", "1"] if monitor_train else []))
    metrics = [json.loads(line) for line in (run / "training_metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 2
    if monitor_train:
        assert all(row["metrics_partition"] == "train" and row["metrics"]["strict"]["lines"] == 18 for row in metrics)
        assert metrics[-1]["full_mask_diagnostics"]["first_eos"]["positions"] == 18
    else:
        assert all(row["metrics"] is None for row in metrics)
        assert not list(run.glob("validation-*.jsonl"))
    assert not (run / "best.json").exists()
    assert checkpoint_path(run) == run / "checkpoint-0002"
    assert json.loads((run / "run.json").read_text())["mode"] == "fixed_epochs_line_pilot"
    cli("thesis/scripts/train_diffusion_line_ocr_model.py", "--dataset-root", dataset, "--split-root", split,
        "--output-dir", run, "--check-only")
    predictions, evaluation = tmp_path / "predictions", tmp_path / "evaluation"
    cli("pipelines/diffusion_line_run.py", "--model", run, "--input-root", dataset, "--split-root", split,
        "--partition", "test", "--output-root", predictions, "--device", "cpu", "--steps", 2,
        *(["--decoding", "greedy_one_pass"] if monitor_train else []))
    cli("thesis/scripts/evaluate_diffusion_line_ocr.py", "--dataset-root", dataset, "--split-root", split,
        "--predictions", "pilot=" + str(predictions), "--output-dir", evaluation)
    result = json.loads((evaluation / "evaluation.json").read_text())["models"]["pilot"]
    assert result["expected"] == 2 and result["missing"] == 0
    cli("thesis/scripts/report_diffusion_stats.py", "--run", run, "--evaluation", evaluation)
    if monitor_train:
        cli("thesis/scripts/diagnose_diffusion_fitting.py", "--run", run, "--dataset-root", dataset,
            "--output-dir", run / "final_fitting_diagnosis", "--device", "cpu")
        report = json.loads(cli("thesis/scripts/report_diffusion_fitting.py", "--run", run, "--evaluation", evaluation,
                                "--format", "json").stdout)
        assert report["partitions"]["train"] == 18
        assert report["test"]["models"]["pilot"]["expected"] == 2
        assert report["final_replay_matches_epoch_metrics"] is True
