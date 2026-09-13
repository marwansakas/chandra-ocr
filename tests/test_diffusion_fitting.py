"""Fitting subsets stay TRAIN-only and checkpoint evaluation uses exact membership."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thesis.diffusion.data import audit_dataset, file_hash, freeze_line_pilot, json_bytes, load_frozen
from thesis.diffusion.subsets import make_subset, page_id, read_subset, run_training_records
from test_diffusion_pilot import cli, pilot_fixture


def test_varied_selection_is_order_independent_and_page_balanced():
    records = [{"sample_id": f"{i + 1:06d}_{i // 4 + 1:04d}_notebook_line_{i:03d}", "text": "ب" * (i + 1)} for i in range(32)]
    records[7]["text"] = "ـبت"
    records[12]["text"] = "بت\nبت"
    records[17]["text"] = "بَت"
    before = copy.deepcopy(records)
    subset = make_subset(records, 16, 42, "fingerprint", "split")
    assert subset == make_subset(list(reversed(records)), 16, 42, "fingerprint", "split")
    assert records == before
    assert len(set(subset["sample_ids"])) == 16
    assert subset["stats"]["pages"] == 8
    assert all(subset["stats"][key] >= 1 for key in ("tatweel_crops", "multiline_crops", "harakat_crops"))
    assert records[0]["sample_id"] in subset["sample_ids"] and records[-1]["sample_id"] in subset["sample_ids"]
    with pytest.raises(ValueError, match="format"):
        page_id({"sample_id": "unknown"})


@pytest.mark.parametrize("change", ["test_id", "duplicate", "split", "count"])
def test_subset_rejects_invalid_membership_or_provenance(tmp_path, change):
    records = [{"sample_id": f"{i + 1:06d}_0001_line_{i}", "text": "بت"} for i in range(8)]
    manifest = make_subset(records, 4, 42, "fp", "hash")
    if change == "test_id":
        manifest["sample_ids"][0] = "not_in_train"
    elif change == "duplicate":
        manifest["sample_ids"][0] = manifest["sample_ids"][1]
    elif change == "split":
        manifest["split_lock_sha256"] = "different"
    else:
        manifest["sample_ids"].pop()
    path = tmp_path / "subset.json"
    path.write_bytes(json_bytes(manifest))
    with pytest.raises(ValueError):
        read_subset(path, records, "fp", "hash", 4)


def test_greedy_subset_training_replay_and_resume(tmp_path):
    dataset, inventory = pilot_fixture(tmp_path)
    for index, folder in enumerate(inventory["ground_folders"]):
        (dataset / folder["title"] / "ground_truth.txt").write_bytes(("أب" * (1 + index % 3)).encode("utf-8"))
    split = tmp_path / "split"
    freeze_line_pilot(audit_dataset(dataset), split)
    parts, lock, _ = load_frozen(split, dataset)
    manifest = make_subset(parts["train"], 6, 42, lock["dataset_fingerprint"], file_hash(split / "split_lock.json"))
    # Deliberately differs from the old first-N behavior, including order.
    manifest["sample_ids"] = [r["sample_id"] for r in reversed(parts["train"][-6:])]
    path = tmp_path / "subset.json"
    path.write_bytes(json_bytes(manifest))
    common = ("thesis/scripts/train_diffusion_line_ocr_model.py", "--dataset-root", dataset, "--split-root", split,
              "--overfit-samples", 6, "--overfit-manifest", path, "--batch-size", 2,
              "--height", 16, "--max-width", 64, "--patch-size", 4, "--d-model", 16, "--heads", 4,
              "--encoder-layers", 1, "--decoder-layers", 1, "--dropout", 0, "--weight-decay", 0,
              "--eos-target", "single", "--full-mask-probability", .25, "--diagnostics-every", 1,
              "--evaluation-decoding", "greedy_one_pass", "--device", "cpu")
    whole, first, resumed = (tmp_path / n for n in ("whole", "first", "resumed"))
    cli(*common, "--output-dir", whole, "--epochs", 2)
    cli(*common, "--output-dir", first, "--epochs", 1)
    cli(*common, "--output-dir", resumed, "--epochs", 2, "--resume", first / "checkpoint-0001")
    a = torch.load(whole / "checkpoint-0002/model.pt", weights_only=True)
    b = torch.load(resumed / "checkpoint-0002/model.pt", weights_only=True)
    assert all(torch.equal(a[key], b[key]) for key in a)
    cli("thesis/scripts/diagnose_diffusion_fitting.py", "--run", whole, "--dataset-root", dataset,
        "--output-dir", whole / "final_fitting_diagnosis", "--device", "cpu")
    predictions = [json.loads(s) for s in (whole / "final_fitting_diagnosis/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["sample_id"] for r in predictions] == manifest["sample_ids"]
    report = json.loads(cli("thesis/scripts/report_diffusion_fitting.py", "--run", whole).stdout)
    assert report["final_replay_matches_epoch_metrics"] is True
    assert report["curve"][-1]["epoch"] == 2
    run = json.loads((whole / "run.json").read_text())
    assert run["sampler"] == "greedy_one_pass"
    assert run["training_options"]["training_subset_sha256"] == file_hash(path)
    assert not set(manifest["sample_ids"]) & {r["sample_id"] for r in parts["test"]}
    changed = copy.deepcopy(manifest)
    changed["sample_ids"].reverse()
    path.write_bytes(json_bytes(changed))
    result = subprocess.run([sys.executable, *map(str, common), "--output-dir", str(tmp_path / "bad_resume"),
                             "--epochs", "2", "--resume", str(first / "checkpoint-0001")], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "settings changed" in result.stderr
    (whole / "training_subset.json").write_bytes(json_bytes(changed))
    with pytest.raises(ValueError, match="subset changed"):
        run_training_records(whole, run, parts["train"])
