import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thesis.scripts.compare_diffusion_runs import (  # noqa: E402
    COMPARISON_FIELDS,
    comparison_evaluation,
    load_run,
    validate_comparison,
)
from thesis.scripts.diffusion_stats_table import format_evaluation_table  # noqa: E402


class CompareDiffusionRunsTests(unittest.TestCase):
    def make_run(
        self, root: Path, *, dropout=0.0, split="split", augmentation="none", weight_decay=0.0,
        early_stopping_patience=30
    ) -> Path:
        args = {field: 1 for field in COMPARISON_FIELDS}
        args.update({
            "augmentation": augmentation,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "vit_checkpoint": None,
        })
        metadata = {
            "arguments": args,
            "dataset_fingerprint": "dataset",
            "split_lock_sha256": split,
        }
        scores = {
            "cer": 0.25,
            "wer": 0.5,
            "char_distance": 25,
            "reference_chars": 100,
            "word_distance": 10,
            "reference_words": 20,
            "exact_lines": 2,
            "lines": 4,
        }
        evaluation = {
            "partition": "test",
            "dataset_fingerprint": "dataset",
            "split_lock_sha256": split,
            "models": {"diffusion": {
                "completed": 4,
                "expected": 4,
                "missing": 0,
                "failed": 0,
                "strict_all_expected": scores,
                "historical_normalized_all_expected": scores,
            }},
        }
        for checkpoint in ("checkpoint-best", "checkpoint-last"):
            (root / checkpoint).mkdir(parents=True)
            (root / checkpoint / "model.pt").write_bytes(checkpoint.encode())
        (root / "best.json").write_text(json.dumps({"checkpoint": "checkpoint-best", "epoch": 2}))
        (root / "last.json").write_text(json.dumps({"checkpoint": "checkpoint-last", "epoch": 4}))
        (root / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
        for suffix, checkpoint in (("", "checkpoint-best"), ("_last", "checkpoint-last")):
            evaluation_dir = root / f"test_evaluation_greedy_one_pass{suffix}"
            prediction_dir = root / f"test_predictions_greedy_one_pass{suffix}"
            evaluation_dir.mkdir()
            prediction_dir.mkdir()
            (evaluation_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
            digest = hashlib.sha256((root / checkpoint / "model.pt").read_bytes()).hexdigest()
            (prediction_dir / "run_summary.json").write_text(json.dumps({"checkpoint_sha256": digest}))
        return root

    def test_dropout_comparison_and_accuracy_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = load_run("current_baseline", self.make_run(root / "baseline"))
            candidate = load_run("dropout01", self.make_run(root / "candidate", dropout=0.1))
            validate_comparison(baseline, candidate, "dropout01")
            table = format_evaluation_table(comparison_evaluation(baseline, candidate), "strict")
            self.assertIn("current_baseline", table)
            self.assertIn("dropout01", table)
            self.assertIn("75.00", table)
            self.assertIn("50.00", table)

    def test_rejects_different_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = load_run("current_baseline", self.make_run(root / "baseline"))
            candidate = load_run("dropout01", self.make_run(root / "candidate", dropout=0.1, split="other"))
            with self.assertRaisesRegex(ValueError, "different frozen split"):
                validate_comparison(baseline, candidate, "dropout01")

    def test_rejects_an_extra_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = load_run("current_baseline", self.make_run(root / "baseline"))
            candidate = load_run(
                "dropout01", self.make_run(root / "candidate", dropout=0.1, augmentation="mild_v1")
            )
            with self.assertRaisesRegex(ValueError, "expected dropout01 experiment"):
                validate_comparison(baseline, candidate, "dropout01")

    def test_accepts_explicit_combined_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = load_run("current_baseline", self.make_run(root / "baseline"))
            candidate = load_run(
                "combined",
                self.make_run(
                    root / "candidate", dropout=0.1, augmentation="mild_v1", weight_decay=0.0001
                ),
            )
            validate_comparison(baseline, candidate, "combined")

    def test_accepts_trocr_view_recipe_and_only_its_source_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = self.make_run(root / "baseline")
            candidate_path = self.make_run(root / "candidate", augmentation="trocr_views_v1")
            baseline_metadata = json.loads((baseline_path / "run.json").read_text())
            candidate_metadata = json.loads((candidate_path / "run.json").read_text())
            baseline_metadata["runtime"] = {"source_hashes": {
                "thesis/diffusion/images.py": "old-images",
                "thesis/diffusion/model.py": "same-model",
            }}
            candidate_metadata["runtime"] = {"source_hashes": {
                "thesis/diffusion/images.py": "new-images",
                "thesis/diffusion/model.py": "same-model",
            }}
            (baseline_path / "run.json").write_text(json.dumps(baseline_metadata))
            (candidate_path / "run.json").write_text(json.dumps(candidate_metadata))
            validate_comparison(load_run("current_baseline", baseline_path),
                                load_run("trocrviews", candidate_path), "trocrviews")

    def test_all_views_recipe_and_best_last_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = load_run("current_baseline_best", self.make_run(root / "baseline"))
            candidate_path = self.make_run(root / "candidate", augmentation="trocr_all_views_v1",
                                           early_stopping_patience=0)
            candidate = load_run("trocrallviews_best", candidate_path)
            validate_comparison(baseline, candidate, "trocrallviews")
            evaluation = comparison_evaluation(
                baseline,
                candidate,
                load_run("current_baseline_last", root / "baseline", "last"),
                load_run("trocrallviews_last", candidate_path, "last"),
            )
            self.assertEqual(list(evaluation["models"]), [
                "current_baseline_best", "trocrallviews_best", "current_baseline_last", "trocrallviews_last"
            ])


if __name__ == "__main__":
    unittest.main()
