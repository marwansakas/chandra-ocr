import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pretrained_ocr.cli import main
from pretrained_ocr.dataset import check_output_path, read_dataset
from pretrained_ocr.metrics import aggregate, edit_distance, normalize, score
from pretrained_ocr.models.hf_common import Prediction


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "ground_truth_lines"
        self.output = self.root / "results/got-ocr2"
        for name, reference in [("000001", "كتاب"), ("000002", "هذا كتاب")]:
            folder = self.dataset / name
            folder.mkdir(parents=True)
            (folder / "image.png").write_bytes(b"mock image for a fake backend")
            (folder / "ground_truth.txt").write_text(reference, encoding="utf-8")
            (folder / "metadata.json").write_text('{"prediction":"must not reach backend"}', encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def call(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(["run", "--model", "got-ocr2", "--dataset", str(self.dataset),
                         "--output", str(self.output), *extra])

    def test_edit_rates_and_arabic_policy(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(score("كتاب", "كتا")['cer'], 0.25)
        self.assertEqual(score("هذا كتاب", "هذا")['wer'], 0.5)
        self.assertEqual(score("ا", "ا ا ا")['wer'], 2)
        self.assertEqual(normalize("  هٰذا\n كِتاب  "), "هٰذا كِتاب")
        self.assertEqual(normalize("كِتـاب", "arabic-light"), "كتاب")
        self.assertNotEqual(normalize("أ", "arabic-light"), normalize("ا", "arabic-light"))
        self.assertNotEqual(normalize("ة", "arabic-light"), normalize("ه", "arabic-light"))
        totals = aggregate([score("a", ""), score("abcdefghi", "abcdefghi")])
        self.assertEqual(totals['cer'], 0.1)  # Corpus weighting, not mean(1, 0).

    def test_missing_and_empty_labels_reported(self):
        for name, text in [("empty", "  "), ("ignored", "IGNORE"), ("missing-image", "نص")]:
            folder = self.dataset / name
            folder.mkdir()
            (folder / "ground_truth.txt").write_text(text, encoding="utf-8")
            if name != "missing-image":
                (folder / "image.png").write_bytes(b"image")
        samples, skipped = read_dataset(self.dataset)
        self.assertEqual([x.sample_id for x in samples], ["000001", "000002"])
        self.assertEqual({x['reason'] for x in skipped}, {"empty_ground_truth", "ignore_marker", "missing_image"})

    def test_source_overlap_rejected(self):
        for output in [self.dataset, self.dataset / "predictions", self.root]:
            with self.assertRaises(ValueError):
                check_output_path(self.dataset, output)

    def test_dry_run_does_not_load_model_or_write_outputs(self):
        with patch("pretrained_ocr.cli.load_backend") as factory:
            self.assertEqual(self.call("--dry-run"), 0)
            factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_full_run_resume_and_no_reference_leak(self):
        before = {p: p.read_bytes() for p in self.dataset.rglob("*") if p.is_file()}
        seen = []
        class FakeBackend:
            info = {"device": "fake"}
            def predict(self, image):
                seen.append(image)
                return Prediction("كتاب" if image.parent.name == "000001" else "هذا كتاب")
        with patch("pretrained_ocr.cli.load_backend", return_value=FakeBackend()):
            self.assertEqual(self.call(), 0)
        self.assertTrue(all(isinstance(path, Path) and path.name == "image.png" for path in seen))
        summary = json.loads((self.output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary['metrics']['cer'], 0)
        self.assertEqual(summary['metrics']['wer'], 0)
        with patch("pretrained_ocr.cli.load_backend") as factory:
            self.assertEqual(self.call("--resume"), 0)
            factory.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(self.call(), 2)  # Never silently replace a run.
        self.assertEqual(self.call("--resume", "--normalization", "raw"), 2)

    def test_failed_samples_penalized_and_retried(self):
        class FailingBackend:
            info = {"device": "fake"}
            def predict(self, image):
                if image.parent.name == "000002":
                    raise RuntimeError("failed OCR")
                return Prediction("كتاب", hit_token_limit=True)
        with patch("pretrained_ocr.cli.load_backend", return_value=FailingBackend()):
            self.assertEqual(self.call(), 1)
        summary = json.loads((self.output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary['failed_samples'], 1)
        self.assertEqual(summary['metrics']['cer'], 8 / 12)
        self.assertEqual(summary['successful_only_metrics']['cer'], 0)
        self.assertEqual(summary['token_limit_samples'], 1)
        backend = argparse.Namespace(info={"device": "fake"}, predict=lambda image: Prediction("هذا كتاب"))
        with patch("pretrained_ocr.cli.load_backend", return_value=backend):
            self.assertEqual(self.call("--resume"), 0)

    def test_changed_image_invalidates_resume(self):
        backend = argparse.Namespace(info={}, predict=lambda image: Prediction("كتاب"))
        with patch("pretrained_ocr.cli.load_backend", return_value=backend):
            self.assertEqual(self.call(), 0)
        (self.dataset / "000001/image.png").write_bytes(b"changed image")
        self.assertEqual(self.call("--resume"), 2)

    def test_comparison_rejects_unmatched_samples(self):
        backend = argparse.Namespace(info={}, predict=lambda image: Prediction("كتاب"))
        with patch("pretrained_ocr.cli.load_backend", return_value=backend):
            self.assertEqual(self.call(), 0)
        summary = json.loads((self.output / "run_summary.json").read_text(encoding="utf-8"))
        summary['dataset_fingerprint'] = 'different'
        other = self.output.parent / "other"
        other.mkdir()
        (other / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["compare", "--results", str(self.output.parent)]), 2)


if __name__ == "__main__":
    unittest.main()
