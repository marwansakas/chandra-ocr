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
from pretrained_ocr.metrics import aggregate, edit_distance, edit_operations, normalize, score
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
        self.assertEqual(normalize("  هٰذا\n كِتاب  "), "هذا كتاب")
        self.assertEqual(normalize("كِتـاب", "arabic-light"), "كتاب")
        self.assertEqual(normalize("أ", "arabic-light"), normalize("ا", "arabic-light"))
        self.assertEqual(normalize("Crème naïve", "arabic-light"), "Creme naive")
        self.assertNotEqual(normalize("ة", "arabic-light"), normalize("ه", "arabic-light"))
        totals = aggregate([score("a", ""), score("abcdefghi", "abcdefghi")])
        self.assertEqual(totals['cer'], 0.1)  # Corpus weighting, not mean(1, 0).

    def test_alignment_metrics_distinguish_error_types(self):
        self.assertEqual(edit_operations("كتاب", "كتا"), (0, 1, 0))
        self.assertEqual(edit_operations("كتاب", "كتابب"), (0, 0, 1))
        self.assertEqual(edit_operations("كتاب", "كتتب"), (1, 0, 0))
        metrics = score("abc", "axcd")
        self.assertEqual(metrics['char_substitutions'], 1)
        self.assertEqual(metrics['char_insertions'], 1)
        self.assertEqual(metrics['char_correct'], 2)
        self.assertEqual(metrics['char_precision'], 0.5)
        self.assertEqual(metrics['char_recall'], 2 / 3)
        self.assertAlmostEqual(metrics['char_f1'], 4 / 7)

    def test_alignment_metrics_ignore_diacritics(self):
        exact = score("كَتَبَ", "كتب")
        self.assertEqual(exact["smith_waterman_similarity"], 1.0)
        self.assertEqual(exact["text_alignment_similarity"], 1.0)
        partial = score("هذا كتاب", "كتاب")
        self.assertGreater(partial["smith_waterman_similarity"], 0)
        self.assertLess(partial["smith_waterman_similarity"], 1)
        self.assertGreaterEqual(partial["text_alignment_similarity"], 0)
        self.assertLess(partial["text_alignment_similarity"], 1)

    def test_diacritics_are_ignored_on_both_sides_by_default(self):
        pairs = [("هٰذَا كِتَابٌ", "هذا كتاب"), ("كَتَبَ", "كُتِبَ")]
        # Tanwin, short vowels, shadda, sukun, dagger alif and a Quranic mark.
        pairs.extend(("ب" + mark, "ب") for mark in "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u06d6")
        for reference, prediction in pairs:
            for left, right in [(reference, prediction), (prediction, reference)]:
                with self.subTest(reference=left, prediction=right):
                    metrics = score(left, right)
                    self.assertEqual(metrics['cer'], 0)
                    self.assertEqual(metrics['wer'], 0)
                    self.assertTrue(metrics['exact_match'])

    def test_base_character_errors_and_counts_are_preserved(self):
        metrics = score("كِتَابٌ", "كِتَا")
        self.assertEqual(metrics['reference_chars'], 4)
        self.assertEqual(metrics['char_edits'], 1)
        self.assertEqual(metrics['cer'], 0.25)
        self.assertEqual(metrics['wer'], 1)
        self.assertFalse(metrics['exact_match'])
        for reference, prediction in [("ب", "ت"), ("ة", "ه"), ("ى", "ي"), ("ب!", "ب")]:
            with self.subTest(reference=reference, prediction=prediction):
                self.assertFalse(score(reference, prediction)['exact_match'])
        for reference, prediction in [("أ", "ا"), ("é", "e"), ("Crème", "Creme")]:
            with self.subTest(reference=reference, prediction=prediction):
                self.assertTrue(score(reference, prediction)['exact_match'])
        self.assertTrue(score("ا\u0654", "أ")['exact_match'])
        totals = aggregate([score("كِتَابٌ", "كِتَا"), score("هٰذَا", "هذا")])
        self.assertEqual(totals['reference_chars'], 7)
        self.assertEqual(totals['cer'], 1 / 7)

    def test_explicit_standard_and_raw_keep_diacritics(self):
        text = "  هٰذا\n كِتاب  "
        self.assertEqual(normalize(text, "standard"), "هٰذا كِتاب")
        self.assertEqual(normalize(text, "raw"), text)
        for mode in ("standard", "raw"):
            with self.subTest(mode=mode):
                metrics = score("كِتاب", "كتاب", mode)
                self.assertEqual(metrics['cer'], 1 / 5)
                self.assertEqual(metrics['wer'], 1)
                self.assertFalse(metrics['exact_match'])
        self.assertFalse(score("é", "e", "standard")['exact_match'])

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
        references = {"000001": "كِتَابٌ", "000002": "هٰذَا كِتَابٌ"}
        for name, reference in references.items():
            (self.dataset / name / "ground_truth.txt").write_text(reference, encoding="utf-8")
        before = {p: p.read_bytes() for p in self.dataset.rglob("*") if p.is_file()}
        seen = []
        class FakeBackend:
            info = {"device": "fake"}
            def predict(self, image):
                seen.append(image)
                return Prediction("كِتَابٌ" if image.parent.name == "000001" else "هذا كتاب")
        with patch("pretrained_ocr.cli.load_backend", return_value=FakeBackend()):
            self.assertEqual(self.call(), 0)
        self.assertTrue(all(isinstance(path, Path) and path.name == "image.png" for path in seen))
        summary = json.loads((self.output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary['metrics']['cer'], 0)
        self.assertEqual(summary['metrics']['wer'], 0)
        self.assertEqual(summary['metrics']['exact_match_rate'], 1)
        self.assertEqual(summary['metrics']['reference_chars'], 12)
        self.assertEqual(summary['metrics']['char_f1'], 1)
        self.assertEqual(summary['metrics']['word_f1'], 1)
        self.assertEqual(summary['performance']['timed_samples'], 2)
        self.assertEqual(summary['normalization'], 'arabic-light')
        self.assertEqual(summary['config']['normalization'], 'arabic-light')
        records = [json.loads(line) for line in (self.output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in records:
            self.assertEqual(row['reference'], references[row['sample_id']])
            self.assertEqual(row['prediction'], "كتاب" if row['sample_id'] == "000001" else "هذا كتاب")
            self.assertEqual(row['raw_prediction'], "كِتَابٌ" if row['sample_id'] == "000001" else "هذا كتاب")
            self.assertTrue(row['exact_match'])
        with patch("pretrained_ocr.cli.load_backend") as factory:
            self.assertEqual(self.call("--resume"), 0)
            factory.assert_not_called()
        resumed = json.loads((self.output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(resumed['metrics'], summary['metrics'])
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
