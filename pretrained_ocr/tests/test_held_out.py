import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pretrained_ocr.cli import main
from pretrained_ocr.dataset import read_dataset, read_test_manifest
from pretrained_ocr.models.hf_common import Prediction
from pretrained_ocr.registry import MODELS


class HeldOutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "ground_truth_lines"
        self.manifest = self.root / "test.jsonl"
        self.results = self.root / "results"
        for sample_id in ("000001", "000002", "000003"):
            folder = self.dataset / sample_id
            folder.mkdir(parents=True)
            (folder / "image.png").write_bytes(b"image fixture")
            (folder / "ground_truth.txt").write_text("كتاب", encoding="utf-8")
        self.write_manifest(["000003", "000001"])

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, ids):
        # Match the project's test.jsonl schema; paths can be from another machine.
        rows = [{"sample_id": key, "image_path": f"/old/project/{key}/image.png", "text": "snapshot"}
                for key in ids]
        self.manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def invoke(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(list(argv))

    def run_model(self, model="got-ocr2", *extra):
        return self.invoke("run", "--model", model, "--dataset", str(self.dataset),
                           "--test-manifest", str(self.manifest), "--resume", *extra)

    def test_only_manifest_samples_are_read(self):
        (self.dataset / "000002/ground_truth.txt").write_bytes(b"\xff invalid unselected label")
        split = read_test_manifest(self.manifest)
        samples, skipped = read_dataset(self.dataset, split["sample_ids"])
        self.assertEqual([sample.sample_id for sample in samples], ["000001", "000003"])
        self.assertEqual([sample.reference for sample in samples], ["كتاب", "كتاب"])
        self.assertEqual(skipped, [])

    def test_all_models_use_held_out_output_and_comparable_scores(self):
        seen = []
        backend = argparse.Namespace(info={"device": "fake"}, predict=lambda path: (seen.append(path) or Prediction("كتاب")))
        before = {path: path.read_bytes() for path in self.dataset.rglob("*") if path.is_file()}
        with patch("pretrained_ocr.cli.DEFAULT_OUTPUT", self.results), patch("pretrained_ocr.cli.load_backend", return_value=backend):
            for model in MODELS:
                self.assertEqual(self.run_model(model), 0)
                output = self.results / "held_out" / model
                summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["selected_samples"], 2)
                self.assertEqual(summary["metrics"]["cer"], 0)
                self.assertEqual(summary["config"]["test_manifest"]["path"], str(self.manifest))
                self.assertFalse((output / "samples/000002").exists())
        self.assertEqual({path.parent.name for path in seen}, {"000001", "000003"})
        self.assertEqual(len(seen), 10)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(self.invoke("compare", "--results", str(self.results / "held_out")), 0)

    def test_limit_applies_after_test_selection(self):
        seen = []
        backend = argparse.Namespace(info={}, predict=lambda path: (seen.append(path) or Prediction("كتاب")))
        with patch("pretrained_ocr.cli.DEFAULT_OUTPUT", self.results), patch("pretrained_ocr.cli.load_backend", return_value=backend):
            self.assertEqual(self.run_model("got-ocr2", "--start-line", "2", "--limit", "1"), 0)
        self.assertEqual([path.parent.name for path in seen], ["000003"])

    def test_invalid_manifests_fail_before_loading_models(self):
        cases = ["", "not json", "[]", "{}", '{"sample_id": 42}',
                 '{"sample_id":"../outside"}', '{"sample_id":"/absolute"}',
                 '{"sample_id":"C:/outside"}', '{"sample_id":"."}',
                 '{"sample_id":"missing"}',
                 '{"sample_id":"000001"}\n{"sample_id":"000001"}']
        with patch("pretrained_ocr.cli.DEFAULT_OUTPUT", self.results), patch("pretrained_ocr.cli.load_backend") as factory:
            for contents in cases:
                with self.subTest(contents=contents):
                    self.manifest.write_text(contents, encoding="utf-8")
                    self.assertEqual(self.run_model(), 2)
            self.manifest.unlink()
            self.assertEqual(self.run_model(), 2)
            factory.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_missing_held_out_label_is_not_silently_skipped(self):
        (self.dataset / "000001/ground_truth.txt").unlink()
        with self.assertRaisesRegex(ValueError, "held-out samples are missing or unusable"):
            read_dataset(self.dataset, ["000001", "000003"])

    def test_changed_manifest_cannot_resume_into_same_run(self):
        backend = argparse.Namespace(info={}, predict=lambda path: Prediction("كتاب"))
        with patch("pretrained_ocr.cli.DEFAULT_OUTPUT", self.results), patch("pretrained_ocr.cli.load_backend", return_value=backend):
            self.assertEqual(self.run_model(), 0)
            self.write_manifest(["000001"])
            self.assertEqual(self.run_model(), 2)


if __name__ == "__main__":
    unittest.main()
