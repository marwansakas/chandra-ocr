import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pretrained_ocr.clean_predictions import clean_text, main


class CleanPredictionsTests(unittest.TestCase):
    def test_clean_text_removes_marks_and_converts_digits(self):
        self.assertEqual(clean_text("كَتـاب ١٢٣ ۴۵۶"), "كتاب 123 456")

    def test_creates_clean_copy_and_summary(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions"
            model = source / "paddle-arabic"
            model.mkdir(parents=True)
            row = {"sample_id": "000001", "status": "ok", "reference": "كِتاب ١٢",
                   "raw_prediction": "كتاب 12", "prediction": "كتاب 12", "seconds": 0.1}
            (model / "predictions.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            (model / "run_summary.json").write_text(json.dumps({"model": "paddle-arabic", "model_name": "Paddle"}), encoding="utf-8")
            destination = root / "cleaned_predictions"
            self.assertEqual(main(["--input", str(source), "--output", str(destination)]), 0)
            cleaned = json.loads((destination / "paddle-arabic" / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(cleaned["reference"], "كتاب 12")
            self.assertEqual(cleaned["prediction"], "كتاب 12")
            with (destination / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                summary = list(csv.DictReader(handle))[0]
            self.assertEqual(summary["letter_accuracy_pct"], "100.00")
            self.assertEqual(summary["total_char_distance"], "0")

    def test_legacy_row_without_status_is_accepted(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions" / "legacy"
            source.mkdir(parents=True)
            row = {"sample_id": "000001", "reference": "١٢", "prediction": "12"}
            (source / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            destination = root / "cleaned_predictions"
            self.assertEqual(main(["--input", str(root / "predictions"), "--output", str(destination)]), 0)
            cleaned = json.loads((destination / "legacy" / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(cleaned["status"], "ok")

    def test_resume_reuses_a_completed_model(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions" / "legacy"
            source.mkdir(parents=True)
            row = {"sample_id": "000001", "reference": "١٢", "prediction": "12"}
            (source / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            destination = root / "cleaned_predictions"
            self.assertEqual(main(["--input", str(root / "predictions"), "--output", str(destination)]), 0)
            self.assertEqual(main(["--input", str(root / "predictions"), "--output", str(destination), "--resume"]), 0)
            with (destination / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)


if __name__ == "__main__":
    unittest.main()
