"""The compact table must agree with the existing OCR comparison metrics."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesis.scripts import common
from thesis.diffusion.metrics import aggregate
from thesis.scripts.diffusion_stats_table import format_evaluation_table


class StatsTableTests(unittest.TestCase):
    def test_matches_existing_ocr_aggregation(self):
        pairs = [("بَت", "بت"), ("هم", ""), ("123", "123"), ("ت", "ت" * 30)]
        totals = common.empty_ocr_aggregate()
        for reference, prediction in pairs:
            common.add_ocr_score(totals, common.score_line_ocr(reference, prediction))
        expected = common.finalize_ocr_aggregate(totals)
        evaluation = {"partition": "test", "models": {"diffusion": {
            "historical_normalized_all_expected": aggregate(pairs, historical=True)}}}
        output = format_evaluation_table(evaluation)
        cells = output.splitlines()[1].split()
        self.assertEqual(cells[1:], ["4", *(f"{100 * expected[key]:.2f}" for key in
                                          ("letter_accuracy", "word_accuracy", "line_accuracy"))])
        self.assertIn("Coverage: all expected", output)

    def test_strict_and_empty_denominators(self):
        evaluation = {"partition": "train", "models": {"diffusion": {
            "strict_all_expected": aggregate([("", "")]),
            "historical_normalized_all_expected": aggregate([("ت", "")])}}}
        strict = format_evaluation_table(evaluation, "strict", "long_run_name")
        self.assertEqual(strict.splitlines()[1].split(),
                         ["diffusion/long_run_name", "1", "0.00", "0.00", "100.00"])
        self.assertIn("Partition: train | Policy: strict", strict)
        self.assertEqual(format_evaluation_table(evaluation).splitlines()[1].split()[-1], "0.00")


if __name__ == "__main__":
    unittest.main()
