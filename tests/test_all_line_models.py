import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thesis/scripts"))
from evaluate_all_line_models import evaluate


def test_same_diacritic_free_scores_for_both_layouts(tmp_path):
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(json.dumps({"sample_id": "line1", "text": "كَتَبَ"}) + "\n", encoding="utf-8")
    roots = {"local": tmp_path / "local", "pretrained": tmp_path / "pretrained" / "samples"}
    for root in roots.values():
        (root / "line1").mkdir(parents=True)
        (root / "line1" / "prediction.txt").write_text("كتب", encoding="utf-8")
    report = evaluate(manifest, roots, tmp_path / "reports")
    for row in report["models"]:
        assert row["cer"] == row["wer"] == 0
        assert row["exact_match_rate"] == row["smith_waterman_similarity"] == row["text_alignment_similarity"] == 1


def test_missing_predictions_reject_partial_comparison(tmp_path):
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(json.dumps({"sample_id": "line1", "text": "كتب"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing 1/1 predictions"):
        evaluate(manifest, {"missing_model": tmp_path / "missing"}, tmp_path / "reports")
    assert not (tmp_path / "reports").exists()
