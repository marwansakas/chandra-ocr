from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "thesis" / "scripts"))
# Unit tests exercise path/status contracts and do not render PDFs.
sys.modules.setdefault("pypdfium2", types.ModuleType("pypdfium2"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load("builder", ROOT / "build_surya_line_ground_truth.py")
batch = load("batch", ROOT / "run_chandra_batch.py")
trainer = load("trainer", ROOT / "thesis" / "scripts" / "train_line_ocr_model.py")


def test_markdown_lookup_never_uses_another_line(tmp_path: Path):
    wrong = tmp_path / "line_001" / "line_001.md"
    wrong.parent.mkdir()
    wrong.write_text("wrong", encoding="utf-8")
    assert builder.find_chandra_markdown(tmp_path, Path("line_002.png")) is None


def test_markdown_lookup_accepts_one_matching_nested_file(tmp_path: Path):
    expected = tmp_path / "line_002" / "line_002.md"
    expected.parent.mkdir()
    expected.write_text("correct", encoding="utf-8")
    assert builder.find_chandra_markdown(tmp_path, Path("line_002.png")) == expected


def test_partial_chandra_output_is_not_complete(tmp_path: Path):
    stem = "page"
    (tmp_path / f"{stem}_metadata.json").write_text("{}", encoding="utf-8")
    assert not batch.output_already_exists(tmp_path, Path(f"{stem}.pdf"))


def test_nested_complete_chandra_output_is_complete(tmp_path: Path):
    nested = tmp_path / "page"
    nested.mkdir()
    (nested / "page.md").write_text("text", encoding="utf-8")
    (nested / "page_metadata.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    assert batch.output_already_exists(tmp_path, Path("page.pdf"))


def test_student_group_removes_rank_and_page_but_keeps_notebook():
    first = "0001_0001_Kfar__Fourth Grade__Notebook_5_Female_Fourth_Grade_Page_030"
    second = "0044_0044_Kfar__Fourth Grade__Notebook_5_Female_Fourth_Grade_Page_032"
    other = "0045_0045_Kfar__Fourth Grade__Notebook_8_Male_Fourth_Grade_Page_072"
    assert trainer.student_group(first) == trainer.student_group(second)
    assert trainer.student_group(first) != trainer.student_group(other)
