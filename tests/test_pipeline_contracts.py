from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "thesis" / "scripts"))
sys.path.insert(0, str(ROOT / "pipelines"))


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
ctc_trainer = load("ctc_trainer", ROOT / "thesis" / "scripts" / "train_ctc_line_ocr_model.py")
line_common = load("line_common", ROOT / "pipelines" / "line_runner_common.py")


def test_find_pdfs_is_recursive_deterministic_and_limited(tmp_path: Path):
    for relative in ("z/three.pdf", "a/two.PDF", "a/one.pdf", "ignore.txt"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    found = builder.find_pdfs(tmp_path)
    selected = builder.select_pdf_batch(found, start_pdf=1, limit=2)
    assert [path.relative_to(tmp_path).as_posix() for _, path in selected] == ["a/one.pdf", "a/two.PDF"]


def test_pdf_batch_preserves_original_positions():
    pdfs = [Path(f"page_{index}.pdf") for index in range(1, 251)]
    selected = builder.select_pdf_batch(pdfs, start_pdf=101, limit=100)
    assert selected[0] == (101, Path("page_101.pdf"))
    assert selected[-1] == (200, Path("page_200.pdf"))


def test_model_runners_only_discover_surya_images(tmp_path: Path):
    for page, line in (("page_b", "line_002"), ("page_a", "line_001")):
        folder = tmp_path / page / line
        folder.mkdir(parents=True)
        (folder / "image.png").touch()
        (folder / "ground_truth.txt").write_text("must not be an input", encoding="utf-8")
    (tmp_path / "unrelated.png").touch()
    images = line_common.find_line_images(tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path in images] == [
        "page_a/line_001/image.png",
        "page_b/line_002/image.png",
    ]


def test_page_complete_requires_image_and_ground_truth(tmp_path: Path):
    original_root = builder.PROJECT_ROOT
    builder.PROJECT_ROOT = tmp_path
    try:
        page = tmp_path / "dataset" / "page"
        line = page / "line_001"
        line.mkdir(parents=True)
        image = line / "image.png"
        label = line / "ground_truth.txt"
        image.touch()
        row = {
            "image_path": str(image.relative_to(tmp_path)),
            "ground_truth_path": str(label.relative_to(tmp_path)),
        }
        (page / "lines.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert not builder.page_is_complete(page)
        label.write_text("", encoding="utf-8")
        (page / "metadata.json").write_text("{}\n", encoding="utf-8")
        assert builder.page_is_complete(page)
    finally:
        builder.PROJECT_ROOT = original_root


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


def test_ctc_vocab_reserves_blank_at_zero():
    records = [
        ctc_trainer.LineRecord("a", "0001_0001_School__Notebook_1_Page_001", 1, 1, "x.png", "ba", "ocr_complete"),
        ctc_trainer.LineRecord("b", "0002_0002_School__Notebook_1_Page_002", 2, 1, "y.png", "ab", "ocr_complete"),
    ]
    vocab = ctc_trainer.build_vocab(records)
    assert vocab[0] == ctc_trainer.BLANK_TOKEN
    assert set(vocab[1:]) == {"a", "b"}


def test_ctc_greedy_decode_collapses_repeats_and_blanks():
    idx_to_char = {0: "<blank>", 1: "a", 2: "b"}
    assert ctc_trainer.ctc_greedy_decode([0, 1, 1, 0, 2, 2, 0, 1], idx_to_char) == "aba"


def test_ctc_min_time_steps_accounts_for_repeated_labels():
    assert ctc_trainer.ctc_min_time_steps([1, 2, 2, 1]) == 5
