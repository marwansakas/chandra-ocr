"""
Shared helpers for the thesis pipeline.

Design goals:
- Standard-library only, so it runs with any Python 3.8+ on the cluster
  (login node, base conda, or the chandra env) without installing anything.
- One source of truth for: config loading, filename parsing, and the ID scheme.

The ID scheme (IMPORTANT for the thesis write-up):
- A "notebook" in the filename is assumed to be ONE student's notebook.
  Therefore student_id == notebook_id. This is an assumption to confirm with
  the data collectors; it is documented here and in thesis/PLAN.md.
- PDF stems are NOT globally unique (the same Notebook_x_..._Page_y name can
  appear in both schools), so every page is keyed by place + grade + stem.
"""

from __future__ import annotations

import json
import re
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent          # thesis/scripts
THESIS_DIR = THIS_DIR.parent                          # thesis
PROJECT_ROOT = THESIS_DIR.parent                      # repo root (chandra-ocr)


def load_config() -> dict:
    cfg_path = THESIS_DIR / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg


def resolve(path_str: str) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    # config paths are written relative to project root
    return (PROJECT_ROOT / p).resolve()


# --------------------------------------------------------------------------
# Filename / path parsing
# --------------------------------------------------------------------------

# Matches, in order of richness:
#   Notebook_1_Female_Fourth_Grade_Page_001
#   Notebook_1_Male_Page_001           (no grade token)
#   Notebook_2_Page_001                (no gender, no grade)
FILENAME_RE = re.compile(
    r"^Notebook_(?P<notebook>\d+)"
    r"(?:_(?P<gender>Male|Female))?"
    r"(?:_(?P<grade_word>First|Second|Third|Fourth)_Grade)?"
    r"_Page_(?P<page>\d+)$"
)

GRADE_WORD_TO_INT = {"First": 1, "Second": 2, "Third": 3, "Fourth": 4}

# Folder grade names -> int (handles the "Fourth grade" lowercase variant).
GRADE_FOLDER_TO_INT = {
    "first grade": 1,
    "second grade": 2,
    "third grade": 3,
    "fourth grade": 4,
}


def slug(text: str) -> str:
    """Filesystem/ID-safe slug: lowercase, spaces/sep -> underscore."""
    text = text.strip().lower()
    text = re.sub(r"[\s/]+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text


@dataclass
class PageRecord:
    page_id: str            # unique, human-readable: place_slug/grade_slug/stem
    stem: str               # filename without extension
    image_path: str         # path to the source PDF (relative to project root)
    place: str              # school / location (folder level 1)
    grade_folder: str       # raw grade folder name (level 2)
    grade: Optional[int]    # 1..4 or None (Unknown)
    grade_source: str       # "filename" | "folder" | "unknown"
    gender: Optional[str]   # Male | Female | None
    notebook_number: Optional[int]
    notebook_id: str        # place__grade__nbN__gender  (== student_id)
    student_id: str         # alias of notebook_id (1 notebook = 1 student assumption)
    page_number: Optional[int]
    filename_parsed: bool   # did the regex match?
    ocr_output_path: Optional[str] = None   # filled by collect_ocr.py
    ocr_text_present: bool = False
    ground_truth_path: Optional[str] = None # filled if a GT file exists
    ground_truth_present: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def parse_pdf(pdf_path: Path, raw_root: Path) -> PageRecord:
    """Build a PageRecord from a PDF path inside split_papers."""
    rel = pdf_path.relative_to(raw_root)
    parts = rel.parts  # e.g. ("Hora", "First Grade", "Notebook_10_..._Page_001.pdf")

    place = parts[0] if len(parts) >= 1 else "Unknown"
    grade_folder = parts[1] if len(parts) >= 3 else "Unknown"

    stem = pdf_path.stem
    m = FILENAME_RE.match(stem)

    gender = None
    notebook_number = None
    page_number = None
    grade = None
    grade_source = "unknown"

    if m:
        gender = m.group("gender")
        notebook_number = int(m.group("notebook"))
        page_number = int(m.group("page"))
        gw = m.group("grade_word")
        if gw:
            grade = GRADE_WORD_TO_INT[gw]
            grade_source = "filename"

    # Fall back to the folder name for grade if filename had none.
    if grade is None:
        folder_key = grade_folder.strip().lower()
        if folder_key in GRADE_FOLDER_TO_INT:
            grade = GRADE_FOLDER_TO_INT[folder_key]
            grade_source = "folder"

    grade_slug = f"grade_{grade}" if grade is not None else "grade_unknown"
    place_slug = slug(place)

    page_id = f"{place_slug}/{grade_slug}/{stem}"

    nb_part = f"nb{notebook_number}" if notebook_number is not None else "nbNA"
    gender_part = (gender or "NA").lower()
    notebook_id = f"{place_slug}__{grade_slug}__{nb_part}__{gender_part}"

    return PageRecord(
        page_id=page_id,
        stem=stem,
        image_path=str(pdf_path.relative_to(PROJECT_ROOT)),
        place=place,
        grade_folder=grade_folder,
        grade=grade,
        grade_source=grade_source,
        gender=gender,
        notebook_number=notebook_number,
        notebook_id=notebook_id,
        student_id=notebook_id,
        page_number=page_number,
        filename_parsed=bool(m),
    )


def iter_pdfs(raw_root: Path):
    yield from sorted(raw_root.rglob("*.pdf"))


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv(path: Path, records, fields=None) -> None:
    """Write dictionaries as CSV, preserving field order when supplied."""
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None and rows:
        fields = list(rows[0].keys())
    if fields is None:
        fields = []
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_utf8_text(path: Path, text: str) -> None:
    """Write a plain-text OCR prediction using the project's UTF-8 contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


ARABIC_DIACRITICS_RE = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e8\u06ea-\u06ed]"
)
ARABIC_EVALUATION_TRANSLATION = str.maketrans(
    {
        "آ": "ا",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ٲ": "ا",
        "ٳ": "ا",
        "ٵ": "ا",
        "ـ": "",
    }
)


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing letters or punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text_for_evaluation(text: str) -> str:
    """Normalize text for OCR metrics without changing distinct hamza letters."""
    text = ARABIC_DIACRITICS_RE.sub("", text or "")
    text = text.translate(ARABIC_EVALUATION_TRANSLATION)
    return normalize_text(text)


def tokenize_words(text: str):
    """Unicode-friendly word tokenizer for Arabic/Hebrew/Latin scripts."""
    return re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)


def edit_distance(a, b) -> int:
    """Levenshtein distance for strings or token lists."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def ratio_error(reference, hypothesis) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def safe_accuracy(error_rate: float) -> float:
    """Convert an error rate to an accuracy, clamped at zero."""
    return max(0.0, 1.0 - error_rate)


def score_line_ocr(reference: str, prediction: str) -> dict[str, float | int]:
    """Score one OCR prediction using the project-wide line OCR metrics."""
    reference = normalize_text_for_evaluation(reference)
    prediction = normalize_text_for_evaluation(prediction)
    ref_words = tokenize_words(reference)
    pred_words = tokenize_words(prediction)
    char_distance = edit_distance(reference, prediction)
    word_distance = edit_distance(ref_words, pred_words)
    char_error = char_distance / len(reference) if reference else (0.0 if not prediction else 1.0)
    word_error = word_distance / len(ref_words) if ref_words else (0.0 if not pred_words else 1.0)
    return {
        "reference_chars": len(reference),
        "prediction_chars": len(prediction),
        "reference_words": len(ref_words),
        "prediction_words": len(pred_words),
        "char_distance": char_distance,
        "word_distance": word_distance,
        "letter_accuracy": safe_accuracy(char_error),
        "word_accuracy": safe_accuracy(word_error),
        "line_accuracy": 1.0 if reference == prediction else 0.0,
    }


def empty_ocr_aggregate() -> dict[str, int]:
    return {
        "evaluated_lines": 0,
        "missing_predictions": 0,
        "exact_lines": 0,
        "reference_chars": 0,
        "char_distance": 0,
        "reference_words": 0,
        "word_distance": 0,
    }


def add_ocr_score(aggregate: dict[str, int], scores: dict[str, float | int], missing: bool = False) -> None:
    aggregate["evaluated_lines"] += 1
    aggregate["missing_predictions"] += int(missing)
    aggregate["exact_lines"] += int(scores["line_accuracy"])
    aggregate["reference_chars"] += int(scores["reference_chars"])
    aggregate["char_distance"] += int(scores["char_distance"])
    aggregate["reference_words"] += int(scores["reference_words"])
    aggregate["word_distance"] += int(scores["word_distance"])


def finalize_ocr_aggregate(aggregate: dict[str, int]) -> dict[str, float | int]:
    evaluated = aggregate["evaluated_lines"]
    line_accuracy = aggregate["exact_lines"] / evaluated if evaluated else 0.0
    letter_accuracy = (
        safe_accuracy(aggregate["char_distance"] / aggregate["reference_chars"])
        if aggregate["reference_chars"]
        else 0.0
    )
    word_accuracy = (
        safe_accuracy(aggregate["word_distance"] / aggregate["reference_words"])
        if aggregate["reference_words"]
        else 0.0
    )
    return {
        **aggregate,
        "letter_accuracy": letter_accuracy,
        "word_accuracy": word_accuracy,
        "line_accuracy": line_accuracy,
    }


def ocr_markdown_output_path(ocr_root: Path, source_path: Path | str) -> Optional[Path]:
    """Build the Chandra-compatible Markdown path for a source page path.

    ``source_path`` may be relative to ``split_papers`` (the usual case) or
    include that directory as an additional leading component. Only the final
    ``<place>/<grade_folder>/<filename>`` components participate in the OCR
    output layout.
    """
    path = Path(source_path)
    if len(path.parts) < 3:
        return None
    place, grade_folder, filename = path.parts[-3:]
    stem = Path(filename).stem
    return ocr_root / place / grade_folder / stem / stem / f"{stem}.md"


def ocr_markdown_path(existing_ocr_root: Path, record: dict) -> Optional[Path]:
    """
    Return the expected Chandra markdown path for a page record.

    Chandra output mirrors:
      output/<place>/<grade_folder>/<stem>/<stem>/<stem>.md

    The full place + grade folder + stem path is used because stems are not
    globally unique in this dataset.
    """
    candidate = ocr_markdown_output_path(existing_ocr_root, record["image_path"])
    if candidate is None:
        return None
    return candidate if candidate.exists() else None


def ground_truth_path(ground_truth_root: Path, record: dict) -> Path:
    """Ground-truth text path matching page_id, e.g. labels/.../hora/grade_1/x.txt."""
    return ground_truth_root / f"{record['page_id']}.txt"
