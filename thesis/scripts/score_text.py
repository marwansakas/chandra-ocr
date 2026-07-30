"""
Score spelling/grammar-related text features.

Preference order for scoring text:
  1. Manual ground truth, if present.
  2. Chandra OCR text, marked as OCR-derived.

Optional spelling support:
  thesis/labels/arabic_wordlist.txt

Each line in the wordlist should contain one valid word. If no wordlist exists,
dictionary-based spelling metrics are left blank.

Writes:
  thesis/reports/text_scores_pages.csv
  thesis/reports/text_scores_students.csv

Run:
  python thesis/scripts/score_text.py
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re

import common


SENTENCE_END_RE = re.compile(r"[.!؟?]+")


def load_wordlist(path: Path):
    if not path.exists():
        return None
    words = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        word = line.strip()
        if word:
            words.add(word)
    return words


def text_for_record(record: dict, gt_root: Path):
    gt_path = common.ground_truth_path(gt_root, record)
    if gt_path.exists() and common.read_text(gt_path).strip():
        return common.read_text(gt_path), "ground_truth", str(gt_path.relative_to(common.PROJECT_ROOT))
    if record.get("ocr_output_path"):
        path = common.resolve(record["ocr_output_path"])
        return common.read_text(path), "ocr", record["ocr_output_path"]
    return "", "missing", ""


def score_record(record: dict, text: str, text_source: str, source_path: str, wordlist):
    norm = common.normalize_text(text)
    words = common.tokenize_words(norm)
    word_count = len(words)
    unique_words = len(set(words))
    sentences = [s for s in SENTENCE_END_RE.split(norm) if s.strip()]
    sentence_count = len(sentences)
    punctuation_count = len(re.findall(r"[،؛,.!؟?:]", norm))
    repeated_adjacent = sum(1 for a, b in zip(words, words[1:]) if a == b)

    oov_count = ""
    oov_rate = ""
    spelling_score = ""
    if wordlist is not None and word_count:
        oov = [w for w in words if w not in wordlist]
        oov_count = len(oov)
        oov_rate_value = len(oov) / word_count
        oov_rate = f"{oov_rate_value:.4f}"
        spelling_score = f"{max(0.0, 1.0 - oov_rate_value):.4f}"

    # Language-agnostic proxy: not a final grammar score, but useful as a stable feature.
    punctuation_per_100_words = (100 * punctuation_count / word_count) if word_count else 0.0
    sentence_length_avg = (word_count / sentence_count) if sentence_count else 0.0
    repetition_rate = (repeated_adjacent / max(1, word_count - 1)) if word_count > 1 else 0.0
    structure_score = 0.0
    if word_count:
        punctuation_component = min(1.0, punctuation_per_100_words / 8.0)
        sentence_component = 1.0 if 3 <= sentence_length_avg <= 25 else 0.5
        repetition_component = max(0.0, 1.0 - repetition_rate)
        structure_score = (punctuation_component + sentence_component + repetition_component) / 3

    return {
        "page_id": record["page_id"],
        "student_id": record["student_id"],
        "place": record["place"],
        "grade": record.get("grade") or "unknown",
        "gender": record.get("gender") or "unknown",
        "text_source": text_source,
        "source_path": source_path,
        "char_count": len(norm),
        "word_count": word_count,
        "unique_word_count": unique_words,
        "type_token_ratio": f"{(unique_words / word_count):.4f}" if word_count else "",
        "sentence_count": sentence_count,
        "avg_sentence_words": f"{sentence_length_avg:.2f}",
        "punctuation_count": punctuation_count,
        "punctuation_per_100_words": f"{punctuation_per_100_words:.2f}",
        "adjacent_repetition_count": repeated_adjacent,
        "adjacent_repetition_rate": f"{repetition_rate:.4f}",
        "dictionary_oov_count": oov_count,
        "dictionary_oov_rate": oov_rate,
        "spelling_score": spelling_score,
        "structure_score": f"{structure_score:.4f}",
    }


def average_float(rows, field):
    values = [float(r[field]) for r in rows if r.get(field) not in ("", None)]
    return sum(values) / len(values) if values else None


def main() -> None:
    cfg = common.load_config()
    metadata_dir = common.resolve(cfg["metadata_dir"])
    labels_dir = common.resolve(cfg["labels_dir"])
    gt_root = common.resolve(cfg["ground_truth_dir"])
    reports_dir = common.resolve(cfg["reports_dir"])

    wordlist = load_wordlist(labels_dir / "arabic_wordlist.txt")
    records = list(common.read_jsonl(metadata_dir / "pages.jsonl"))

    page_rows = []
    for record in records:
        text, text_source, source_path = text_for_record(record, gt_root)
        if not text.strip():
            continue
        page_rows.append(score_record(record, text, text_source, source_path, wordlist))

    common.write_csv(reports_dir / "text_scores_pages.csv", page_rows)

    by_student = defaultdict(list)
    for row in page_rows:
        by_student[row["student_id"]].append(row)

    student_rows = []
    for student_id, rows in sorted(by_student.items()):
        first = rows[0]
        source_counts = Counter(r["text_source"] for r in rows)
        spelling_avg = average_float(rows, "spelling_score")
        student_rows.append(
            {
                "student_id": student_id,
                "place": first["place"],
                "grade": first["grade"],
                "gender": first["gender"],
                "pages_scored": len(rows),
                "ground_truth_pages": source_counts.get("ground_truth", 0),
                "ocr_pages": source_counts.get("ocr", 0),
                "total_words": sum(int(r["word_count"]) for r in rows),
                "avg_type_token_ratio": f"{average_float(rows, 'type_token_ratio') or 0.0:.4f}",
                "avg_structure_score": f"{average_float(rows, 'structure_score') or 0.0:.4f}",
                "avg_spelling_score": "" if spelling_avg is None else f"{spelling_avg:.4f}",
            }
        )
    common.write_csv(reports_dir / "text_scores_students.csv", student_rows)

    print(f"Scored {len(page_rows)} pages.")
    print(f"Wrote {reports_dir / 'text_scores_pages.csv'}")
    print(f"Wrote {reports_dir / 'text_scores_students.csv'}")
    if wordlist is None:
        print("No arabic_wordlist.txt found, so dictionary spelling metrics were left blank.")


if __name__ == "__main__":
    main()
