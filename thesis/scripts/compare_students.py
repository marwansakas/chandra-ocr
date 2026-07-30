"""
Build a student/notebook comparison table from available scores.

Reads:
  thesis/reports/text_scores_students.csv
  thesis/reports/handwriting_features_students.csv

Writes:
  thesis/reports/student_comparison.csv
  thesis/reports/student_scores_easy.csv
  thesis/reports/student_scores_summary.md

Run:
  python thesis/scripts/compare_students.py
"""

from __future__ import annotations

import csv
from collections import defaultdict

import common


def read_csv(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except ValueError:
        return None


def score_100(value):
    number = as_float(value)
    if number is None:
        return ""
    return f"{max(0.0, min(100.0, number * 100)):.1f}"


def label_for_score(value):
    number = as_float(value)
    if number is None:
        return "not available"
    if number >= 0.80:
        return "strong"
    if number >= 0.65:
        return "good"
    if number >= 0.50:
        return "needs review"
    return "weak"


def md_table(rows, columns):
    lines = []
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return lines


def main() -> None:
    cfg = common.load_config()
    reports_dir = common.resolve(cfg["reports_dir"])

    text_rows = read_csv(reports_dir / "text_scores_students.csv")
    hand_rows = read_csv(reports_dir / "handwriting_features_students.csv")
    hand_by_student = {r["student_id"]: r for r in hand_rows}

    comparison = []
    for row in text_rows:
        hand = hand_by_student.get(row["student_id"], {})
        structure = as_float(row.get("avg_structure_score"))
        spelling = as_float(row.get("avg_spelling_score"))
        handwriting = as_float(hand.get("avg_handwriting_legibility_proxy"))

        available_scores = [v for v in [structure, spelling, handwriting] if v is not None]
        combined = sum(available_scores) / len(available_scores) if available_scores else None
        comparison.append(
            {
                "student_id": row["student_id"],
                "place": row["place"],
                "grade": row["grade"],
                "gender": row["gender"],
                "text_pages_scored": row["pages_scored"],
                "ground_truth_pages": row["ground_truth_pages"],
                "ocr_pages": row["ocr_pages"],
                "total_words": row["total_words"],
                "avg_structure_score": row["avg_structure_score"],
                "avg_spelling_score": row["avg_spelling_score"],
                "handwriting_pages_scored": hand.get("pages_scored", "0"),
                "avg_handwriting_legibility_proxy": hand.get("avg_handwriting_legibility_proxy", ""),
                "combined_available_score": "" if combined is None else f"{combined:.4f}",
                "score_note": "uses available metrics only; prefer ground-truth pages for final thesis claims",
            }
        )

    comparison.sort(
        key=lambda r: (
            r["grade"],
            -(as_float(r["combined_available_score"]) or -1.0),
            r["student_id"],
        )
    )
    common.write_csv(reports_dir / "student_comparison.csv", comparison)

    easy_rows = []
    by_grade = defaultdict(list)
    for row in comparison:
        easy = {
            "rank_in_grade": "",
            "student_id": row["student_id"],
            "place": row["place"],
            "grade": row["grade"],
            "gender": row["gender"],
            "overall_score_100": score_100(row["combined_available_score"]),
            "structure_score_100": score_100(row["avg_structure_score"]),
            "spelling_score_100": score_100(row["avg_spelling_score"]),
            "handwriting_score_100": score_100(row["avg_handwriting_legibility_proxy"]),
            "pages_scored": row["text_pages_scored"],
            "ground_truth_pages": row["ground_truth_pages"],
            "ocr_pages": row["ocr_pages"],
            "confidence": "better" if int(row["ground_truth_pages"]) > 0 else "ocr-only",
            "level": label_for_score(row["combined_available_score"]),
        }
        easy_rows.append(easy)
        by_grade[easy["grade"]].append(easy)

    for grade, rows in by_grade.items():
        rows.sort(
            key=lambda r: (
                -(as_float(r["overall_score_100"]) or -1.0),
                r["student_id"],
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["rank_in_grade"] = rank

    easy_rows.sort(key=lambda r: (str(r["grade"]), int(r["rank_in_grade"]), r["student_id"]))
    easy_fields = [
        "rank_in_grade",
        "student_id",
        "place",
        "grade",
        "gender",
        "overall_score_100",
        "structure_score_100",
        "spelling_score_100",
        "handwriting_score_100",
        "pages_scored",
        "ground_truth_pages",
        "ocr_pages",
        "confidence",
        "level",
    ]
    common.write_csv(reports_dir / "student_scores_easy.csv", easy_rows, fields=easy_fields)

    summary_lines = [
        "# Student Scores Summary",
        "",
        "Scores are shown from 0 to 100. Higher is better.",
        "",
        "Important: rows marked `ocr-only` are based on Chandra output, not verified ground truth. Use them for exploration, not final thesis claims.",
        "",
    ]
    for grade in sorted(by_grade.keys(), key=str):
        rows = sorted(
            by_grade[grade],
            key=lambda r: (
                -(as_float(r["overall_score_100"]) or -1.0),
                r["student_id"],
            ),
        )
        summary_lines.extend(
            [
                f"## Grade {grade}",
                "",
                f"Students scored: {len(rows)}",
                "",
            ]
        )
        summary_lines.extend(
            md_table(
                rows[:10],
                [
                    ("Rank", "rank_in_grade"),
                    ("Student", "student_id"),
                    ("Place", "place"),
                    ("Gender", "gender"),
                    ("Overall", "overall_score_100"),
                    ("Text Structure", "structure_score_100"),
                    ("Spelling", "spelling_score_100"),
                    ("Handwriting", "handwriting_score_100"),
                    ("Pages", "pages_scored"),
                    ("Confidence", "confidence"),
                    ("Level", "level"),
                ],
            )
        )
        summary_lines.append("")

    summary_lines.extend(
        [
            "## How To Read This",
            "",
            "- `Overall`: average of available score types.",
            "- `Text Structure`: punctuation, sentence-length, and repetition proxy score.",
            "- `Spelling`: available only if you add a wordlist or manual scoring.",
            "- `Handwriting`: available only after PDF image rendering is installed and handwriting features are extracted.",
            "- `Confidence`: `ocr-only` means OCR errors may affect the score.",
        ]
    )
    summary_path = reports_dir / "student_scores_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Wrote {reports_dir / 'student_comparison.csv'}")
    print(f"Wrote {reports_dir / 'student_scores_easy.csv'}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
