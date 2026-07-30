"""
Run the non-destructive thesis pipeline steps.

This does not run Chandra and does not overwrite manual labels. It refreshes
metadata, links existing OCR, reports coverage, evaluates any filled labels,
and scores available text/handwriting features.

Run:
  python thesis/scripts/run_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


STEPS = [
    "build_metadata.py",
    "collect_ocr.py",
    "report_coverage.py",
    "evaluate_ocr.py",
    "score_text.py",
    "score_handwriting.py",
    "compare_students.py",
]


def main() -> None:
    for step in STEPS:
        script = THIS_DIR / step
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
