# Thesis OCR Workflow

This folder turns the Chandra OCR output into reports you can use for comparing
children by page, student/notebook, grade, gender, and school/place.

## 1. Build and Link Metadata

```bash
python thesis/scripts/build_metadata.py
python thesis/scripts/collect_ocr.py
python thesis/scripts/report_coverage.py
```

Important outputs:

- `thesis/metadata/pages.jsonl`
- `thesis/metadata/ocr_index.jsonl`
- `thesis/reports/ocr_coverage_summary.md`
- `thesis/reports/missing_ocr_pages.csv`

## 2. Finish Missing OCR

Check missing pages first:

```bash
python thesis/scripts/run_missing_ocr.py --dry-run
```

Run Chandra on missing pages:

```bash
python thesis/scripts/run_missing_ocr.py --continue-on-error
python thesis/scripts/collect_ocr.py
python thesis/scripts/report_coverage.py
```

## 3. Create Ground Truth

Make a balanced labeling batch:

```bash
python thesis/scripts/sample_ground_truth.py --size 120
```

Then manually fill the empty files under:

```text
thesis/labels/ground_truth/
```

Each file should contain the correct text for that page.

## 4. Evaluate OCR

```bash
python thesis/scripts/evaluate_ocr.py
```

Use the OCR error results before judging spelling. A high WER/CER page should
be corrected manually before it is used for student spelling conclusions.

## 5. Score Text and Handwriting

```bash
python thesis/scripts/score_text.py
python thesis/scripts/score_handwriting.py
python thesis/scripts/compare_students.py
```

Text reports:

- `thesis/reports/text_scores_pages.csv`
- `thesis/reports/text_scores_students.csv`

Handwriting reports:

- `thesis/reports/handwriting_features_pages.csv`
- `thesis/reports/handwriting_features_students.csv`
- `thesis/reports/student_comparison.csv`

For handwriting features on PDFs, install PyMuPDF if it is not available:

```bash
pip install pymupdf
```

## 6. Refresh Everything Non-Destructive

```bash
python thesis/scripts/run_pipeline.py
```

This command does not run Chandra and does not overwrite manual labels.
