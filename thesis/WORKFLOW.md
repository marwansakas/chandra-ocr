# Thesis OCR Workflow

This folder turns the Chandra OCR output into reports you can use for comparing
children by page, student/notebook, grade, gender, and school/place.

The four systems have separate entry points:

```bash
python pipelines/surya.py --limit 300
python pipelines/chandra.py --continue-on-error
python pipelines/invizo.py --command "YOUR INVIZO COMMAND {input} {output}"
python pipelines/ctc.py
```

Surya creates line crops and geometry. Chandra recognizes those line crops.
Invizo produces independent OCR predictions. The thesis model is a custom CTC
line recognizer trained on verified line pairs.

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

Chandra-generated labels are pseudo-labels until a person verifies them. Do not
report evaluation against unverified Chandra output as human ground truth.

## 5. Train and Run the Custom CTC Line Model

Create train/test manifests grouped by notebook/student:

```bash
python thesis/scripts/train_ctc_line_ocr_model.py --prepare-only
```

Train with label-preserving full-line augmentation:

```bash
python thesis/scripts/train_ctc_line_ocr_model.py \
  --epochs 20 --batch-size 8 --augment
```

Generate current thesis-model predictions:

```bash
python pipelines/ctc.py
python thesis/scripts/evaluate_ctc_predictions.py
```

## 6. Score Text and Handwriting

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

## 7. Refresh Everything Non-Destructive

```bash
python thesis/scripts/run_pipeline.py
```

This command does not run Chandra and does not overwrite manual labels.
