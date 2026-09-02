# Arabic children handwriting OCR

This project creates line images from elementary-school Arabic handwriting and
runs three independent OCR systems plus one dictionary-aware TrOCR variant over
the same images:

1. Chandra
2. Invizo
3. The fine-tuned thesis TrOCR model
4. General-Arabic dictionary-aware TrOCR decoding

The clean labeled line dataset supports held-out evaluation. Chandra-generated
pseudo-labels must not be treated as manual ground truth.

## Project flow

```text
split_papers/**/*.pdf
        |
        v
Surya line detection
        |
        v
thesis/datasets/ground_truth_lines/<global>_<page>_<line>/image.png
        |
        +--> chandra_run.py  --> thesis/predictions/chandra/
        +--> invizo_run.py   --> thesis/predictions/invizo/
        +--> my_TrOCR_run.py --> thesis/predictions/my_TrOCR/
        +--> general_arabic_trocr_run.py
                             --> thesis/predictions/general_arabic_TrOCR/
```

The OCR runners read only `image.png`. Training and evaluation read
`ground_truth.txt` from the clean `ground_truth_lines` dataset.

## 1. Create Surya line images

Surya recursively reads the already ranked PDFs under `ranked_pdfs` and
processes every PDF by default. It renders the first page of each selected PDF
and creates one folder per detected line.

First batch, PDFs 1-100:

```bash
python pipelines/surya.py \
  --start-pdf 1 \
  --limit 100 \
  --continue-on-error
```

Next batch, PDFs 101-200:

```bash
python pipelines/surya.py \
  --start-pdf 101 \
  --limit 100 \
  --continue-on-error
```

To split all currently ranked PDFs in one run, omit `--limit`:

```bash
python pipelines/surya.py --continue-on-error
```

Surya output:

```text
thesis/datasets/surya_lines/
└── 0001_pdf_name/
    ├── metadata.json
    ├── lines.jsonl
    └── line_001/
        ├── image.png
        ├── ground_truth.txt
        └── metadata.json
```

`ground_truth.txt` starts empty and is reserved for later manual labeling.

## 2. Create the clean labeled line dataset

After manually filling `ground_truth.txt` files, build the clean dataset:

```bash
python build_ground_truth_lines_dataset.py --pages 100 --force
```

To continue from a later sorted page folder:

```bash
python build_ground_truth_lines_dataset.py --start-folder 101 --pages 100 --force
```

The result is:

```text
thesis/datasets/ground_truth_lines/
└── 000001_0001_pdf_name_line_001/
    ├── image.png
    ├── ground_truth.txt
    └── metadata.json
```

Empty labels and labels containing `ignore` are skipped.

## 3. Train the thesis TrOCR model

With 1099 usable labeled lines, the default 80/20 split uses about 879 lines
for training and 220 lines for held-out testing. First create the split:

```bash
rm -rf thesis/datasets/ground_truth_lines_ml
python thesis/scripts/train_line_ocr_model.py --prepare-only --split-strategy line
```

Then train:

```bash
python thesis/scripts/train_line_ocr_model.py \
  --epochs 5 \
  --batch-size 4
```

The model is saved to:

```text
thesis/models/ground_truth_line_trocr/
```

Predict only the held-out test split:

```bash
python thesis/scripts/predict_trocr_test_samples.py --device cuda
```

## 4. Chandra runner

Activate the environment containing Chandra, then run:

```bash
conda activate chandra

python pipelines/chandra_run.py \
  --input-root thesis/datasets/ground_truth_lines \
  --output-root thesis/predictions/chandra \
  --test-manifest thesis/datasets/ground_truth_lines_ml/test.jsonl \
  --chandra "$(command -v chandra)" \
  --method hf \
  --continue-on-error
```

Continue with line 101 using `--start-line 101`. Existing predictions are
skipped unless `--force` is provided.

## 5. Invizo runner

Invizo releases do not expose one stable command-line interface. Supply the
inference command from the downloaded Invizo repository. The template must
contain `{input}` and `{output}`:

```bash
python pipelines/invizo_run.py \
  --input-root thesis/datasets/ground_truth_lines \
  --output-root thesis/predictions/invizo \
  --test-manifest thesis/datasets/ground_truth_lines_ml/test.jsonl \
  --command "python external/invizo/infer.py --input {input} --output {output}" \
  --continue-on-error
```

Replace `python external/invizo/infer.py ...` with the actual launcher from the
local Invizo checkout. Its result may be one TXT/Markdown file or a JSON file
with a `text`, `paragraph`, `transcription`, or `prediction` field.

## 6. Thesis TrOCR runner

To run the trained TrOCR model over the same clean line dataset:

```bash
python pipelines/my_TrOCR_run.py \
  --model thesis/models/ground_truth_line_trocr \
  --input-root thesis/datasets/ground_truth_lines \
  --output-root thesis/predictions/my_TrOCR \
  --batch-size 8 \
  --device cuda
```

TrOCR uses a ViT image encoder and autoregressive text decoder.

## 7. General-Arabic dictionary-aware TrOCR

This decoder does not retrain the image model. It generates six TrOCR beam
candidates, checks Arabic word forms with the LibreOffice Hunspell dictionary,
and selects a dictionary-supported candidate only when it beats the raw TrOCR
prediction by a configurable margin. Otherwise it preserves the raw prediction.

Install the decoder dependency and download the pinned dictionary:

```bash
python -m pip install -r requirements.txt
python thesis/scripts/download_general_arabic_dictionary.py
```

Run it on the held-out test split:

```bash
python pipelines/general_arabic_trocr_run.py \
  --model thesis/models/ground_truth_line_trocr \
  --test-manifest thesis/datasets/ground_truth_lines_ml/test.jsonl \
  --device cuda
```

Each line output contains:

```text
prediction.txt       # selected dictionary-aware result or raw fallback
raw_prediction.txt   # best raw TrOCR beam
candidates.json      # beams, scores, coverage, and proposed corrections
metadata.json
```

The dictionary is a linguistic decoding aid, not a source of ground truth. The
Grade 1 decoder is intentionally deferred until the correct curriculum is
available.

## 8. Compare all prediction models on the held-out test split

After Chandra, Invizo, raw TrOCR, and dictionary-aware TrOCR predictions exist,
compare them against the same ground-truth test lines:

```bash
python thesis/scripts/compare_line_ocr_models.py
```

Outputs:

```text
thesis/reports/line_ocr_model_comparison.csv
thesis/reports/line_ocr_model_comparison_summary.csv
```

The report includes letter accuracy, word accuracy, and exact-line accuracy for
Chandra, Invizo, raw thesis TrOCR, and general-Arabic dictionary-aware TrOCR.

## Prediction output

Each runner owns a separate output tree:

```text
thesis/predictions/<model>/<pdf>/<line>/
├── prediction.txt
├── metadata.json
└── raw/                 # Chandra/Invizo native output, when applicable
```

Each run also creates:

```text
predictions.jsonl
run_summary.json
```

No runner copies predictions into `ground_truth.txt`.

## Resuming and previewing

Use `--start-line` and `--limit` to process batches. Use `--dry-run` to preview
selected images without loading or running a model:

```bash
python pipelines/chandra_run.py --start-line 101 --limit 100 --dry-run
```

Use `--force` only when existing predictions should be regenerated.

