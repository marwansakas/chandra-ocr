# Arabic children handwriting OCR

This project creates line images from elementary-school Arabic handwriting and
runs three independent OCR systems over the same images:

1. Chandra
2. Invizo
3. The fine-tuned thesis TrOCR model

Accuracy evaluation is intentionally not part of the current workflow because
the manual ground-truth labels are not ready yet.

## Project flow

```text
split_papers/**/*.pdf
        |
        v
Surya line detection
        |
        v
thesis/datasets/surya_lines/<pdf>/<line>/image.png
        |
        +--> chandra_run.py  --> thesis/predictions/chandra/
        +--> invizo_run.py   --> thesis/predictions/invizo/
        +--> my_TrOCR_run.py --> thesis/predictions/my_TrOCR/
```

The OCR runners read only `image.png`. They do not read, require, modify, or
score `ground_truth.txt`.

## 1. Create Surya line images

Surya recursively sorts the PDFs under `split_papers` and processes 100 PDFs by
default. It renders the first page of each selected PDF and creates one folder
per detected line.

First batch, PDFs 1-100:

```bash
python pipelines/surya.py \
  --input-root split_papers \
  --start-pdf 1 \
  --limit 100 \
  --continue-on-error
```

Next batch, PDFs 101-200:

```bash
python pipelines/surya.py \
  --input-root split_papers \
  --start-pdf 101 \
  --limit 100 \
  --continue-on-error
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

## 2. Chandra runner

Activate the environment containing Chandra, then run:

```bash
conda activate chandra

python pipelines/chandra_run.py \
  --input-root thesis/datasets/surya_lines \
  --output-root thesis/predictions/chandra \
  --chandra "$(command -v chandra)" \
  --method hf \
  --start-line 1 \
  --limit 100 \
  --continue-on-error
```

Continue with line 101 using `--start-line 101`. Existing predictions are
skipped unless `--force` is provided.

## 3. Invizo runner

Invizo releases do not expose one stable command-line interface. Supply the
inference command from the downloaded Invizo repository. The template must
contain `{input}` and `{output}`:

```bash
python pipelines/invizo_run.py \
  --input-root thesis/datasets/surya_lines \
  --output-root thesis/predictions/invizo \
  --command "python external/invizo/infer.py --input {input} --output {output}" \
  --start-line 1 \
  --limit 100 \
  --continue-on-error
```

Replace `python external/invizo/infer.py ...` with the actual launcher from the
local Invizo checkout. Its result may be one TXT/Markdown file or a JSON file
with a `text`, `paragraph`, `transcription`, or `prediction` field.

## 4. Thesis TrOCR runner

The TrOCR runner is ready, but it does not need to be run until the model and
line dataset are ready:

```bash
python pipelines/my_TrOCR_run.py \
  --model thesis/models/surya_chandra_line_ocr \
  --input-root thesis/datasets/surya_lines \
  --output-root thesis/predictions/my_TrOCR \
  --batch-size 8 \
  --device cuda
```

TrOCR uses a ViT image encoder and autoregressive text decoder. It processes a
complete line image and writes predicted text. It does not require ground truth
during inference.

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

