# Arabic children handwriting OCR

This repository compares three OCR engines and trains a fourth, thesis-specific
line recognizer for Arabic elementary-school handwriting.

## Architecture

```text
split_papers/
  |
  +-- Surya -> line images + line geometry
                  |
                  +-- Chandra -> pseudo-labels for each line
                  |
                  +-- manually verified ground truth
                                   |
                                   +-- thesis TrOCR training/inference

split_papers/ -> Invizo -> paragraph predictions

ground truth + engine predictions -> CER/WER comparison reports
```

Surya detects lines; it does not recognize their text. Chandra supplies initial
pseudo-labels. Invizo is evaluated independently at paragraph/page level. The
thesis model is a TrOCR vision-encoder/text-decoder OCR model, not a diffusion
model. Diffusion models generate images and are not suitable for decoding text.

## External repositories and data

Keep downloaded upstream repositories in `external/` or provide their commands
on the CLI. The folder is ignored because those projects have their own Git
history:

```text
external/
  surya/
  chandra/
  invizo/
```

The ignored `split_papers/`, model weights, and generated output directories are
expected to exist locally.

## Run each engine

```bash
python pipelines/surya.py --limit 300
python pipelines/chandra.py --continue-on-error
python pipelines/invizo.py \
  --command "python external/invizo/infer.py --input {input} --output {output}"
python pipelines/thesis.py
```

Invizo releases differ in how inference is launched. Its command is explicit so
the adapter works with the downloaded release without assuming a nonexistent
Python package. It must write one TXT/Markdown file, or a JSON file containing
`text`, `paragraph`, `transcription`, or `prediction`.

## Train the thesis recognizer

First replace or verify Chandra pseudo-labels with human ground truth. Prepare a
student-isolated split:

```bash
python thesis/scripts/train_line_ocr_model.py --prepare-only
```

Train with three full-line views per sample:

```bash
python thesis/scripts/train_line_ocr_model.py \
  --epochs 5 --batch-size 4 --views-per-line 3
```

Extra views preserve the whole line and apply brightness/contrast/border
augmentation. Do not split a line into fragments and attach the full
transcription to every fragment; that gives the model false labels.

See `thesis/WORKFLOW.md` for metadata, evaluation, and reporting commands.

