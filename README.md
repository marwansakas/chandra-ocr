# Arabic children handwriting OCR

This repository compares OCR workflows and trains a fourth, thesis-specific
line recognizer for Arabic elementary-school handwriting.

## Architecture

```text
split_papers/
  |
  +-- Surya -> line images + line geometry
                  |
                  +-- Chandra -> pseudo-labels for each line
                  |
                  +-- Invizo -> alternate line OCR baseline
                  |
                  +-- manually verified ground truth
                                  |
                                  +-- custom CTC training/inference

split_papers/ -> Invizo -> paragraph predictions

ground truth + engine predictions -> CER/WER comparison reports
```

Surya detects lines; it does not recognize their text. Chandra supplies initial
pseudo-labels. Invizo provides a second OCR baseline. The thesis model is a
custom line recognizer trained with CTC over full line images. Diffusion models
generate images and are not suitable for decoding text.

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
python pipelines/ctc.py
python thesis/scripts/evaluate_ctc_predictions.py
```

Invizo releases differ in how inference is launched. Its command is explicit so
the adapter works with the downloaded release without assuming a nonexistent
Python package. It must write one TXT/Markdown file, or a JSON file containing
`text`, `paragraph`, `transcription`, or `prediction`.

## Train the custom CTC recognizer

First replace or verify Chandra pseudo-labels with human ground truth. Prepare a
student-isolated split:

```bash
python thesis/scripts/train_ctc_line_ocr_model.py --prepare-only
```

Train on the full line crops:

```bash
python thesis/scripts/train_ctc_line_ocr_model.py \
  --epochs 20 --batch-size 8 --augment
```

Do not split a line into fragments and attach the full transcription to every
fragment; that gives the model false labels. The CTC model learns a
left-to-right character sequence from each complete line image.

The CTC pipeline writes line-level predictions to `thesis/predictions_ctc/` and
comparison reports to `thesis/reports/ctc_line_evaluation*`.

See `thesis/WORKFLOW.md` for metadata, evaluation, and reporting commands.

