# Arabic children handwriting OCR

This project creates line images from elementary-school Arabic handwriting and
runs independent OCR systems over the same images:

1. Chandra
2. Invizo
3. The fine-tuned thesis TrOCR model
4. General-Arabic dictionary-aware TrOCR decoding
5. Experimental CNN-fronted TrOCR
6. PaddleOCR Arabic PP-OCRv5 (pretrained)
7. GOT-OCR2.0 (pretrained)
8. GLM-OCR (pretrained)
9. HunyuanOCR (pretrained)
10. LightOnOCR-2 (pretrained)

For the five pretrained baselines, see the
[installation and evaluation guide](pretrained_ocr/README.md). Each reads the
existing `ground_truth_lines/<sample>/image.png` layout and reports CER, WER,
exact-line matches and failures against `ground_truth.txt`, without fine-tuning.
Add `--test-manifest thesis/datasets/ground_truth_lines_ml/test.jsonl` to evaluate
only your existing held-out samples. Those results go under
`thesis/predictions/pretrained/held_out/<model>/`.

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
        +--> cnn_TrOCR_run.py
                             --> thesis/predictions/cnn_TrOCR/
        +--> python -m pretrained_ocr run --model <model>
                             --> thesis/predictions/pretrained/<model>/
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
  --batch-size 4 \
  --views-per-line 3 \
  --augmentation-profile light
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

## 7. TrOCR augmentation experiments

To generate separate Arabic synthetic training pairs or variations of your real
training handwriting, see [Synthetic data](thesis/SYNTHETIC_DATA.md).
The standard, CNN, and augmentation trainers accept --synthetic-root.

The baseline TrOCR trainer can produce multiple models from the same data split
by changing only the image augmentation used for extra training views:

```bash
python thesis/scripts/train_trocr_augmentation_models.py \
  --profiles none,light,scan,geometry,stroke,mixed \
  --epochs 5 \
  --batch-size 4 \
  --views-per-line 3 \
  --device cuda
```

Each profile still trains the same `my_TrOCR` architecture. The profiles save
separate checkpoints under:

```text
thesis/models/augmentation_trocr/<profile>/
```

and separate predictions under:

```text
thesis/predictions/augmentation_trocr/<profile>/
```

The launcher writes the augmentation comparison to:

```text
thesis/reports/trocr_augmentation_experiments.csv
```

Available profiles:

- `none`: no image changes; control model
- `light`: brightness, contrast, and white margin changes
- `scan`: brightness, contrast, blur/sharpen, and scanner-like intensity noise
- `geometry`: small rotation, translation, and width scaling
- `stroke`: small stroke thickening or thinning
- `mixed`: combines scan, geometry, and stroke changes

## 8. Experimental CNN-fronted TrOCR

The CNN experiment adds a small residual convolutional block before the TrOCR
Transformer encoder. It is trained and stored separately from the baseline so
the baseline remains a clean comparison point.

Train the CNN-fronted model on the same split:

```bash
python thesis/scripts/train_cnn_trocr_model.py \
  --epochs 5 \
  --batch-size 4
```

The model is saved to:

```text
thesis/models/ground_truth_line_cnn_trocr/
```

Predict only the held-out test split:

```bash
python thesis/scripts/predict_cnn_trocr_test_samples.py --device cuda
```

To run it over the clean line dataset:

```bash
python pipelines/cnn_TrOCR_run.py \
  --model thesis/models/ground_truth_line_cnn_trocr \
  --input-root thesis/datasets/ground_truth_lines \
  --output-root thesis/predictions/cnn_TrOCR \
  --batch-size 8 \
  --device cuda
```

## 9. General-Arabic dictionary-aware TrOCR

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

## 10. Compare all prediction models on the held-out test split

After Chandra, Invizo, raw TrOCR, dictionary-aware TrOCR, and CNN-TrOCR
predictions exist, compare them against the same ground-truth test lines:

```bash
python thesis/scripts/compare_line_ocr_models.py
```

Outputs:

```text
thesis/reports/line_ocr_model_comparison.csv
thesis/reports/line_ocr_model_comparison_summary.csv
```

The report includes letter accuracy, word accuracy, and exact-line accuracy for
Chandra, Invizo, raw thesis TrOCR, general-Arabic dictionary-aware TrOCR, and
CNN-TrOCR.

All line scores use the project `arabic-light` normalization: Arabic combining
marks/tashkeel and tatweel are removed before scoring. Reports also include
Smith–Waterman local character similarity and whole-text global alignment
similarity. The pretrained baselines write the same metrics under
`thesis/predictions/pretrained/held_out/` and can be summarized with:

```bash
python -m pretrained_ocr compare --results thesis/predictions/pretrained/held_out
```

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

