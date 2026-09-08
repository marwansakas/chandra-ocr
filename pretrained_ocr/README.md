# Pretrained OCR evaluation on ground_truth_lines

This module adds five independent OCR baselines to `chandra-ocr`. It runs published
weights **without training or fine-tuning**, then measures their predictions against
your labels. It does not require access to the models' original training datasets.

The requested Linux dataset is:

```text
/home/sakas/chandra-ocr/thesis/datasets/ground_truth_lines/
  000001_.../
    image.png
    ground_truth.txt
    metadata.json
  000002_.../
    image.png
    ground_truth.txt
    metadata.json
```

Only image pixels and a fixed model-specific prompt are sent to recognition.
`ground_truth.txt` is read for scoring. Existing `metadata.json` is not used as an
OCR prompt. No input files are modified. Every model gets a separate output folder.

## Models and checkpoints

| Command name | Model / published weights | Evaluation choice |
| --- | --- | --- |
| `paddle-arabic` | [PaddleOCR Arabic PP-OCRv5](https://huggingface.co/PaddlePaddle/arabic_PP-OCRv5_mobile_rec) | `arabic_PP-OCRv5_mobile_rec`; recognition only for your cropped lines |
| `got-ocr2` | [GOT-OCR2.0](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) | Native Transformers checkpoint; plain text mode |
| `glm-ocr` | [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | Official `Text Recognition:` prompt |
| `hunyuan-ocr` | [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) | Original 1.0 checkpoint in `v1.0/`, with a fixed transcription prompt |
| `lighton-ocr2` | [LightOnOCR-2](https://huggingface.co/lightonai/LightOnOCR-2-1B) | Image-only input; final OCR checkpoint, not the base training checkpoint |

Checkpoint revisions for the four Transformers models are pinned in `registry.py`,
verified September 7, 2026. Hunyuan's root now contains version 1.5, so the original
1.0 is selected explicitly. PaddleOCR uses its named published model and records
the installed package versions; its automatically downloaded weights are not pinned
by a Hugging Face commit.

The supplied S/A rankings are a candidate list, not measured results on your data.
Arabic support is explicit for the Paddle model. GOT's original release focuses
on English/Chinese; GLM's model card lists eight languages without Arabic;
LightOn's release emphasizes European languages. Treat these as experimental
Arabic baselines. Published weights and fine-tuning support do not establish
accuracy on children's handwriting. See the [GOT official repository](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
and [LightOn release discussion](https://lighton.ai/fr-blog-posts/lighton-opens-a-new-field-for-ai-with-lightonocr-2-document-intelligence).

## Put the module on the Linux project

The module belongs at `/home/sakas/chandra-ocr/pretrained_ocr/`, alongside
`pipelines/` and `thesis/`. Update your existing Linux checkout from GitHub:

```bash
cd /home/sakas/chandra-ocr
git pull --ff-only origin main
ls pretrained_ocr/models
```

The command fast-forwards the current branch to `origin/main`; it stops if local
commits have diverged or conflicting edits would be overwritten. It does not copy
or rebuild the dataset. Keep the existing dataset at
`thesis/datasets/ground_truth_lines/` and run all commands from the project root.

## Install once

Use Python 3.10 or later (3.12 preferred). Two environments keep Paddle dependencies
separate from the four Transformers models. Run the following from the project root:

```bash
cd /home/sakas/chandra-ocr

python3 -m venv .venv-ocr-hf
.venv-ocr-hf/bin/python -m pip install --upgrade pip
.venv-ocr-hf/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.venv-ocr-hf/bin/python -m pip install -r pretrained_ocr/requirements/hf.txt

python3 -m venv .venv-ocr-paddle
.venv-ocr-paddle/bin/python -m pip install --upgrade pip
.venv-ocr-paddle/bin/python -m pip install -r pretrained_ocr/requirements/paddle.txt
```

The PyTorch command is for an NVIDIA GPU with a driver supporting CUDA 12.8. For a
different driver/GPU, use the appropriate [official PyTorch installation command](https://pytorch.org/get-started/previous-versions/).
For CPU-only use, replace `/cu128` with `/cpu` in the PyTorch install command and
use `--device cpu` in the four run commands below. CPU VLM inference can be slow.
Paddle runs on CPU by default; its small recognition model is a useful first check.
Torchvision is required by GLM's processor even for still-image input.

For a Slurm server, run GPU inference inside a GPU allocation or batch job using
your site's partition/account settings. Check CUDA in that allocation:

```bash
.venv-ocr-hf/bin/python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

First inference downloads pretrained weights and needs network access. Subsequent
runs reuse the Hugging Face or Paddle cache. Each VLM has hundreds of millions to
about one billion parameters; allow several GB for caches and model runtime.
No CUDA, FlashAttention, vLLM, or custom remote Python code is needed for the CPU path.

## Validate the dataset

```bash
cd /home/sakas/chandra-ocr
DATA=/home/sakas/chandra-ocr/thesis/datasets/ground_truth_lines
.venv-ocr-hf/bin/python -m pretrained_ocr validate --dataset "$DATA" --check-images
```

Folders are sorted deterministically. Missing images/labels, blank labels, and labels
equal to `ignore` (case-insensitive) are skipped and reported. Other text containing
the word `ignore` is retained. Nested sample folders are supported. Use the same
dataset and sample range for every model. `metadata.json` is optional.

## Evaluate only the existing held-out test set (no training)

Use `--test-manifest` to restrict inference to the sample folders listed in the
project's existing `thesis/datasets/ground_truth_lines_ml/test.jsonl`. This does
not create or change a split, train, or fine-tune a model. It uses `sample_id` to
find each image under `--dataset`, and reads its current `ground_truth.txt` for
scoring; stored manifest image paths and text are not used as recognition inputs.

After updating the project and installing the environments above, run:

```bash
cd /home/sakas/chandra-ocr
DATA=/home/sakas/chandra-ocr/thesis/datasets/ground_truth_lines
TEST=/home/sakas/chandra-ocr/thesis/datasets/ground_truth_lines_ml/test.jsonl

.venv-ocr-hf/bin/python -m pretrained_ocr validate --dataset "$DATA" --test-manifest "$TEST" --check-images

.venv-ocr-paddle/bin/python -m pretrained_ocr run --model paddle-arabic --dataset "$DATA" --test-manifest "$TEST" --device cpu --resume
.venv-ocr-hf/bin/python -m pretrained_ocr run --model got-ocr2 --dataset "$DATA" --test-manifest "$TEST" --device cuda --resume
.venv-ocr-hf/bin/python -m pretrained_ocr run --model glm-ocr --dataset "$DATA" --test-manifest "$TEST" --device cuda --resume
.venv-ocr-hf/bin/python -m pretrained_ocr run --model hunyuan-ocr --dataset "$DATA" --test-manifest "$TEST" --device cuda --resume
.venv-ocr-hf/bin/python -m pretrained_ocr run --model lighton-ocr2 --dataset "$DATA" --test-manifest "$TEST" --device cuda --resume

.venv-ocr-hf/bin/python -m pretrained_ocr compare --results thesis/predictions/pretrained/held_out
```

Results default to `thesis/predictions/pretrained/held_out/<model>/` with the same
per-sample predictions, CSV scores and JSON reports described below. Full-dataset
results remain in their own folders. The manifest path and hash are recorded in
the run configuration. A missing manifest or an invalid, duplicate, missing or
unusable test sample stops the run before loading model weights; it never falls
back to evaluating the whole dataset. `--limit` and `--start-line` apply after the
held-out selection. If your existing test manifest is elsewhere, change `TEST`.

Use the same existing manifest for every model. If you already started a run
without `--test-manifest`, it remains a full-dataset run; this option takes effect
when you start a new command. The reported `ok` status means inference returned
successfully, not that the predicted text was correct. CER/WER measure correctness.

## Run and evaluate each model on the entire dataset

These commands use the **entire eligible dataset**, automatically score it, and
resume completed predictions when rerun. No separate scoring command is needed.

```bash
cd /home/sakas/chandra-ocr
DATA=/home/sakas/chandra-ocr/thesis/datasets/ground_truth_lines

# 1. PaddleOCR Arabic PP-OCRv5
.venv-ocr-paddle/bin/python -m pretrained_ocr run --model paddle-arabic --dataset "$DATA" --device cpu --resume

# 2. GOT-OCR2.0
.venv-ocr-hf/bin/python -m pretrained_ocr run --model got-ocr2 --dataset "$DATA" --device cuda --resume

# 3. GLM-OCR
.venv-ocr-hf/bin/python -m pretrained_ocr run --model glm-ocr --dataset "$DATA" --device cuda --resume

# 4. HunyuanOCR (original 1.0 checkpoint)
.venv-ocr-hf/bin/python -m pretrained_ocr run --model hunyuan-ocr --dataset "$DATA" --device cuda --resume

# 5. LightOnOCR-2
.venv-ocr-hf/bin/python -m pretrained_ocr run --model lighton-ocr2 --dataset "$DATA" --device cuda --resume

# Compare all completed runs on matching samples
.venv-ocr-hf/bin/python -m pretrained_ocr compare
```

For a first smoke run, add `--limit 5` to a model's command, then remove it to
evaluate all samples. `--resume` reuses those first five predictions. Keep identical
model settings. Use `--dry-run` to validate selection without loading weights or
writing outputs. `--start-line 101 --limit 100` selects the next block of lines;
use a separate `--output` for independently processed blocks. Comparison covers
the selection from each model's most recent invocation.

The default generation limit is 512 tokens per line. If the report flags token
limits, increase `--max-new-tokens` and use a fresh `--output` directory for all
models being compared. Hunyuan uses eager attention by default per its
[native Transformers usage guidance](https://huggingface.co/docs/transformers/model_doc/hunyuan_vl).

## Results and scoring

```text
thesis/predictions/pretrained/
  paddle-arabic/                 # same structure for every model
    samples/<sample-folder>/
      prediction.txt
      metadata.json
    predictions.jsonl
    scores.csv
    run_config.json
    run_summary.json
    skipped_samples.json
  got-ocr2/
  glm-ocr/
  hunyuan-ocr/
  lighton-ocr2/
  comparison.md
```

- **CER**: total character insertions/deletions/substitutions divided by total
  reference characters after normalization. Arabic diacritics do not count by
  default; spaces count. Lower is better; 0% means perfect.
- **WER**: the same calculation on whitespace-delimited words. Lower is better.
- **Exact lines**: the fraction of lines with an identical normalized transcription.
- Rates are corpus-weighted, can exceed 100%, and are not clamped.
- The default `arabic-light` scoring applies NFKC Unicode, collapses whitespace,
  and removes Arabic combining marks (tashkeel/harakat) and tatweel from both the
  reference and prediction before calculating CER, WER, and exact lines. For
  example, `كِتَابٌ` and `كتاب` match perfectly, and the reference has four scored
  characters. Raw reference and prediction strings remain in the reports.
- Punctuation, spelling and Arabic letter distinctions are preserved, including
  dots: `ب` and `ت`, `أ` and `ا`, `ة` and `ه`, and `ى` and `ي` remain different.
- `--normalization standard` selects the previous NFC/whitespace scoring, which
  counts diacritics. `--normalization raw` compares the exact file contents
  (including line endings in CER). Use consistent normalization for comparison.
- Existing runs retain their saved scores. To evaluate with the new default, use
  a fresh `--output` directory for each model; runs saved with `standard` cannot
  resume as `arabic-light`. To resume an older run with its original scoring,
  explicitly pass `--normalization standard`.
- Failed inference counts as an empty prediction in the main score. Successful-only
  scores and failure counts are also reported. A run with failures exits with code 1;
  rerunning with `--resume` retries failed lines.
- Resume checks the run settings and each cached sample's image/label fingerprint.
  Changed inputs or settings require a new output directory. Comparison rejects
  different datasets/selections, normalization, or interrupted runs.

These are pretrained baseline scores on your labels. If labels were produced by
another OCR system, the scores measure agreement with those labels; manually
verified transcriptions are needed to measure handwriting recognition accuracy.

## Code structure and validation

`models/` holds one adapter per model. `registry.py` identifies checkpoints;
`dataset.py` reads your layout; `metrics.py` scores text; `cli.py` coordinates
inference, resuming and reports. Models are loaded lazily, so validating the dataset
or comparing reports does not load weights.

Run the offline contract tests with the standard library:

```bash
python3 -m unittest discover -s pretrained_ocr/tests -v
```

Optional real-processor check (downloads configurations/tokenizers, no weights):

```bash
.venv-ocr-hf/bin/python -m pretrained_ocr.verify_hf_processors
```

Full inference and accuracy on your Linux dataset must be verified on that host;
the Windows workspace cannot access its dataset or GPU allocation.
