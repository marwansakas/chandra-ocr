# Complete line OCR experiment

Run from `/home/sakas/chandra-ocr`. Each phase initializes its own paths; it does
not depend on DATA or SPLIT variables in your terminal. The default experiment
name is `round_20260913`. For a later experiment prefix **every** phase with the
same `OCR_RUN=your_new_name` value. The split phase refuses an existing experiment.

Use your existing training environment for Python commands. Run GPU training
and inference inside a Slurm GPU allocation, following your cluster's allocation
rules. `TRAIN_PYTHON=/absolute/path/to/python` can select another training Python.
The pretrained phases use the existing `.venv-ocr-hf` and `.venv-ocr-paddle`
environments described in `pretrained_ocr/README.md`.

```bash
cd /home/sakas/chandra-ocr
git pull --ff-only origin main

# 1. Download only new top-level line folders; existing folders are skipped.
bash thesis/scripts/run_line_experiments.sh download

# 2. Prepare approximately 80/20 by lines, keeping student groups together.
bash thesis/scripts/run_line_experiments.sh split

# 3. Train and test the six additional TrOCR augmentation variants.
bash thesis/scripts/run_line_experiments.sh train-variants

# 4. Train TrOCR and CNN-TrOCR on original data (two sequential jobs).
bash thesis/scripts/run_line_experiments.sh train-real

# 5. Generate 1,000 training-only handwriting augmentations, then train both
# architectures and the six augmentation variants with the added samples.
bash thesis/scripts/run_line_experiments.sh synthetic
bash thesis/scripts/run_line_experiments.sh train-synthetic
bash thesis/scripts/run_line_experiments.sh train-variants-synthetic

# 6. Test local models and dictionary decoding, then external baselines.
bash thesis/scripts/run_line_experiments.sh test-local
bash thesis/scripts/run_line_experiments.sh test-pretrained
bash thesis/scripts/run_line_experiments.sh test-chandra
bash thesis/scripts/run_line_experiments.sh setup-invizo
bash thesis/scripts/run_line_experiments.sh test-invizo

# 7. Re-score every model with one normalization policy and one test manifest.
bash thesis/scripts/run_line_experiments.sh metrics
```

Run phases individually and stop on any error. Five epochs is a fixed starting
budget, not a validated optimum. The training scripts currently evaluate the test
partition every epoch; do not use those results to select hyperparameters.

The wrapper records hashes of dataset images, labels, metadata, and split files.
Later phases refuse changed data or splits. Trainers still recreate the split
deterministically, using the fixed settings. Do not modify data during a running
phase. Split snapshots are also saved in the experiment report directory.

TrOCR, CNN-TrOCR, and the augmentation variants support training. The five
pretrained adapters (GOT-OCR2, GLM-OCR, HunyuanOCR, LightOnOCR2, Paddle Arabic)
currently support inference only; these commands do not fine-tune them. The
dictionary decoder reuses trained TrOCR weights. Legacy checkpoints trained on
an older split are not included because they may have seen the new test lines.

## Invizo

The [official documentation](https://github.com/Hedrax/Invizo-OCR) describes a
desktop application. Our line runner calls its recognition module directly on
already cropped images, with upstream preprocessing, vocabulary, and greedy
decoding. It does not run document detection or template matching.

Setup fetches `OCRModule.py` and `OCRConfig.py` at commit
`2b6f416b6218adc87de06c941382b2f708749090` and the recognition model from
[v0.2-alpha](https://github.com/Hedrax/Invizo-OCR/releases/tag/v0.2-alpha).
It uses a separate ONNX Runtime CPU environment. Prediction failures are surfaced
instead of accepting the partial strings that upstream can return on an error.

## Reports

Outputs are under `thesis/reports/round_20260913/comparison/`:

- `summary.csv`: one row per model.
- `per_line.csv`: every model/sample pair and normalized reference/prediction.
- `summary.json`: results, test-manifest hash, and metric definitions.

The evaluator removes Arabic combining diacritics and tatweel, applies NFKC, and
normalizes whitespace to both sides. Hamza letters remain distinct. It uses
whitespace-delimited words for WER. CER/WER are corpus edit rates and can exceed
1; exact match and alignment scores are averaged per line. All rates are fractions
(0.12 means 12%). Character/word accuracies are max(0, 1 - error rate).
Smith-Waterman measures local character alignment; text alignment uses global
Needleman-Wunsch. Both use match +2, mismatch -1, gap -1, normalized by twice the
longer line and clipped to [0,1]. Missing predictions stop comparison rather than
changing which samples are scored.

The combined report requires all 25 configured runs: four trained architectures/
data combinations, twelve augmentation runs, two dictionary decoders, five
pretrained adapters, Chandra, and Invizo. To compare a smaller explicitly chosen
set, invoke `evaluate_all_line_models.py` directly with repeated
`--prediction NAME=PATH` arguments, pointing to each folder containing
`sample_id/prediction.txt`. Pretrained adapters use their `samples` subfolder.
