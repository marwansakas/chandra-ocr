# Frozen TrOCR + text diffusion refiner

This additional model caches a frozen TrOCR encoder's visual tokens and its
draft transcription, then trains a small bidirectional transformer to reconstruct
text using both. The existing diffusion recognizer and existing TrOCR trainers
are unchanged. The default refiner has width 192, three layers and six heads;
the large TrOCR backbone receives no gradient updates.

This is experimental. Local checks demonstrate functioning training/inference,
not improved recognition of children's handwriting. Full accuracy is measured
by the cluster workflow below on the actual held-out data.

## Method and model selection

- Character-level targets preserve literal Arabic Unicode order, dots, spelling,
  spaces and diacritics. A single EOS is supervised; subsequent PAD positions
  are excluded from loss and replaced with MASK before decoding. Target lengths
  and labels never enter inference.
- The existing absorbing forward process masks each target independently with
  probability `t=k/T`. The objective is masked CE divided by `t` and the number
  of valid positions. By default 25% of training examples instead use full-mask
  reconstruction, an explicit mixture objective rather than the pure ELBO.
- The reverse kernel reveals masked positions with probability `(t-s)/t`, then
  samples a character from the image/draft-conditioned transformer. Already
  revealed positions stay fixed. The OCR draft is separate conditioning, not
  an irreversible initial output. The refiner can output different text lengths.
- Whole-draft dropout (50%) reduces dependence on potentially memorized drafts.
  For a project-trained base, drafts are in-sample training predictions. This is
  a practical first version, **not out-of-fold refinement**; dropout does not
  eliminate the resulting train/inference mismatch. A base trained on external
  data avoids this particular project-data mismatch.
- Every epoch compares one-pass greedy reconstruction and eight-step ancestral
  diffusion on validation. The best learned checkpoint/mode is saved. If its CER
  does not beat the unchanged base recognizer, the selected strategy remains the
  base recognizer. This selection occurs before test evaluation. It cannot
  guarantee an improvement on unseen test data.
- The one-pass result is a decoding ablation of the same trained model, not an
  independently trained conventional correction transformer. An optional control
  can be trained in a separate run with `--full-mask-probability 1`.
- Selection defaults to `arabic-light` CER (NFKC, whitespace normalization,
  removal of tatweel and Arabic combining marks). Both literal raw CER/WER and
  Arabic-light metrics are reported. Use `--selection-normalization raw` to select
  on literal CER. Neither mode performs spelling correction.

The architecture draws on image-conditioned discrete text diffusion such as
[DiffusionSTR](https://arxiv.org/abs/2306.16707). It is an independent extension
of this repository's absorbing-mask implementation, not a paper reproduction.

## Install on the cluster

Use Python 3.10+ (3.12 recommended), an environment with the university-supported
CUDA PyTorch build, and an allocated GPU. Preserve your existing OCR environment;
a separate virtualenv with access to that installed PyTorch is suitable:

```bash
cd /home/sakas/chandra-ocr
python -m venv --system-site-packages .venv-refiner
source .venv-refiner/bin/activate
python -m pip install -r requirements-refiner.txt
python -m pytest tests/test_trocr_diffusion_refiner.py -q
```

The tests use tiny random local weights, synthetic colored rectangles and Arabic
fixture labels. They need no downloaded models, project data, or GPU. Installing
requirements alone does not establish that your PyTorch build supports the GPU;
the run script explicitly checks CUDA before starting.

## Choose the dataset and split

Reuse your existing compatible frozen three-way split. The files must include
`split_lock.json`, `train.jsonl`, `validation.jsonl`, `test.jsonl`, `audit.json`
and `tokenizer.json`. The two-way first-100-page pilot cannot select checkpoints
and is rejected. Dataset files, labels, metadata, vocabulary and split membership
are verified against their saved hashes.

For a NEW split with authoritative writer IDs:

```bash
python -m thesis.refiner prepare \
  --dataset-root thesis/datasets/ground_truth_lines \
  --split-root thesis/datasets/refiner_split_v1 \
  --split-kind writer --writer-map /path/to/reviewed_writer_map.json
```

The map is `{ "sample_folder_id": "child_id", ... }`. Do not infer child identity
from page numbers. If writer identities are unavailable, you can explicitly run
the same command with `--split-kind line` and omit `--writer-map`. This uses
70/10/20 line partitions with exact duplicate images kept together, but may share
children across partitions. Report it as a line-split experiment, not performance
on unseen children. Existing splits are never overwritten; audit issues are
reported rather than silently excluding lines.

## Choose the frozen base

Preferred: a complete local Arabic TrOCR checkpoint fitted only on the selected
training partition, with a saved TrOCR processor. Set `BASE_MODEL` to that directory
and `BASE_TRAINING_MANIFEST` to the manifest actually used to train it. Each JSONL
record must contain `sample_id`, literal `text` and `image_sha256`, consistent with
the current frozen training records. The existing frozen `train.jsonl` has this
format, but use it only if it really describes the base's training data. The base
checkpoint must also have been selected without consulting the final test set.
Held-out/unknown training IDs, changed labels and changed images are rejected.
Unknown provenance cannot be made safe by relabeling an old manifest.

Runnable fallback: the published `microsoft/trocr-base-handwritten`, pinned to
`eaacaf452b06415df8f10bb6fad3a4c11e609406`. The shell script uses this automatically
if `BASE_MODEL` is unset. It supplies transferable image features, but its English
handwriting decoder is **not a strong Arabic baseline**. Training the new character
decoder on Arabic is still required; do not assume this fallback will beat your
fine-tuned Arabic TrOCR. For another externally pretrained base, explicitly set
`EXTERNAL_BASE=1` and, for a remote model, an immutable `BASE_REVISION` commit.

The first cache build downloads remote weights/configuration if needed. If GPU
nodes have no network, pre-download the pinned snapshot on an allowed node and
use that directory as `BASE_MODEL` with `EXTERNAL_BASE=1`. No custom remote Python
code is executed. A directory with configuration but no weights is rejected.

## Train and test in one allocated GPU job

Set real paths; `SPLIT_ROOT` below is the new-split example above. Use a new
`RUN_ROOT` for each experiment. The cache can be reused if the split, base,
extraction settings and package versions match exactly.

```bash
cd /home/sakas/chandra-ocr
source .venv-refiner/bin/activate
export DATASET_ROOT="$PWD/thesis/datasets/ground_truth_lines"
export SPLIT_ROOT="$PWD/thesis/datasets/refiner_split_v1"
export CACHE_ROOT="$PWD/thesis/refiner_artifacts/cache_v1"
export RUN_ROOT="$PWD/thesis/refiner_artifacts/run_v1"
# Optional, only for a compatible project-trained Arabic TrOCR:
# export BASE_MODEL=/absolute/path/to/complete/arabic_trocr_checkpoint
# export BASE_TRAINING_MANIFEST=/absolute/path/to/its/actual/train.jsonl
bash thesis/scripts/run_trocr_diffusion_refiner.sh
```

The script caches train/validation data, trains up to 30 epochs with validation
early stopping (patience 7), locks model selection, then caches and evaluates the
test partition. It records original, one-pass, diffusion and selected outputs on
the same complete test sample set. `EPOCHS`, `PATIENCE`, `BATCH_SIZE`,
`CACHE_BATCH_SIZE`, `SAMPLING_STEPS`, `MAX_NEW_TOKENS`, `SEED`, `PYTHON` and
`REPORT_ROOT` can be configured through environment variables. Reduce
`CACHE_BATCH_SIZE` first if the frozen TrOCR pass runs out of GPU memory.

For Slurm, use your site's actual partition, account and GPU resource flags.
After exporting these variables and creating `thesis/logs`, a submission pattern is:

```bash
sbatch --job-name=trocr-refiner --gres=gpu:1 --cpus-per-task=4 \
  --mem=24G --time=08:00:00 --output=thesis/logs/refiner-%j.log \
  thesis/scripts/run_trocr_diffusion_refiner.sh
```

The resource values are an initial request, not measured requirements. Add the
appropriate partition/account and ensure the activated environment is available
to the batch job (or export `PYTHON` as its absolute executable path). Never run
GPU training on a login node. If time is short, run a separate small pilot with
`EPOCHS=2` before allocating a full run; do not use its test results for tuning.

## Separate steps and recovery

All commands expose `--help`. For example, for a pinned external base:

```bash
python -m thesis.refiner cache --dataset-root "$DATASET_ROOT" \
  --split-root "$SPLIT_ROOT" --cache-root "$CACHE_ROOT" \
  --external-base --partition train validation --device cuda --resume

python -m thesis.refiner train --dataset-root "$DATASET_ROOT" \
  --split-root "$SPLIT_ROOT" --cache-root "$CACHE_ROOT" \
  --output-dir "$RUN_ROOT" --epochs 30 --device cuda

python -m thesis.refiner cache --dataset-root "$DATASET_ROOT" \
  --split-root "$SPLIT_ROOT" --cache-root "$CACHE_ROOT" \
  --external-base --partition test --device cuda --resume

python -m thesis.refiner evaluate --dataset-root "$DATASET_ROOT" \
  --split-root "$SPLIT_ROOT" --cache-root "$CACHE_ROOT" \
  --run-dir "$RUN_ROOT" --output-dir "${RUN_ROOT}_test" --device cuda
```

An interrupted cache can resume; saved tensors are hashed and checked. Training
saves `last.pt` and `best.pt`, but **optimizer resume is not implemented** in this
first version. An interrupted training job must restart in a fresh run directory
using the verified cache. A completed training run does not need retraining if
only test caching/evaluation failed: run the last two commands separately.

Each feature cache is approximately 0.9 MB per line for a standard 384x384 TrOCR
encoder, plus filesystem overhead; allow several GB for thousands of lines.
Changing preprocessing or augmentation requires a new cache. No new image
augmentation is applied during cached training.

## Outputs

`RUN_ROOT` contains model configuration, tokenizer, `best.pt`, `last.pt`,
per-epoch losses/validation metrics, `best_validation.json`, runtime/source
provenance and `selection.json`. `completed.json` records artifact hashes and
is written only after successful training completion.

`${RUN_ROOT}_test/comparison.md` is the readable accuracy table;
`evaluation.json` contains aggregate CER/WER, exact lines, corrected/damaged
reference-character counts, capacity/unknown-character diagnostics and timings.
`predictions.jsonl` retains each literal reference and each model's prediction.
Test labels that exceed capacity or contain unseen characters are still scored
and flagged, not silently truncated or removed. Inference failures stop the run
rather than producing a report that silently excludes failed lines.

Reported base timings come from feature generation and refiner timings exclude
cache reads. They are not live end-to-end production latency. The default sampler
is seeded; identical results across GPU architectures/package versions are not
promised. The evaluation uses the sampling settings saved during validation.

Weights, caches, datasets and generated reports stay local and are not published
by these commands. The GitHub contribution contains source, tests and this guide.
