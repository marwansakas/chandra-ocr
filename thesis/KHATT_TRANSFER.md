# KHATT pretraining, then children-only MDiff fine-tuning

This portable Python workflow reuses `thesis.mdiff_arabic_v1.model.Recognizer`
with `arm='mdiff'`. It does not edit or resume Ammar's completed experiment
packages. This is a new two-stage experiment, not his joint KHATT/children AR
experiment or a claim of improved recognition accuracy. Offline synthetic tests
verify training, inference and recovery; real KHATT/children accuracy must be
measured on the cluster.

The sequence is **KHATT-only training -> validation-selected KHATT weights ->
children-only training with a fresh optimizer -> explicit children test**.
Both stages use structured masked reconstruction and visible-token correction,
with the existing 0.5/0.5 losses plus 0.1 tail-EOS loss. Inference uses three
denoise/correction rounds (six decoder passes), without target tokens/length.

## Slurm submission and recovery after a terminated preparation

`Terminated` during preparation means the process stopped before the completion
marker was written. A subsequent missing `prepared.json` is a consequence, not
a model-training error. The termination message alone cannot distinguish a login
node policy, wall-time/resource limit, manual signal or another cause. Do not
run the GPU phases directly on a login node.

From `/home/sakas/chandra-ocr`, after updating Git and setting up the environment:

```bash
if PREP_JOB=$(sbatch --parsable thesis/scripts/slurm_prepare_khatt.sbatch); then
  PREP_JOB=${PREP_JOB%%;*}
  sbatch --dependency="afterok:$PREP_JOB" \
    thesis/scripts/slurm_khatt_gpu.sbatch pretrain
fi

squeue -u "$USER"
```

The supplied preparation script requests 4 CPUs, 16 GB RAM and six hours; the
GPU script requests one GPU, 4 CPUs, 24 GB host RAM and 24 hours. These are
starting resource requests, not measured requirements for your cluster. Add
your university's actual `--partition`/`--account` options to each `sbatch`
command if required, and choose an internet-enabled preparation queue/node.
The GPU job is released only when preparation exits successfully.

Logs are `khatt-prepare-JOBID.log` and `khatt-mdiff-JOBID.log` in the repository
root. Check a preparation job with:

```bash
tail -n 50 "khatt-prepare-$PREP_JOB.log"
sacct -j "$PREP_JOB" --format=JobID,State,ExitCode,MaxRSS,Elapsed
```

If preparation is interrupted, submit the same preparation script again. It
reuses verified encoder assets, pins the original dataset revision and resumes
its committed image journal. A legacy incomplete directory containing only the
old `images/` cache (or an empty directory) is adopted without deleting it;
cached images are checked against the source before reuse. No manual recursive
delete or new artifact root is necessary for that recovery. Training has not
started when `prepared.json` is missing, so its first submission uses `pretrain`
without `--resume`. Only an interrupted training run with `last.pt` uses:

```bash
sbatch thesis/scripts/slurm_khatt_gpu.sbatch pretrain --resume
```

For later children training after its preparation has completed:

```bash
sbatch thesis/scripts/slurm_khatt_gpu.sbatch finetune
```

## Short cluster commands

After updating Git and setting up the environment described below, use one
explicit phase per command:

```bash
bash thesis/scripts/run_khatt_transfer.sh prepare-khatt
# GPU allocation required for the next command.
bash thesis/scripts/run_khatt_transfer.sh pretrain

# Wait for pretraining to complete before preparing the children stage.
bash thesis/scripts/run_line_experiments.sh download
bash thesis/scripts/run_khatt_transfer.sh prepare-children
# GPU allocation required.
bash thesis/scripts/run_khatt_transfer.sh finetune

# Only after final validation decisions; GPU allocation required.
bash thesis/scripts/run_khatt_transfer.sh test
```

If interrupted, repeat `pretrain --resume` or `finetune --resume`. Downloads and
preparation are not automatically chained into GPU training. Repeating a
completed preparation verifies its snapshot; it does not resplit the data.

Wrapper defaults match the detailed commands below. `TRAIN_PYTHON` can select an
explicit interpreter; `KHATT_TRANSFER_ROOT` selects another artifact root.
`KHATT_HF_ROOT` reuses earlier saved HF splits. `MDIFF_CAPACITY` and
`MDIFF_MAX_WIDTH` configure the initial KHATT model and must stay fixed on resume.
For children, `CHILDREN_DATA_ROOT`, `CHILDREN_WRITER_MAP`,
`CHILDREN_SPLIT_STRATEGY` and `CHILDREN_ASSIGNMENTS` select data/grouping/saved
assignments before first preparation. Changing them after preparation does not
replace an existing snapshot; choose a new artifact root for a new study.

## 1. Update the cluster checkout and environment

```bash
cd /home/sakas/chandra-ocr
git switch main
git pull --ff-only origin main

# Use your cluster-supported CUDA/PyTorch environment first.
python -m venv --system-site-packages .venv-khatt-transfer
source .venv-khatt-transfer/bin/activate
python -m pip install -r requirements-khatt-transfer.txt
```

Python 3.10+ and PyTorch >=2.6,<3 are required. Install the PyTorch build appropriate
for your university GPU/driver before the requirements if necessary; the example
venv can reuse an existing cluster PyTorch installation. NumPy, Pillow, datasets
and huggingface-hub are the remaining runtime dependencies. Transformers, Paddle,
PowerShell, FlashAttention and custom remote Python code are not needed.
The optional pytest dependency comes from the existing diffusion requirements.

Run downloads on an allowed internet-enabled node. GPU training must run within
your university's allocation/batch-job rules. No partition/account is guessed
here. Inside the allocation, verify the active interpreter sees your GPU:

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'
```

CUDA uses BF16 on supported GPUs and FP32 otherwise; CPU uses FP32 only. The
observed precision and relevant package/device versions are recorded. If a long
line exhausts GPU memory, batch size 1 is already the default; use a GPU with
more memory. Images are not silently squeezed, cropped or dropped.

## 2. Download and freeze KHATT

```bash
python -m thesis.khatt_transfer setup-assets \
  --output thesis/transfer_artifacts/encoder

python -m thesis.khatt_transfer prepare-khatt \
  --output thesis/transfer_artifacts/khatt_data

python -m thesis.khatt_transfer check \
  --prepared thesis/transfer_artifacts/khatt_data \
  --capacity 256 --max-width 4096
```

The visual assets are the existing official MDiff4STR-small encoder, pinned at
Hugging Face revision `70c320bb8ab539ecc08f820ef94ba7679b6e40f2`, with SHA-256
verification. Only visual weights transfer; the Arabic MDiff head starts fresh.
Upstream source attribution remains in `thesis/mdiff_arabic_v1/vendor/NOTICE.md`.

KHATT comes from
<https://huggingface.co/datasets/johnlockejrr/KHATT_v1.0_dataset>.
Preparation resolves and records an immutable dataset revision, then reads only
its supplied `train` and `validation` splits. `--revision COMMIT_SHA` can select
an explicit revision. This is **not** Ammar's Kaggle mirror or its 9,497-row
training set. No fixed row count from that different mirror is assumed.

This HF mirror documents intentionally reversed transcriptions for PyLaia. Its
adapter reverses those stored code points exactly once to restore logical order,
records the original mirror string, and preserves image orientation. The only
derived-target normalization is `no_harakat_v1`, removing U+064B..U+0652.
Spelling, tatweel, other combining marks and literal whitespace remain.

If the earlier download commands have already saved both HF splits under
`thesis/datasets/khatt_hf_train` and `khatt_hf_validation`, replace the
`prepare-khatt` command with:

```bash
python -m thesis.khatt_transfer prepare-khatt \
  --hf-root thesis/datasets \
  --output thesis/transfer_artifacts/khatt_data
```

The saved-split mode records local provenance and freezes every image/text hash;
it cannot reconstruct a remote revision absent from those older downloads.
Neither mode requests KHATT test samples. Preparation writes atomic images and
a per-split journal, flushing each committed sample. Repeat the same command
after interruption: completed journal entries are verified and skipped, and an
incomplete final journal append is recovered. Changing the source revision or
inventory is rejected. An incomplete snapshot cannot be trained. A completed
snapshot is verified rather than overwritten. Completed snapshots can move
between machines because image paths are relative, not bound to Windows paths.

Vocabulary comes from KHATT TRAIN plus the declared Arabic/Latin/digit/punctuation
inventory and CR/LF/TAB/inequality. Validation/test never enlarge it. If an
additional character inventory is known in advance, use `--inventory-file FILE`
when preparing KHATT. Unknown characters in later children TRAIN/validation
cause an error; adding characters requires a deliberately new pretraining run,
not changing a trained output head's token IDs.

The example declares capacity 256 (including EOS) and aspect-preserving RGB
height 32, width limit 4096, stride alignment 4. Preflight reports overlong or
oversized samples before training. Increase these limits before KHATT training
if needed; do not truncate targets. Capacity, model dimensions and tokenizer
remain unchanged for children fine-tuning. The new defaults use no image
augmentation; this is not a reproduction of Ammar's augmentation/budget sweep.

## 3. Pretrain on KHATT only

Inside an allocated GPU job, after activating the environment:

```bash
cd /home/sakas/chandra-ocr
source .venv-khatt-transfer/bin/activate

python -m thesis.khatt_transfer train \
  --prepared thesis/transfer_artifacts/khatt_data \
  --output thesis/transfer_artifacts/khatt_run \
  --encoder-checkpoint thesis/transfer_artifacts/encoder/best.pth \
  --capacity 256 --max-width 4096 \
  --epochs 100 --patience 10 \
  --batch-size 1 --accumulation 16 \
  --device cuda
```

The full KHATT training split is visited once per epoch. Defaults are seed
54101, decoder LR 3e-4, encoder LR 1e-5, AdamW weight decay .01, clip 1, five
warmup epochs, encoder frozen for five epochs, and cosine decay to 0.1 of the
initial LR. BatchNorm running statistics stay frozen, as in the reused model.
One hundred epochs is a maximum, not a measured optimal budget.

`best.pt` minimizes validation character edits, then word edits, then maximizes
exact lines; ties retain the earlier epoch. Stage epoch 0 is eligible, preventing
a training stage from automatically replacing its starting weights with worse
validation results. CER/WER are corpus edit rates and can exceed 100%.
Missing-EOS lines remain in all metrics and are counted explicitly. There is no
test evaluation or automatic model promotion during training.

## 4. Download children data and prepare a new children-only split

```bash
bash thesis/scripts/run_line_experiments.sh download

python -m thesis.khatt_transfer prepare-children \
  --dataset-root thesis/datasets/ground_truth_lines \
  --init-from thesis/transfer_artifacts/khatt_run/best.pt \
  --split-strategy notebook \
  --output thesis/transfer_artifacts/children_data

python -m thesis.khatt_transfer check \
  --prepared thesis/transfer_artifacts/children_data \
  --init-from thesis/transfer_artifacts/khatt_run/best.pt
```

The download uses your configured `gdrive:` rclone remote and adds missing
folders only; it does not refresh labels inside existing folders.

Preparation copies image/text pairs, leaving original data intact. Blank labels,
exact `ignore` and standalone `≠` markers are listed in `exclusions.json` and
excluded. Other literal text is retained. Missing labels or conflicting duplicate
images cause errors. Exact pixel duplicates and requested writer/notebook groups
are joined before assigning train/validation/test. Duplicate images cannot cross
splits. Defaults target 70/10/20 by lines, subject to whole-group sizes.

`notebook` derives a notebook proxy from the existing `_Page_###` folder names;
it does not verify child identity across notebooks. For authoritative child IDs,
replace `--split-strategy notebook` with `--writer-map /path/to/writers.json`, a
JSON object mapping relative sample-folder IDs to child IDs. If neither is
available, explicitly choose `--split-strategy line`; this may share children
across partitions and is not unseen-writer evaluation.

By default this creates a **new split**, not Ammar's historical 2,103/150/752
assignment or the legacy TrOCR split. To preserve an existing three-way split,
add `--assignments /path/to/assignments.json`, a JSON object mapping **every
usable sample ID** to `train`, `validation` or `test`. The supplied membership
is retained exactly; inconsistent writer/notebook/duplicate grouping is rejected.
The selected `--split-strategy` or writer map still controls group verification.
For example, an existing line split needs explicit `--split-strategy line` and
cannot be called writer-disjoint. A two-way split needs a validation subset
carved from TRAIN, preserving all previously reserved TEST assignments.

Previously consulted data are not made untouched by reshuffling them. Do not
move a separately reserved existing test set into training: supply its reviewed
three-way assignments, or use new study data outside that reserved set.
No writer-disjoint generalization or comparable historical score is claimed.

Only the parent checkpoint's tokenizer is accepted. Children training/validation
must fit that vocabulary, capacity and image policy. Children test references
do not control vocabulary/capacity or model selection. Training does not open
`test.jsonl`; final evaluation retains unknown/overlong references in scoring.
Cross-stage image overlap with the parent training/validation snapshot is rejected.

## 5. Fine-tune on children only

Inside an allocated GPU job:

```bash
python -m thesis.khatt_transfer train \
  --prepared thesis/transfer_artifacts/children_data \
  --init-from thesis/transfer_artifacts/khatt_run/best.pt \
  --output thesis/transfer_artifacts/children_run \
  --epochs 30 --patience 7 --warmup-epochs 1 \
  --decoder-lr 3e-5 --encoder-lr 1e-6 \
  --batch-size 1 --accumulation 16 \
  --device cuda
```

All model weights transfer strictly. The optimizer, stage epoch and early-stop
counter reset. Children fine-tuning defaults to no frozen encoder epochs.
The stage's starting model is evaluated on children validation; no KHATT score
is reused for children checkpoint selection. Parent hashes, tokenizer, data,
configuration, source bytes and relevant environment are recorded. A new stage
can run on a different machine with its own provenance; it does not claim
bitwise equivalence across platforms.

## Resume and outputs

If a job is interrupted, rerun its exact training command with `--resume` added,
using the same output directory and settings. Do not rerun preparation. For
example, repeat the children command above with its same arguments plus
`--resume`. Extending the maximum epochs or changing LR/precision/environment
requires a new stage/run, not a silent resume.

`last.pt` contains model, optimizer, RNG, best selection, early stopping and
history. An interrupted incomplete training epoch replays from the last
committed epoch. `trained.pt` commits a completed epoch before validation;
if validation fails, resume evaluates those weights without repeating training.
Best and last checkpoints are atomically published. The OS lock prevents two
writers to one run. Completed/early-stopped runs cannot extend on resume.
Only best, latest resume, and transient pending-epoch model/optimizer copies
are kept; per-epoch validation reports remain. Allow storage for several
hundred MB per run plus data snapshots. Runs and private datasets/reports are
ignored by Git under `thesis/transfer_artifacts/`.

Main outputs are `run.json`, `best.pt`, `best.json`, `last.pt`, `history.json`,
`validation-####.json` and `complete.json`. Model checkpoints include their
tokenizer/config and cross-stage lineage. The command never pushes private
data, per-line reports or trained weights to GitHub.

## Explicit final children test

After freezing decisions on validation, run once inside a GPU allocation:

```bash
python -m thesis.khatt_transfer evaluate \
  --prepared thesis/transfer_artifacts/children_data \
  --checkpoint thesis/transfer_artifacts/children_run/best.pt \
  --partition test \
  --output thesis/transfer_artifacts/children_test.json \
  --device cuda
```

The report includes CER, WER, exact lines, substitutions/insertions/deletions,
complete sample denominators, missing-EOS counts and literal per-line outputs.
Inference errors stop instead of silently omitting samples. Existing reports
are never overwritten. This evaluation applies `no_harakat_v1` consistently;
it is not interchangeable with the legacy `arabic-light` scoring policy.

## Offline verification

```bash
python -m pytest -q tests/test_khatt_transfer.py
```

These tests use generated images, local random weights and mocked HF downloads.
They do not pretrain on real KHATT, fine-tune on real children's data, download
official weights or establish GPU throughput/accuracy.
