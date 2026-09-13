# Arabic line diffusion prototype

Implemented 6 September 2026, starting from repository revision
`546659f267ec6f17b2da31659f96b65fc9a3e5e9`. This is a functioning, independently
implemented **image-conditioned absorbing-mask baseline**, with a separate
autoregressive control. It is not a reproduction of the complete MDiff4STR
training/correction system. Local synthetic checks establish code behavior,
not Arabic handwriting accuracy. The original implementation was developed without
downloading a real training set or pretrained weights. The user subsequently
authorized a local first-100-page Drive pilot; see the separate workflow below.

## First-100-page line pilot

The user requested a separate 90/10 split of individual labeled lines from the
first 100 Surya pages, using only the corresponding clean `ground_truth_lines`
image/text pairs. `thesis/scripts/prepare_diffusion_pilot.py` verifies the local
Drive inventory and freezes this explicit `line_pilot` mode. It does not require
writer IDs, and pages/children can occur in both train and test. The original
three-way writer-isolated workflow remains unchanged.

Pilot training uses fixed epochs with no validation subset or test-based
checkpoint selection. `last.json` points to the final completed epoch. Ordinary
validation runs preserve `best.json` for the absolute lowest strict validation
CER and `best_exact.json` for the most exact validation lines among checkpoints
within 0.5 absolute CER percentage points of that minimum. Ties use lower CER,
then the earlier epoch. The test partition cannot expand the training vocabulary
or output capacity.

The downloaded pilot, PowerShell wrapper, pairing CSV, per-page counts, and
instructions are under `../Diffusion/drive_first100`. Read its
[README](../../Diffusion/drive_first100/README.md) for the train/test/stat commands.
`thesis/scripts/report_diffusion_stats.py` displays saved data, training, and
evaluation statistics.

## Reference mapping and deliberate departures

The recognition reference remains OpenOCR/MDiff4STR at
`0d522801ec6dc1df852c6b6d4ed6a08f5127ed97`.

| Paper component | Pinned implementation inspected | Consequence for this prototype |
|---|---|---|
| §3.1 visual-conditioned token decoder | `MDiffDecoder`, token self-attention plus visual cross-attention | Same conditioning pattern; independent PyTorch implementation |
| §3.2 structured mask strategies | `MDiffLabelEncode.forward_process`: full, forward, backward, block, cloze and random branches | Replaced by independent absorbing corruption for an explicitly derived baseline |
| §3.3 replacement correction | `reflect_random_idices` corrupts tokens by replacement; `forward_reflect` exposes a correction pass | Deferred until mask-only real-data validation; no claimed ability to revise committed letters yet |
| §3.4 objectives | `forward_train` weights masked CE and adds all-valid-token reflection CE | Baseline uses the discrete-time ELBO estimator below, without reflection loss |

The label encoder saves `p_mask` as the **realized masked fraction**, forces at
least one mask in its random branch, and masks inputs after EOS while excluding
PAD targets. These details matter when distinguishing its training recipe from
independent Bernoulli corruption. See the pinned
[label encoder](https://github.com/Topdu/OpenOCR/blob/0d522801ec6dc1df852c6b6d4ed6a08f5127ed97/openrec/preprocess/mdiff_label_encode.py).

The decoder divides masked CE by `p_mask` and then `length + 1`; the encoder's
saved `length` already includes EOS. `MDiffLoss` simply exposes the returned
loss. The supplied configuration enables three corruption samples, but sets
every alternative decoding flag false. `forward()` therefore falls back to one
parallel pass; `sampler_step` alone does not enable iterative correction.
Low-confidence decoding and reflection exist as separate methods. No claim is
made that this configuration reproduces all paper experiments. See the pinned
[decoder](https://github.com/Topdu/OpenOCR/blob/0d522801ec6dc1df852c6b6d4ed6a08f5127ed97/openrec/modeling/decoders/mdiff_decoder.py),
[configuration](https://github.com/Topdu/OpenOCR/blob/0d522801ec6dc1df852c6b6d4ed6a08f5127ed97/configs/rec/mdiff4str/svtrv2_mdiffdecoder_small.yml), and
[paper §§3.1–3.4](https://arxiv.org/html/2512.01422v1).

No upstream implementation is copied into the Python package. The adjacent
Diffusion research folder retains the inspected source excerpts/files with
OpenOCR's Apache-2.0 license for reference.

## Exact baseline method

Let `C` be the saved capacity, including EOS. Encode a literal line into code
points, append EOS, and repeat EOS through slot `C`. This defines a fixed-size
clean sequence without unknown-length padding. The first predicted EOS ends the
transcription. Repeated EOS positions participate in training; long capacities
can therefore emphasize EOS prediction. This is an explicit experimental
choice to evaluate, not an established optimal length model.

For time `0 <= t <= 1`, independently at each slot:

```text
alpha(t) = 1 - t
q(x_t | x_0) = (1-t) delta(x_0) + t delta(MASK)
```

For `s < t`, forward transitions keep MASK absorbing and keep a clean token with
probability `(1-t)/(1-s)`. They induce the marginals above. The learned reverse
kernel carries any revealed token unchanged. At a masked slot it leaves MASK
with probability `s/t`, otherwise samples a clean token from the image- and
text-conditioned decoder with probability `(t-s)/t`. MASK and PAD have zero
output probability. This follows the absorbing reverse parameterization in
[MDLM §3.2.3](https://arxiv.org/html/2406.07524v2).

Training samples `k` uniformly from `1..T` (default `T=256`), sets `t=k/T`, then
samples the forward corruption. For adjacent reverse times the reveal
probability is `1/k`. The resulting discrete negative-ELBO estimator is:

```text
mean_over_batch [ sum_over_masked_valid_slots CE(x_0, prediction) / (t * N_valid) ]
```

The `T/k = 1/t` factor accounts for uniform timestep sampling. Under the normal
fixed-EOS representation, `N_valid=C`, so this is the sequence objective scaled
by a fixed constant. Empty masks contribute zero and are not resampled. The
prior at `t=1` is deterministically all MASK, independent of image labels or
length. PAD, if explicitly supplied by callers, is excluded from corruption
and loss and replaced by MASK before decoding; it is never an attention hint.

Inference caches visual features and walks a uniform time grid from 1 to 0.
The last step reveals every remaining mask. It uses categorical sampling, not
confidence ordering. Changing inference steps changes the learned-kernel
approximation; it does not make samples exact draws from the unknown data
posterior. There is **no remasking, replacement correction, spellchecker,
segmentation dependency, ground-truth text, or ground-truth length input** in
this baseline. It preserves misspellings in the targets but can still make
transcription errors after training.

## Data and Unicode contract

Supported layouts are `<sample_id>/image.png` plus `ground_truth.txt`, or
`<page>/line_*/image.png` plus its label. Every usable sample needs a reviewed
literal UTF-8 label. Human review status cannot be established by a nonempty
file. The audit reports records with nonempty labels, not a verified-label
count. It also reports missing/unreadable files, labels containing `ignore`,
characters, code-point lengths, image dimensions and exact RGB pixel duplicates.

Targets undergo **no Unicode normalization**. Arabic logical order, combining
marks, presentation forms, digits, punctuation, tatweel, hamza variants and edge
spaces survive. Label reading removes at most one terminal LF or CRLF; embedded
line breaks are preserved for curated crops containing stacked writing. Each
image/text folder still counts as one sample. Tokenizer policy version 2 records
this behavior; early synthetic checkpoints using policy version 1 must be rebuilt.
Vocabulary is training characters plus the declared
inventory in `tokenizer.py` and an optional inventory file. OOV characters and
length overflow are explicit errors in train/validation. No unknown token or
silent truncation is used.

`--writer-map` is a JSON object mapping each sample ID (including `/line_*` for
nested layouts) to an authoritative child ID:

```json
{
  "page_a/line_001": "child_17",
  "page_b/line_004": "child_17",
  "page_c/line_003": "child_23"
}
```

The mapping must join the same child across pages **and notebooks**. Filename
and metadata student IDs are audit hints only. Writer groups connected by exact
duplicate images are joined before splitting. Conflicting labels on identical
images block freezing. Ratios apply to writer/duplicate groups, so line counts
need not match the requested ratios. At least three independent groups are
required. Near duplicates and incorrect human writer mappings still need review.

Freezing writes train/validation/test JSONL, a tokenizer, full audit and a lock
with SHA-256 hashes. Paths use `/` relative to `--dataset-root`, with traversal
rejected. This differs from the legacy common manifest reader, which resolves
paths against the repository. Use these new CLIs for new manifests; sample IDs
and prediction directory conventions remain compatible with existing outputs.
Loading verifies manifests and content; training and evaluation verify images,
labels and metadata against the freeze. Inference never opens label files.

Capacity defaults to max train/validation code-point length plus EOS; **test
labels do not select vocabulary or capacity**. Evaluation reports test OOV and
overflow cases without dropping them. Use a new split directory when changing
policy, labels, mappings or vocabulary. Freeze directories and run directories
are never overwritten. Audit exclusions require `--allow-exclusions` only after
their recorded reasons have been reviewed.

## Images, encoders and checkpointing

Images retain aspect ratio at the configured height, use RGB/255 and saved
channel normalization, and pad on the right to patch-aligned widths. Oversized
lines are rejected; there is no squeeze-to-square, cropping, binarization or
dot-removal step. There is currently no width-bucket sampler; lower the batch
size when mixed widths consume too much memory.

The default encoder is a small patch transformer initialized from scratch.
Spatial sinusoidal coordinates are independent of batch padding. Encoder
self-attention and decoder cross-attention mask padded patches.

The optional `--vit-checkpoint` loads an **existing local ViTModel directory**
and its image processor, with `local_files_only=True`. Each line is encoded at
its own patch-aligned width because ViT has no image-padding attention-mask
argument. Rectangular position embeddings are interpolated; only the resulting
feature batch is padded and masked. This adapter is intentionally simple and
can be slower than a bucketed encoder. ViT config and normalization are saved
inside our checkpoint, so loading our trained checkpoint does not require the
original pretrained directory or network access. OpenOCR SVTR weights and full
TrOCR/DeiT checkpoints are not directly compatible with this adapter.

Only ViT visual tensors load from that source. The Arabic vocabulary embedding
and output head are newly initialized. Missing/mismatched pretrained weights
fail, and loaded tensor names and source file hashes are recorded. Checkpoint
hashes protect configuration, tokenizer order and model state; loading uses
strict state shapes. A reordered same-size vocabulary cannot silently reuse a
head. Test coverage uses tiny random ViT weights, not an Arabic pretrained model.

Saved checkpoints contain model/config/tokenizer, optimizer, epoch and Python/
NumPy/torch/CUDA RNG state. Runs retain frozen manifests, source copies/hashes, source
revision/dirty status, versions, preprocessing, parameter count, precision,
sampling settings, validation predictions and traces. Validation strict CER
selects `best.json`. A nondominated checkpoint frontier preserves every model
that could later win the 0.5-point exact-line near-tie rule without saving every
epoch. Its selected pointer is `best_exact.json`, and its complete selection
state is `checkpoint_selection.json`. Resume is at an epoch boundary into a
**new run directory**; it restores optimizer/RNG and retains the applicable
earlier selection checkpoints. It rejects
changed splits or core training settings. Architecture and preprocessing come
from the saved checkpoint when resuming. Cross-device/version bitwise equality
is not guaranteed. Disk usage grows by one checkpoint and validation trace set
per epoch; no checkpoints are automatically deleted.

`--initialize-from CHECKPOINT` is a separate fine-tuning operation for an
ordinary train/validation split. It verifies the checkpoint against the current
frozen data/tokenizer, loads its model weights and image/model configuration,
then resets the optimizer and stage epoch count in a new run directory. The
source checkpoint is copied as stage epoch 0 and remains eligible for both
recognition pointers, so a worse fine-tuning stage cannot replace it. Run
metadata records the parent checkpoint path, epoch and model hash. It cannot be
combined with `--resume`, a new ViT checkpoint, line-pilot mode or overfit mode.

## Local correctness checks

Requires Python 3.10+ and the isolated dependencies. These commands are real
CLIs; `--help` is available without model loading.

```bash
python -m pip install -r requirements-diffusion.txt
python -m pytest tests -q
python thesis/scripts/smoke_diffusion_line_ocr.py --output-dir /scratch/diffusion-smoke
```

To include the optional ViT adapter test:

```bash
python -m pip install -r requirements-diffusion-vit.txt
python -m pytest tests/test_diffusion_ocr.py -q
```

The smoke command creates generated image patterns and arbitrary Arabic labels,
freezes fixture splits, trains one tiny CPU epoch, reloads, runs inference and
scores saved predictions. Its CER is not a handwriting result. Tests cover
noise endpoints/weighting/reverse transitions, PAD, Unicode, overflow, leakage,
mixed widths, EOS/no-EOS, gradients, deterministic samples, target-free runner
behavior, literal output spaces, vocabulary integrity, epoch resume and the
causal autoregressive control.

## Server workflow

Run from the repository root on the university machine. Choose the installed
CUDA-compatible PyTorch build there; this document assumes no GPU/VRAM budget.
These path variables are placeholders for server locations you supply.

```bash
python -m venv .venv-diffusion
source .venv-diffusion/bin/activate
python -m pip install -r requirements-diffusion.txt
python -m pip freeze > /scratch/diffusion-environment.txt

DATA_ROOT=/path/to/verified/ground_truth_lines
WRITER_MAP=/path/to/reviewed_writer_map.json
SPLIT_ROOT=/scratch/arabic_diffusion_split_v1
RUN_ROOT=/scratch/arabic_diffusion_run_v1

python thesis/scripts/audit_diffusion_dataset.py \
  --dataset-root "$DATA_ROOT" --writer-map "$WRITER_MAP" \
  --report /scratch/diffusion-audit-v1.json

python thesis/scripts/audit_diffusion_dataset.py \
  --dataset-root "$DATA_ROOT" --writer-map "$WRITER_MAP" \
  --freeze "$SPLIT_ROOT" --seed 42

python thesis/scripts/audit_diffusion_dataset.py \
  --dataset-root "$DATA_ROOT" --verify "$SPLIT_ROOT"

python thesis/scripts/train_diffusion_line_ocr_model.py \
  --dataset-root "$DATA_ROOT" --split-root "$SPLIT_ROOT" \
  --output-dir "$RUN_ROOT" --check-only --device cpu
```

Use audit train/validation dimensions to select height and max width, then keep
them fixed for comparisons. Default sizes are implementation defaults, not a
recommended university GPU allocation. If width preflight fails, explicitly
increase `--max-width` (a patch-size multiple) or review the crop. If validation
has an unknown character, review/add it to a declared inventory and freeze a new
version. Do not inspect test labels to choose a capacity or alphabet.

First attempt a tiny **training-subset overfit diagnostic**. `--overfit-samples`
uses only the first N frozen training records for both fit and diagnostic
evaluation; its score must never be presented as validation or test accuracy.
It deliberately does not run held-out evaluation.

```bash
python thesis/scripts/train_diffusion_line_ocr_model.py \
  --dataset-root "$DATA_ROOT" --split-root "$SPLIT_ROOT" \
  --output-dir /scratch/arabic_diffusion_overfit_v1 \
  --overfit-samples 8 --epochs 100 --batch-size 2 --device cuda
```

Inspect the saved transcriptions, EOS outcomes and traces. Overfit success is a
diagnostic to measure, not an assumed outcome. Only after that succeeds, run
validation experiments:

```bash
python thesis/scripts/train_diffusion_line_ocr_model.py \
  --dataset-root "$DATA_ROOT" --split-root "$SPLIT_ROOT" \
  --output-dir "$RUN_ROOT" --epochs 20 --batch-size 4 --device cuda

# Continue an existing run at an epoch boundary into a new directory.
# Keep original batch size, seed, learning rate and sampling settings.
python thesis/scripts/train_diffusion_line_ocr_model.py \
  --dataset-root "$DATA_ROOT" --split-root "$SPLIT_ROOT" \
  --resume "$RUN_ROOT/checkpoint-0020" \
  --output-dir /scratch/arabic_diffusion_resume_v1 \
  --epochs 40 --batch-size 4 --device cuda
```

To start from a chosen **locally staged ViT visual checkpoint**, install
`requirements-diffusion-vit.txt` and add
`--vit-checkpoint /path/to/local/vit` to the initial training command. Use the same
checkpoint for the autoregressive control. No ready-made Arabic weights are
assumed. This option must point to a ViTModel with `config.json`, saved weights,
and `preprocessor_config.json`; inspect its pretraining history separately.

Run the matched control with the same flags and seed, changing only its output
directory and adding `--decoder-kind autoregressive`. It uses the same encoder,
tokenizer, image policy, capacity and decoder dimensions, with causal attention
and shifted teacher forcing. MASK serves only as a start input. Its inference
uses greedy sequential decoding, stops at EOS, and records actual decoder calls;
`--steps` does not control the autoregressive loop. Training budgets and
initialization/pretraining must also match to interpret decoder differences.

Freeze sampling settings on validation, then make final held-out predictions:

```bash
python pipelines/diffusion_line_run.py \
  --model "$RUN_ROOT" --input-root "$DATA_ROOT" --split-root "$SPLIT_ROOT" \
  --partition test --output-root /scratch/arabic_diffusion_test_v1 \
  --steps 16 --seed 42 --trace --device cuda

python thesis/scripts/evaluate_diffusion_line_ocr.py \
  --dataset-root "$DATA_ROOT" --split-root "$SPLIT_ROOT" --partition test \
  --predictions diffusion=/scratch/arabic_diffusion_test_v1 \
  --predictions trocr=/path/to/trocr/predictions \
  --predictions matched_ar=/path/to/matched_ar/predictions \
  --output-dir /scratch/arabic_diffusion_comparison_v1
```

Existing TrOCR predictions can be read by sample ID when they correspond to the
same crops and literal ground truth. **They are only a held-out baseline if their
model was trained without the frozen test writers.** The legacy trainer's own
train/test split is not evidence of that. Check training manifests or train an
appropriately isolated TrOCR baseline before interpreting the comparison.

Outputs follow the common `prediction.txt`, `metadata.json`, `predictions.jsonl`
and `run_summary.json` convention. The new runner records path bases explicitly
and restores literal edge spaces after calling the common writer. Each selected
sample has a completed/failed record. Failures return exit code 1 after saving
the run report. No-EOS predictions remain visible with a flag and are scored.

Evaluation reports corpus strict/normalized CER and WER, exact-line accuracy,
substitution/insertion/deletion and space-error counts, OOV/overflow and every
missing/failed sample. Strict WER splits on whitespace and retains digits and
punctuation; historical WER retains the existing digit-excluding tokenizer.
Missing/failed predictions count as empty in the primary all-expected metrics;
completed-only metrics are separate. `--before-root` compares earlier complete
predictions to measure aligned reference-character fixes/damage, with a saved
alignment tie policy. It does not interpret partially masked traces as complete
transcriptions.

Inference records batch-one latency including image read/preprocess, transfer,
encoder and decoder calls, excluding checkpoint load/output writes. CUDA peak
allocated memory is recorded; CPU peak memory is unavailable and explicitly
null. Validation traces and literal misspelling examples support later analysis.
No replacement-correction or runtime/accuracy advantage is claimed. Real-data
overfit, validation selection, visual-dependence diagnostics, writer-reviewed
misspelling analysis, multiple seeds and final fair comparisons remain server
experiments to perform before Stage 4 refinements.

## Local controlled fitting experiments (September 2026)

The first-100-page pilot has now been trained and evaluated locally. Its high
training CER and failed sixteen-crop memorization check motivate the controlled
suite in `../Diffusion/drive_first100/ABLATION16_README.md`. Use that guide for
current local commands; the server roadmap above is historical.

The trainer supports `--eos-target single` to supervise characters plus one EOS,
with loss-excluded PAD converted to MASK before decoder attention. The default
`repeat` preserves legacy behavior. `--full-mask-probability 0.25` mixes the
original uniform-time diffusion objective with full-mask cross-entropy. This
mixture is explicitly a different objective, not an unchanged ELBO estimator.

`--text-policy no_harakat_v1` derives targets by removing U+064B..U+0652 and
excludes those output IDs. It preserves the original ground truth and records
both raw and derived targets/scores. The checkpoint's explicit target policy is
authoritative for the experiment; the saved tokenizer retains the original
inventory/file policy for compatibility. New options default to literal targets,
repeated EOS and zero full-mask mixture probability when loading old checkpoints.

`--ink-mode dominant_soft` is an optional conservative color ablation. Mixed
color inputs fall back to original RGB, and minority ink is only de-emphasized,
never completely removed. Input policy is saved in the checkpoint and reused by
inference. It is not a handwriting/annotation segmentation model.

`--diagnostics-every 20` logs full-mask character, first-EOS and tail-EOS scores.
`--checkpoint-every 20` saves periodic, absolute-best-CER, exact-line-frontier
and final checkpoints, while epoch metrics remain complete. `--save-traces`
opts into detailed sampling traces.
`diagnose_diffusion_run.py` evaluates an explicit final memorization checkpoint
with three sampling seeds and optional image controls; it does not evaluate the
held-out test partition. `report_diffusion_ablations.py` produces the comparison
JSON. No replacement-refinement sampler or visual encoder change is included in
this first controlled comparison.
