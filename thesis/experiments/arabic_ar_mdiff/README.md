# Arabic handwriting: Transformer, MDiff and hybrid experiments

This is the current record of the experiments implemented and completed during
29–31 October 2026. It covers the matched recognizers, readiness failures and fix,
historical evaluation, the first whole-line selector, research, and all three new
fusion methods. It records successful and unsuccessful results together.

**Best aggregate historical result: `span_merge`, CER 48.77%, WER 83.28%,
151/752 exact lines (20.08%).** Dual rescoring also improved all three aggregate
metrics over the Transformer. The shared-backbone adaptation was weaker.

These are **historical development-used comparisons**. Validation and the 752
historical lines have been consulted repeatedly; neither is a new final test set.
No automatic model promotion occurred. The new methods still damaged some
previously exact Transformer lines, so aggregate gains do not satisfy the strict
zero-damage preservation check.

## Results on the same 752 lines

All rows have the same 13,493 reference characters and 2,793 reference words.
CER/WER are aggregate edit distances divided by their reference denominators,
not averages of per-line percentages. Lower CER/WER is better; higher exact
accuracy is better. The following figures are from the saved report, not a new
evaluation performed while writing this README.

| Model | Lines | CER % | WER % | Exact % | Exact lines | Sub | Ins | Del |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paddle | 752 | 59.39 | 92.84 | 6.52 | 49 | 1463 | 104 | 6446 |
| Frozen v2.2 | 752 | 58.56 | 93.02 | 6.25 | 47 | 2704 | 111 | 5086 |
| Span hybrid verifier v4 | 752 | 55.91 | 93.09 | 6.25 | 47 | 2626 | 117 | 4801 |
| MDiff Arabic raw | 752 | 53.84 | 91.87 | 13.30 | 100 | 5031 | 596 | 1637 |
| Transformer Arabic control | 752 | 51.40 | 84.21 | 18.09 | 136 | 4364 | 845 | 1727 |
| Whole-line hybrid v1 | 752 | 51.13 | 84.57 | 16.36 | 123 | 4375 | 845 | 1679 |
| Dual rescoring | 752 | 49.09 | 83.57 | 19.95 | 150 | 4205 | 730 | 1689 |
| **Span merge** | **752** | **48.77** | **83.28** | **20.08** | **151** | **4160** | **743** | **1678** |
| Shared MDiff raw | 752 | 56.00 | 92.91 | 11.84 | 89 | 5163 | 585 | 1808 |
| Shared encoder fusion | 752 | 50.15 | 83.96 | 18.88 | 142 | 4232 | 798 | 1737 |

The whole-line hybrid removed only 37 character errors relative to AR but added
10 word errors and reduced exact lines by 13. Dual rescoring removed 312
character errors; span merge removed 355. Span merge's additional gain over dual
rescoring is 43 character edits, eight word edits and one exact line.

Exact-line gains must be read together with damage:

| Method versus the Transformer | Exact lines lost | Exact lines gained | Net exact gain | Character edits | Word edits |
|---|---:|---:|---:|---:|---:|
| Whole-line hybrid v1 | 17 | 4 | -13 | 6899 | 2362 |
| Dual rescoring | 10 | 24 | +14 | 6624 | 2334 |
| Span merge | 17 | 32 | +15 | 6581 | 2326 |
| Shared encoder fusion | 8 | 14 | +6 | 6767 | 2345 |

Span merge is the leading **research candidate by aggregate accuracy**, not a
zero-damage replacement. Dual rescoring is a useful simpler comparison and loses
fewer previously exact lines. The specific shared-backbone recipe did not beat
the two methods using separately fine-tuned encoders; this does not establish
that every possible shared-encoder design will fail.

Machine-readable aggregates are in [RESULTS.json](RESULTS.json). Local report and
checkpoint identities are in [ARTIFACTS.json](ARTIFACTS.json). These snapshots omit
student texts and per-line IDs. Full reports, predictions, images and weights
remain in the existing local experiment directories.

## Data, targets and evaluation policy

- Canonical split: **2,103 TRAIN / 150 validation / 752 historical comparison**,
  under `Diffusion/drive_all/splits_70_05_25_v3` in the parent project directory.
- Images/labels live under `Diffusion/drive_all/ground_truth_lines`. Split locks,
  image bytes, decoded RGB pixels and literal labels are checked by the workflow.
- Vocabulary is derived only from TRAIN. Full lines are recognized in logical
  Unicode order; Arabic text is never reversed.
- `no_harakat_v1` removes only U+064B through U+0652. All other characters,
  children's spelling, tatweel, digits, punctuation and literal whitespace stay.
  There is no dictionary/spelling correction or external language model.
- Capacity is 144 tokens including EOS. RGB images are resized to height 32 with
  aspect ratio preserved, width limit 640, and at most three pixels of white
  stride alignment. No word cropping, truncation, horizontal flips or width squash.
- Failed/no-EOS outputs stay in coverage accounting and reference denominators.
  EOS terminates output; PAD/MASK are never emitted as transcription characters.
- Existing split overlap is documented: train/validation shares 106 pages and
  23 notebook proxies; train/historical shares 225 pages and 30 proxies.
  Proxies are not verified writer identities. Repeated use and group overlap
  limit claims about generalization; no new grouped holdout was fabricated.

## The two primary recognizers

Both use the official SVTRv2 small pretrained encoder transferred exactly from
the pinned MDiff4STR small checkpoint, with new Arabic character heads. They
start from the same complete initialization and use the same data and maximum
training schedule. The encoder has stage depths `[3,6,3]`, dimensions
`[128,256,384]`, and two Conv blocks before stage-2 FGlobal, matching the selected
checkpoint rather than a generic upstream configuration.

There is a 384-to-256 visual projection, deterministic positional encodings and
three 256-wide, eight-head Transformer decoder layers. Each original recognizer
has 15,069,703 parameters. Encoder weights are pretrained; English output and
decoder weights are not transferred to the original Arabic heads.

**AR / Transformer:** predicts the next character conditioned on the image and
earlier characters, using a causal mask. Training averages two independently
dropped-out causal CE views. Inference is greedy, causal character generation.

**MDiff:** reconstructs masked text conditioned on the image and visible text.
Its six uniformly selected corruption patterns are full masking, logical prefix,
logical suffix, contiguous block, one-token cloze and a random subset. A second
training call learns correction of wrong visible tokens and includes 25%
identity examples. Loss is `.5 * masked CE + .5 * correction CE + .1 * tail EOS CE`.
The structured loss is a surrogate, not a claimed unbiased diffusion ELBO.

MDiff inference uses three rounds, each containing masked denoising followed by
visible-token correction: **six decoder calls**. Between rounds it remasks
low-confidence active tokens. The correction call can change visible text,
including confidently wrong characters. This is an Arabic full-line adaptation
of MDiff ideas, not a faithful reproduction or an official Arabic checkpoint.

### Same initialization does not mean the same trained encoder

Both primary encoders began with tensor hash
`c6fa52b3592b18971c80aa10c2f5db5a9c9cb3c73b302e595e03eb801ee6bfeb`.
After the initial five frozen epochs they were fine-tuned independently. Their
selected final encoder hashes differ:

| Encoder | Final encoder tensor SHA256 |
|---|---|
| AR | `4ac4264133c860c4f1c070416fb134c2b02e3c4dfe9b47509e7fdc15674e4d8c` |
| MDiff | `fa48882969185b0f9597e952f48e08166b7e2eca87fc27ffa5a692f301d63704` |

Consequently, frozen dual rescoring/span merge compute separate image memories.
Their heads cannot simply exchange encoder features or average weights without
an additional model change. Method 3 below creates actual sharing deliberately.

### Training, selection and completed checkpoints

Primary recipe: seed 54101, AdamW, decoder LR 3e-4, encoder LR 1e-5, weight decay
.01, gradient clip 1, batch 1 with accumulation 16, five warmup/frozen-encoder
epochs, cosine decay to .1 of initial LR, maximum 200 epochs and patience 30.
Existing `trocr_views_v1` augmentation is seeded by sample and epoch. Pretrained
BatchNorm statistics remain frozen. No synthetic training images were added.

Hardware/runtime: native Windows, Python 3.13.14, Torch 2.10.0+cu126, CUDA 12.6,
RTX 4050 Laptop GPU with about 6 GiB memory and BF16 support. NumPy 2.3.2,
Pillow 11.3.0 and PyYAML 6.0.3 were recorded. Training used **CUDA/BF16**;
small numeric selectors run on CPU. No packages were changed by the preparation.

Checkpoint selection minimizes raw validation character edits, then word edits,
then negative exact count, then epoch. Selection and release eligibility are
separate: a selected best checkpoint can fail the release gate.

| Run | Completed epoch | Selected epoch | Checkpoint | Validation CER / WER / exact |
|---|---:|---:|---|---|
| `ar1b_s54101` | 123 | 93 | `m093.pt` | 47.47% / 82.01% / 29 of 150 |
| `ma1b_s54101` | 200 | 177 | `m177.pt` | 52.23% / 91.86% / 21 of 150 |
| Shared MDiff head | 34 | 24 | `shared/e024_20261031T114014558069Z/model.pt` | 55.07% / 92.72% / 21 of 150 |

Selected checkpoint SHA256 values:

- AR: `2355240de7b059736845c778bec2a8f62c84124ca6321908e735856e1d0ade97`.
- MDiff: `af4b05ac4bf6dc2b6eabba706b382c7e4e154138dd65b1902263a4fa44a77e45`.
- Shared head: `c0e7bcf25b270138d04959e902e58ee74871bae662434971264b4cf78b835cd2`.

AR completed 16,236 optimizer updates; MDiff completed 26,400. Primary training
time was about 374.43 minutes for AR and 647.12 for MDiff, excluding about 32.31
and 23.71 minutes of validation respectively. Matched maximum schedules do not
mean equal actual compute, especially with different early stopping.

## What we tried, including failures

### 1. Initial implementation and numerical checks

Exact transfer verified 164 encoder tensors with no missing or shape-mismatched
keys. Initial mechanics checks covered loss/gradients, literal text, variable
image widths, padded visual memory, determinism, EOS, corruption support and
six actual MDiff sampling calls. A small FP16 scaled probe overflowed; BF16 on
the supported GPU resolved the numerical issue while retaining finite checks.

The first 120-update TRAIN-only probe did not establish readiness: MDiff had
48.57% normal CER and damaged all four correct-input lines; AR memorized all
four examples but also recognized their blank-image counterparts exactly.
These are diagnostic results, not validation recognition scores.

### 2. Failed overfit readiness and the geometry fix

With 1,200 updates the original AR tiny-subset report still failed only the
normal-versus-blank advantage: both were 0% CER. Distinct image widths could
identify the four examples after blanking. The MDiff paired probe passed its
numeric thresholds, but the differing geometries still limited interpretation.

The v1.1 protocol groups TRAIN examples by encoder extent and chooses four
distinct labels with identical blank input shape/pixels (`1 x 3 x 32 x 68`).
Shuffling swaps content within that same geometry. It uses no validation/test
error ranking. **No readiness threshold was weakened.** New run names
`ar1b_s54101` / `ma1b_s54101` received fresh checks and readiness fits before the
completed primary training. Failed reports and old names were retained.

### 3. Primary validation gate and explicit historical comparison

The frozen gate required each arm to beat v4 character edits, have no more word
edits, lose no line in the union of v4/v2.2 exact predictions, and have complete
successful EOS coverage. The paired gate failed: AR lost four old exact lines;
MDiff lost five and also did not beat v4 validation CER. This explains the
reported `validation_passes_release_gate is not true` stop.

The gate was not changed to true. At the user's explicit request a separately
versioned exploratory evaluator generated the historical comparison while
retaining checkpoint/data/source integrity checks and recording authorization.
The historical cache is `mdar_20261031_x01`. Later workflows consume its frozen
candidates instead of silently rerunning the old baselines. After the user
requested commands-only execution, training/evaluation commands were supplied
for the user to execute; no further real OCR runs were made while preparing the
fusion suite or this documentation.

### 4. Complementarity and whole-line selector v1

On 150 saved validation lines MDiff had fewer character errors on 50 lines, AR
on 58, with 42 ties. They shared 16 exact lines; MDiff alone was exact on five,
AR alone on thirteen. A reference-dependent whole-line oracle would have
42.67% CER and 34 exact lines versus AR's 47.47% and 29. This is an upper bound,
never an inference policy or an achieved hybrid score.

The v2.2-inspired v1 selector used 14 string/EOS/aspect-ratio features and a new
ridge regression, regularization 10. It did not reuse Paddle-specific gate
weights or invent unavailable confidence values. AR was the default and the
selector could only replace a complete line with MDiff's output.

Notebook-proxy roles were 90 fit / 30 calibration / 30 audit. The selected
threshold was .05. Full development preservation failed because one previously
exact AR line was lost. Explicit exploratory historical testing yielded CER
51.13%, WER 84.57%, exact 16.36%: a small CER gain with worse word/exact results.
This motivated image-conditioned scoring and local rather than whole-line edits.

### 5. Research informing the new methods

The [archived research](research/2026-10-31_ar_mdiff_hybrid_research/FINDINGS.md)
distinguishes published evidence from our adaptations:

- [HTR CTC-prefix rescoring](https://arxiv.org/abs/2110.05909) supports combining
  complementary image-conditioned decoder evidence; CTC is not MDiff.
- [Masked Language Model Scoring](https://aclanthology.org/2020.acl-main.240/)
  motivates masked pseudo-likelihood; its original tasks are speech/translation.
- [LevOCR](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3820_ECCV_2022_paper.php)
  motivates image-guided insertion/deletion refinement.
- [Speculative Refinement](https://aclanthology.org/2026.gem-main.33/) studies AR
  drafts followed by selective diffusion refinement and damage to correct tokens;
  its evaluations are not OCR.
- [PARSeq](https://arxiv.org/abs/2207.06966) motivates training for shared AR and
  bidirectional refinement behaviors, not merely changing a trained causal mask.
- [Controlled Arabic HTR decoder study](https://link.springer.com/article/10.1007/s44443-026-00970-6)
  cautions that decoder combinations can retain common visual/boundary errors.

No published benchmark score was treated as an expected score on these children’s
handwriting images. The earlier October-29 research and preparation notes are
also archived, including their source/implementation manifests.

### 6. Dual rescoring

Keep both selected recognizers frozen. For each original AR/MDiff candidate:

1. AR scores characters plus first EOS by shifted causal teacher forcing.
2. MDiff masks each scored position before predicting it, including EOS. Tail
   slots are MASK, and cloze noise fraction is `1 / active_length`. Eight
   independent masked cases are batched at a time.
3. Average each model's active-token log probabilities. Combine as
   `(1-alpha)*AR_mean_logp + alpha*MD_mean_logp` and require gain over the AR
   draft strictly greater than the configured margin.

PAD and repeated EOS are excluded. MDiff's score is pseudo-likelihood, not a
normalized sequence probability; a fully visible copying score is not used.
The unchanged AR candidate remains available. Actual visual evidence replaces
the earlier string-only gate, without training either parent recognizer.

### 7. Span merge

Use deterministic Levenshtein alignment to locate disagreements in logical
Unicode order, representing insertions/deletions explicitly. Consider up to six
short spans, at most eight characters on either side. Build individual MDiff-span
replacements in the AR draft plus one combination of the two shortest eligible
spans. Deduplicate and enforce a ten-candidate bound and token capacity.

Dual rescoring then evaluates each **complete proposed line**, preserving the AR
candidate. This lets a useful local MDiff correction survive without importing
all of MDiff's other errors. The implemented version is bounded span splicing.
Optional masked regeneration, a newly trained insertion/deletion editor, AR beam
search expansion and joint encoder training were discussed but **not executed
or claimed as part of these results**.

### 8. Shared encoder with an adapted MDiff head

Freeze the selected AR encoder **and its visual projection**, in eval mode.
Initialize MDiff's decoder/embedding/time/output from selected MDiff epoch 177.
Add a residual `Linear(256,256) -> GELU -> Linear(256,256)` adapter whose final
layer starts at zero. Only the MDiff head and adapter are optimized; AR's decoder
and backbone stay unchanged. Training retains the original MDiff objective.

The new recipe uses canonical TRAIN only, AdamW LR 1e-4, weight decay .01,
accumulation 16, gradient clipping 1, two warmup epochs, cosine decay to .1,
maximum 60 epochs and patience 10. It stopped at epoch 34 and selected epoch 24
by raw MDiff validation errors. Backbone parameter/buffer hashes are checked
before training, after every epoch and at load time.

Inference performs one AR image encoding, giving that memory directly to AR and
through the adapter to MDiff. Both raw shared-MDiff and fused AR/shared-MDiff are
reported. Raw CER worsened to 56.00%; fusion recovered to 50.15%, but neither
beat the independently encoded dual/span methods.

## Calibration, validation and preservation results

Each new method searches the same alpha grid `[.25,.5,.75]` and margin grid
`[0,.05,.15]` on 120 development lines, excluding the grouped 30-line audit from
this grid search. All selected **alpha .75, margin 0**. No 752-line reference or
error is used to choose those settings or to train/select the shared head.

| Method | Full 150 CER % | Full WER % | Full exact | Full AR-exact lost/gained | Audit CER % | Audit exact | Audit lost/gained |
|---|---:|---:|---:|---|---:|---:|---|
| Dual rescoring | 46.01 | 82.01 | 29 | 4 / 4 | 53.12 | 7/30 | 0 / 0 |
| Span merge | 45.34 | 80.30 | 32 | 4 / 7 | 52.76 | 8/30 | 0 / 1 |
| Shared encoder fusion | 46.30 | 81.16 | 31 | 1 / 3 | 53.30 | 7/30 | 0 / 0 |

All three passed the AR-relative preservation checks on the small audit but
failed them on the full development set. Both primary checkpoint selection and
shared-head checkpoint selection had already used all 150 validation labels;
the audit excludes only the subsequent fusion-grid tuning. It is not untouched
model-selection data. Historical testing was explicitly requested despite these
failures and reports the selected raw methods without changing any prior gate.

The saved candidate-scoring times on the historical set were about 83.39 seconds
for the frozen pool jointly supporting dual/span methods, and 46.67 seconds for
the shared pool. Peak allocated CUDA memory was 154,429,440 bytes. These are
cached workflow measurements, not a rigorous independent per-method end-to-end
latency benchmark; model loading and shared preparation costs differ.

## Repository contents and local artifact layout

| Location in this repository | Purpose |
|---|---|
| `thesis/mdiff_arabic_v1/` | Primary recognizers, data/provenance checks, lifecycle and licensed SVTR vendor sources |
| `thesis/ar_mdiff_hybrid_v1/` | First whole-line selector |
| `thesis/ar_mdiff_fusion_v2/` | Dual scoring, span proposals, adapted shared model and resumable workflow |
| `thesis/scripts/run_*mdiff*.py` | CLI entry points |
| `tests/test_mdiff_arabic_v1.py`, `test_ar_mdiff_hybrid_v1.py`, `test_ar_mdiff_fusion_v2.py` | Contract and synthetic verification |
| This folder's `runners/` | Byte-identical deployable PowerShell launchers and standalone exploratory evaluator/tests |
| This folder's `research/` | Original findings, prompts, manifests and time-specific verification notes |
| `RESULTS.json`, `ARTIFACTS.json`, `archive_manifest.json` | Aggregate results, local artifact hashes and archived-file identities |

The existing Windows project layout expected by the CLI is:

```text
final project/
  .codex-pretrained-selection/       # this Git checkout
  Diffusion/
    run_mdiff_arabic_v1_1.ps1
    run_ar_mdiff_hybrid_v1.ps1
    run_ar_mdiff_fusion_v2.ps1
    evaluate_mdiff_exploratory_v1.py
    drive_all/
      ground_truth_lines/
      splits_70_05_25_v3/
      pretrained/mdiff_s_v1/
      runs/ar1b_s54101/              # selected m093.pt
      runs/ma1b_s54101/              # selected m177.pt
      exploratory/mdar_20261031_x01/
      hybrids/ar_md_h1_s54101/
      hybrids/ar_md_fusion2_s54101/
        shared/                    # selected adapted head + latest resume state
        calibration/{frozen,shared}/
        historical_752/             # final report.json, report.txt, predictions, seal
      logs/
```

Weights, datasets, original full reports and caches are intentionally local,
outside this commit. Source code alone cannot reproduce the measured scores
without the matching local artifacts. Existing data and selected weights must
be retained/backed up separately. Exact package versions, executable, absolute
paths, source bytes, assets, split locks and parent checkpoints are part of the
runtime bindings; this is not a promise that an arbitrary fresh path/environment
can resume an existing run unchanged.

Git attributes preserve exact bytes of checksum-bound sources and runner
snapshots, including existing CRLF/mixed line endings. This prevents Git's
automatic newline conversion from invalidating completed source bindings.
Four reused helpers use explicit CRLF attributes without functional changes;
`metrics.py` preserves its existing mixed endings and adds the required
`no_harakat_v1` scoring mode.

## Install the archived runners when restoring a checkout

The active runners originally lived outside this Git repository. This commit
archives byte-identical copies. **Run the deployed copies in `Diffusion`, not
the archive files directly**, because the launchers derive the parent project
from their location. On the existing project they are already installed.

This PowerShell block copies only missing files and rejects a different existing
file rather than overwriting a frozen experiment's runner:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $repo = 'C:\Users\ammar\OneDrive\Desktop\final project\.codex-pretrained-selection'
    $source = Join-Path $repo 'thesis\experiments\arabic_ar_mdiff\runners'
    $destination = Join-Path (Split-Path $repo -Parent) 'Diffusion'
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    foreach ($file in Get-ChildItem -LiteralPath $source -File) {
        $target = Join-Path $destination $file.Name
        if (Test-Path -LiteralPath $target) {
            if ((Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash -ne
                (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash) {
                throw "Existing runner differs; preserve it and restore a compatible checkout: $target"
            }
        } else {
            Copy-Item -LiteralPath $file.FullName -Destination $target
        }
    }
}
```

## Commands: saved results, new runs and resume

The completed run is `ar_md_fusion2_s54101`. To verify and print its saved table,
without training or scoring new lines:

```powershell
$runner = 'C:\Users\ammar\OneDrive\Desktop\final project\Diffusion\run_ar_mdiff_fusion_v2.ps1'
& $runner -Action Stats -Device cuda -RunName ar_md_fusion2_s54101
```

To verify/reuse the completed pipeline, including its frozen settings and caches:

```powershell
& $runner -Action All -Device cuda -RunName ar_md_fusion2_s54101
```

For a genuinely new repeat, choose an unused RunName. This command **will** train
the shared head and execute the historical comparison; it is supplied for the
user to run, not invoked by documentation/verification:

```powershell
& $runner -Action All -Device cuda -RunName ar_md_fusion2_repeat01
```

Defaults bind AR `ar1b_s54101`, MDiff `ma1b_s54101`, historical cache
`mdar_20261031_x01`, previous hybrid `ar_md_h1_s54101`, 60 maximum shared epochs
and patience 10. To inspect routing without accessing data/models, add `-DryRun`.

Ctrl+C may interrupt preparation, training or testing. Per-line scoring caches
and report publication are atomic. Shared training resumes its last committed
epoch with optimizer/selection state; an unfinished epoch is replayed. Continue
an incomplete repeat with:

```powershell
& $runner -Action All -Device cuda -RunName ar_md_fusion2_repeat01 -Resume
```

The OS lock prevents simultaneous writers under one RunName and releases on
process exit. Only the new shared run's redundant committed state/model files
are cleaned after committing newer state; selected/latest checkpoints remain.
Changed code, data, environment, epochs or patience require a new RunName.

Stages can be run independently with the same arguments: `Check`, `Prepare`,
`Calibrate`, `TrainShared`, `PrepareShared`, `CalibrateShared`, `Test`, `Stats`.
`Test` fixes evaluation membership to exactly 752 lines and requires all three
configurations to have been frozen first. It does not override or edit the
original primary/v1 release gates.

For primary training and the v1 selector, retain the stage-specific workflows:

- [Primary architecture and original workflow](../../MDIFF_ARABIC_V1.md).
- [Geometry-matched readiness v1.1 and corrected run names](../../MDIFF_ARABIC_V1_1_READINESS.md).
- [Whole-line hybrid v1](../../AR_MDIFF_HYBRID_V1.md).
- [Fusion v2 detailed recipe and PowerShell commands](../../AR_MDIFF_FUSION_V2.md).

Primary and v1 lifecycle/resume rules differ from the new fusion suite; do not
apply its automatic per-line/epoch replay assumptions to the older runner.

## Verification and provenance

Preparation verification comprised 79 passing code checks across the three
versioned packages, including synthetic 752-line reports, token masking, causal
scoring, frozen-backbone gradients, interruption/resume, checkpoint cleanup,
atomic publication, tamper detection and Windows PowerShell 5.1 exit handling.
The original primary source bindings/checkpoint hashes and previous hybrid
source hashes matched. The real fusion accuracy above comes from the user's
subsequent run, not those synthetic checks.

To run the code checks from the repository root after deploying the runners:

```powershell
python -m pytest -q tests/test_mdiff_arabic_v1.py tests/test_ar_mdiff_hybrid_v1.py tests/test_ar_mdiff_fusion_v2.py
```

The standalone exploratory evaluator has three additional synthetic tests in
`runners/test_exploratory_mdiff_v1.py`; after deployment run the copy in
`Diffusion/test_exploratory_mdiff_v1.py`. These tests do not run real OCR training
or evaluate the actual 752 images. Historical verification documents describe
status **at preparation time**, including failures and then-unexecuted steps;
this README and the aggregate snapshots describe the later completed outcomes.

Before committing this record, only the staged files and their tracked Python
dependency closure were exported into a separate temporary project. After
deploying the archived runners there, **82 tests passed in 56.62 seconds**
(the original 79 plus the three standalone evaluator checks). Primary and both
hybrid source hashes matched their frozen bindings after checkout. Archived
runner/research hashes, current README links and aggregate snapshots were also
verified. The staged secret/size scan and whitespace check passed. No new real
OCR training or historical evaluation was run during this documentation commit.

Upstream attribution: [OpenOCR](https://github.com/Topdu/OpenOCR), commit
`1ccfc6ee6161f7133b192e16af3a9265273997ba`; Apache-2.0 license and modifications
are retained in `thesis/mdiff_arabic_v1/vendor/{LICENSE,NOTICE.md}`. Official
[MDiff4STR small assets](https://huggingface.co/topdu/MDiff4STR) are pinned to
revision `70c320bb8ab539ecc08f820ef94ba7679b6e40f2`; asset SHA256 values and exact
transfer details are in the primary guide. No new license for upstream weights
is invented, and no pretrained/trained weights are redistributed here.
