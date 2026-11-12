# KHATT + children: five AR Transformer experiments

Prepared on 2026-11-06. Real training and historical model evaluation are **user-run only**. The preparation step has generated and verified the manifests. No new child-only experiment is included.

## Completed historical results (2026-11-10)

The user completed training and the four-run historical comparison. All models covered the same 752 lines with zero failed, missing-EOS or length-failure outputs. [RESULTS.json](RESULTS.json) and [RESULTS.csv](RESULTS.csv) contain aggregate exports; [ARTIFACTS.json](ARTIFACTS.json) records checkpoint/input/source hashes without publishing per-line data.

| Model | CER % | WER % | Exact lines |
|---|---:|---:|---:|
| Previous AR | 51.40 | 84.21 | 136/752 |
| Previous span merge | 48.77 | 83.28 | 151/752 |
| KHATT only | 73.84 | 109.63 | 14/752 |
| **100% KHATT + children** | **29.92** | **65.74** | **185/752** |
| 50% KHATT + children | 37.97 | 72.75 | 167/752 |
| 30% KHATT + children | 37.85 | 69.89 | 184/752 |

The full mixture reduces CER by 38.7% relative to the previous span merge. It gains 72 previously non-exact span lines but loses 38 previously exact ones, so aggregate improvement is not zero-damage preservation. The 30%/50% ordering is not strictly monotonic. KHATT-only transfers poorly to this children dataset. The extra 704 children training lines and changed schedule/vocabulary prevent attributing the whole improvement solely to KHATT.

The new Transformer and both old span components, with exact tokenizers and configuration, are preserved through [Git LFS](../arabic_ocr_best/README.md). The [MDiff handoff for Marwan](MDIFF_HANDOFF.md) specifies the next matched experiment and the later new-pair hybrid. No further percentage sweep or new child-only training is requested.

**Scope update (2026-11-10):** the user abandoned `khatt10_ours` after observing the validation trend. It remains an incomplete run with its saved epoch-15 best preserved; it is not marked completed and will not be tested. The other four runs have completed, with selected validation-best epochs **34 / 40 / 34 / 87** for KHATT-only / 100% mixed / 50% mixed / 30% mixed. No further training is needed for this comparison.

Use the existing epochs on the same historical 752 lines:

```powershell
$runner = 'C:\Users\ammar\OneDrive\Desktop\final project\Diffusion\resume_khatt_ablation_v1.ps1'
& $runner -Action TestAll -SuiteName khatt_ar_v1fix1_s54101 -Skip10Percent -Device cuda
```

`Check -Skip10Percent` prepares and hash-verifies the four-run plan without inference. `trial_decision.json` records the abandonment, and `test_plan_without10/request.json` binds the four existing checkpoints, their validation selections and the added evaluator sources. The compatibility launcher's subsequent `TrainAll` skips the abandoned trial, explicit `Train -Trial khatt10_ours` is rejected, and `TestAll`/`Stats` use the four-run plan. The original training source, dataset manifests, best/checkpoint files and 10% saved progress are preserved. The existing historical evaluator runs with the included four-trial set in this locked process; its original trial configuration is restored afterward. The same per-line caches, 752-line checks, comparison baselines and metrics apply. `test_plan_without10/result_binding.json` connects the final report to this plan. Model testing has not been launched by the assistant.

**Validation adapter fix (2026-11-06):** the original `khatt_ar_v1_s54101` run failed at its first validation because `Recognizer.generate()` returns `(token_ids, metadata)` and the adapter treated the whole pair as the tensor. The adapter now unpacks that return value and checks the batch shape. The original mock has been corrected, and a regression check runs the real randomly initialized encoder/decoder on one synthetic CPU image without training or using a trained checkpoint.

The corrected default suite is **`khatt_ar_v1fix1_s54101`**. Its prepared manifests, tokenizer, validation and training recipe are identical to the original. The failed suite's source snapshot, check and `run.json` are preserved. No checkpoint had been saved in that failed run, so its first epoch must restart; there are no saved weights to recover. A new suite preserves the original source binding rather than silently rewriting it.

**Resume compatibility update (2026-11-08):** installation of `opencv-python==5.0.0.93` added one package to the global Python inventory and triggered the original launcher's strict environment binding. All original source files, data, package versions, CUDA, Torch and training settings still match. KHATT-only completed at epoch 44 (selected 34); KHATT + children has a committed epoch-35 state. Do not discard those runs or create another suite for this package-only change.

The commands below now use **`resume_khatt_ablation_v1.ps1`**, a separate audited compatibility launcher. It accepts only that exact added OpenCV distribution (or an exact original environment). It rejects all original dependency/version changes, unreviewed additions, code/data changes, or a loaded/referenced `cv2` module. Its check action verified the saved model/state hashes without model inference. It preserves the original bound launcher, code, check and checkpoint identities. The full observed environment and compatibility-helper source snapshot are recorded under the suite's `environment_compatibility/` directory; the package addition is not erased from the audit. Seven dedicated synthetic compatibility tests passed.

The user selected **AR Transformer only**, with all **2,807** usable children TRAIN lines: the original 2,103 plus the 704 newly synchronized lines. The original **150 children validation lines** and **752 historical children test lines** retain their assignments. No KHATT line enters validation or the historical test.

## Training sets

| Trial name | KHATT TRAIN lines | Children TRAIN lines | Total TRAIN | Validation |
|---|---:|---:|---:|---:|
| `khatt100_only` | 9,497 | 0 | 9,497 | 150 children |
| `khatt100_ours` | 9,497 | 2,807 | 12,304 | 150 children |
| `khatt50_ours` | 4,748 | 2,807 | 7,555 | 150 children |
| `khatt30_ours` | 2,849 | 2,807 | 5,656 | 150 children |
| `khatt10_ours` | 949 | 2,807 | 3,756 | 150 children |

Percentages refer to the **9,497 annotated rows in the mirror's `Train.csv`**, not a batch mixture ratio. Counts use `floor(9497 * percentage / 100)`. A seed-54101 SHA-256 ranking of stable KHATT sample IDs defines fixed, nested subsets: 10% is contained in 30%, then 50%, then 100%. Selected lines are visited once per epoch, shuffled. The mixed runs include every children TRAIN line once per epoch; no replacement sampling or repeated balancing is applied.

The separately supplied KHATT `Validation.csv` (1,901 annotated lines) is unused. There is no supplied Test folder in this Kaggle mirror. Twenty-five images in its Train folder have no CSV row and are explicitly listed in `prepared.json`; they are preserved on disk and excluded because they have no transcription. The existing three standalone children `≠` non-transcription markers remain excluded; embedded literal `≠` remains supported.

## Run these PowerShell commands

The original training command is retained here for reference. It resumes incomplete included runs and skips completed or abandoned trials; the four included runs are now complete:

```powershell
$runner = 'C:\Users\ammar\OneDrive\Desktop\final project\Diffusion\resume_khatt_ablation_v1.ps1'
& $runner -Action TrainAll -SuiteName khatt_ar_v1fix1_s54101 -Device cuda -Epochs 100 -Patience 10 -Resume
```

`100` is a maximum. A trial stops after **10 consecutive validation checks without improvement**. Selection minimizes validation character edits, then word edits, then maximizes exact lines; ties retain the earlier epoch. Validation runs after each full epoch on exactly the same 150 children lines. Test scores never select a checkpoint.

To run or resume trials individually, run the desired command:

```powershell
$runner = 'C:\Users\ammar\OneDrive\Desktop\final project\Diffusion\resume_khatt_ablation_v1.ps1'
& $runner -Action Train -Trial khatt100_only -Device cuda -Epochs 100 -Patience 10 -Resume
& $runner -Action Train -Trial khatt100_ours -Device cuda -Epochs 100 -Patience 10 -Resume
& $runner -Action Train -Trial khatt50_ours -Device cuda -Epochs 100 -Patience 10 -Resume
& $runner -Action Train -Trial khatt30_ours -Device cuda -Epochs 100 -Patience 10 -Resume
```

The included four trials are complete. This separate command evaluates their selected checkpoints on the historical 752 lines and prints the table (already run by the user):

```powershell
$runner = 'C:\Users\ammar\OneDrive\Desktop\final project\Diffusion\resume_khatt_ablation_v1.ps1'
& $runner -Action TestAll -SuiteName khatt_ar_v1fix1_s54101 -Skip10Percent -Device cuda
```

`TrainAll` never invokes `TestAll`. The current four-run test plan requires all four included runs to have finalized checkpoints; it does not require the abandoned 10% trial to finish. Interrupted testing resumes from saved per-line outputs. It compares the four included recognizers against the original Paddle, v2.2, v4, MDiff and AR recognizers, plus the **existing frozen span-merge predictions**. It does not refit span merge or combine the new recognizers with MDiff.

The table shows `Model / Lines / CER % / WER % / Exact % / Sub / Ins / Del`. JSON and CSV also include exact lines lost/gained versus span merge. All 752 lines remain in the denominator, including no-EOS outputs. Model errors stop execution with the already committed work retained; missing predictions cannot silently reduce the evaluated set.

Reprint a completed table without inference:

```powershell
& $runner -Action Stats
```

Preparation is idempotent. Metadata/source/CUDA verification and a launcher-only dry run are available:

```powershell
$originalRunner = 'C:\Users\ammar\OneDrive\Desktop\final project\Diffusion\run_khatt_ablation_v1.ps1'
& $originalRunner -Action Prepare
& $runner -Action Check -Device cuda
& $runner -Action TrainAll -Device cuda -Resume -DryRun
```

## Initialization and recipe

Each run starts from the same official pretrained SVTRv2-small encoder used by our existing AR experiment, with a **fresh Arabic decoder**, seed 54101. None starts from the already child-trained AR checkpoint, including `khatt100_only`. A common declared Arabic/Latin/digit/punctuation vocabulary, with literal CR/LF/TAB/inequality support, gives all five runs the same head shape and initial parameter values. No validation or test reference contributes vocabulary entries. This vocabulary is larger than the old children TRAIN-derived vocabulary; the old trained head cannot be loaded into it directly.

The architecture is our existing AR Transformer: SVTRv2-small, projection to 256, three 8-head Transformer decoder layers, dropout 0.1, capacity 144, greedy image-only decoding. Images retain logical orientation and aspect ratio at RGB height 32, aligned to multiples of 4. KHATT's longest TRAIN image requires width 2,090 at height 32, so the shared cap is **2,196**, including a margin. Child images retain their existing per-image resize; the cap does not stretch them. No line is truncated or discarded to fit the old width-640 limit. Maximum TRAIN transcription length is 131 characters plus EOS, within capacity 144.

Other settings follow the earlier AR recipe: batch 1, gradient accumulation 16, AdamW weight decay 0.01, decoder LR 3e-4, encoder LR 1e-5, gradient clip 1, warmup five epochs, cosine decay to 10%, encoder frozen for five epochs, fixed pretrained BatchNorm running statistics, encoder gradient checkpointing, BF16 on CUDA. Each batch uses the two AR dropout views from the existing implementation. The original/photometric/geometry augmentation cycle is unchanged. Only the new suite uses the requested 100-epoch maximum and patience 10.

The launcher defaults to **CUDA**, checks CUDA and BF16 availability, and fails if unavailable instead of silently using CPU. CPU is an explicit alternative for a new suite, with FP32; a CUDA-bound run cannot silently resume under another device/environment.

## Labels and provenance

Downloaded source: [Kaggle KHATT mirror](https://www.kaggle.com/datasets/nizarcharrada/khattarabic), version 2. The original download's manifest, image verification and source metadata are in `Diffusion/datasets/KHATT/`. This is a derivative mirror with deskewed JPEGs and CSV character codes; its Train/Validation membership is preserved rather than relabeled as the original official splits.

The main code-to-Unicode mapping was checked visually against the official [KHATT readme, Arabic–Latin lookup table, pages 5–6](https://catalog.ldc.upenn.edu/docs/LDC2015T23/readme_KHATT_v1.0.pdf). It distinguishes hamza/alif forms, ta marbuta, alif maqsura, qaf/kaf, ASCII digits, and punctuation. Mirror-specific lowercase `tee` aliases official `teE`. The mirror's compound `laaa/laae/laah/laam` tokens expand to logical `لا/لأ/لإ/لآ`; these are separately identified mirror conventions, not claimed to appear in the official table. The corresponding source images were inspected, including `AHTD3A0085_Para2_5` (لآ) and `AHTD3A0104_Para2_1` (لإ).

Empty CSV cells are padding. A literal `;` appearing as the final nonempty cell is the CSV's paragraph terminator, not a transcribed punctuation character; 4,825 TRAIN rows use it. An interior `;`, unknown code, duplicate annotation, missing image, or unsupported target stops preparation. There is no spelling correction, whitespace collapsing, RTL reversal, alif/hamza folding, or silent unknown-token replacement. Children follow the existing `no_harakat_v1` policy, removing only U+064B–U+0652. Original data files remain intact.

## Integrity, stopping and disk usage

Prepared manifests bind the original split lock, original split file hashes, CSV hash, mapping, tokenizer, image and child label hashes. New children TRAIN images are checked against validation/test pixel hashes. The suite records source hashes, environment, official encoder asset hashes and frozen comparator file hashes. Old model source packages, checkpoints, validation gates and saved historical results are unchanged.

Press **Ctrl+C** to stop. Re-run `TrainAll -Resume` to continue. A completed epoch atomically commits model, optimizer, random state, early-stopping counter and selected best. An interrupted epoch restarts from the last committed epoch; interruption during the first epoch restarts that epoch. A crash immediately after a stopping epoch cannot extend training past patience or the 100-epoch maximum. An OS file lock prevents two processes from using the same suite concurrently.

Only the best selected model and the latest resume state are retained per new run. Obsolete model/optimizer copies from this suite's own epoch transaction folders are pruned after a successful commit. Per-epoch validation predictions and history remain. Existing experiment artifacts are never candidates for this pruning.

Suite folder:

`C:\Users\ammar\OneDrive\Desktop\final project\Diffusion\drive_all\khatt_ablations\khatt_ar_v1fix1_s54101`

Main outputs: `prepared.json`, per-trial TRAIN manifests, `validation.jsonl`, `test_identity.jsonl`, `runs/<trial>/best.json`, `runs/<trial>/complete.json`, and later `historical_752/{report.json,predictions.jsonl,comparison.csv,seal.json}`. Transcript logs are under `Diffusion/drive_all/logs/khatt_ar_v1fix1_s54101/`. Completed prediction caches and reports are hash-sealed. Choose a different `-SuiteName` for a deliberate new experiment; existing bound runs reject changed source, manifests, assets or environment.

## Interpretation and verification

The old AR/span models used 2,103 child TRAIN lines; the mixed new models use 2,807. Improvements over the old span model therefore combine the effects of **704 additional child lines, KHATT, and the new training schedule/vocabulary**. The mixed trials provide a useful KHATT dose comparison, but there is no matched updated-child-only control because the user explicitly declined another child-only run. Different TRAIN sizes also imply different optimizer-update budgets at the same epoch limit. KHATT-only still selects its checkpoint on children validation, as requested.

The 752 lines have been used in earlier development. These results are a **historical development-used comparison**, not a new untouched generalization claim. Existing page/notebook overlap is unchanged, and the 704 added children lines may also share writers/pages; writer-disjoint performance is not established. No test result automatically promotes a model or changes the old validation release gates.

The 26 CPU-only protocol tests cover nested counts, conversion errors, manifest tampering, schedule/patience limits, owned checkpoint pruning, inference isolation, complete test membership, exact-line damage, environment compatibility, the four-run decision, separation of training and testing, and synthetic interrupted/resumed training with matching final weights, optimizer and RNG state. The decoder return contract is exercised with the real encoder/decoder using random weights and one synthetic image. No dataset image or trained OCR checkpoint is evaluated by these tests:

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\final project\.codex-pretrained-selection'
python -X utf8 -m unittest discover -s tests -p 'test_khatt_*v1.py' -v
```
