# Three AR/MDiff fusion experiments

**Completed run:** all three methods have now been executed by the user. See the
[current experiment README](experiments/arabic_ar_mdiff/README.md) for measured
results, selected epoch/configurations, preservation damage and artifact hashes.

This isolated suite prepares the three methods discussed in the 2026-10-31 research report. The user runs every real preparation, training and evaluation command. The implementation was verified using CPU synthetic fixtures and command dry runs; no new OCR inference or training was executed while preparing it.

Runner: `C:\Users\ammar\OneDrive\Desktop\final project\Diffusion\run_ar_mdiff_fusion_v2.ps1`

Default output: `C:\Users\ammar\OneDrive\Desktop\final project\Diffusion\drive_all\hybrids\ar_md_fusion2_s54101`

## Methods in the final table

| Name | Implementation | Weight updates |
|---|---|---|
| `dual_rescore` | Both frozen models score the existing AR and MDiff candidates against their own image features. | None |
| `span_merge` | The same scoring layer additionally evaluates bounded combinations of aligned MDiff spans in the AR draft. | None |
| `shared_encoder` | Freeze the selected AR encoder and visual projection. Initialize the MDiff decoder from its selected checkpoint, add a residual adapter, and train only the head/adapter on canonical TRAIN. Then rescore AR/shared-MDiff candidates. | New MDiff head and adapter only |
| `shared_mdiff_raw` | The adapted MDiff head's output before fusion, shown as an additional control. | Same trained model as method 3 |

The first two models began from the same encoder initialization but have different final encoders. Their visual memories are computed separately. The third model genuinely reuses the AR encoder and projection: test-time code performs one image encoding and supplies that memory to AR directly and to MDiff through the adapter. The AR decoder remains unchanged. This is a frozen-backbone dual-decoder adaptation, not a jointly trained replacement for the transformer.

The final table also includes Paddle, frozen v2.2, v4, the original transformer and MDiff, and the previous whole-line hybrid when its bound report exists. Every row must cover the same canonical **752 historical lines**. Failed/no-EOS examples remain in the metrics and denominators.

## Run everything in PowerShell

Paste this block. It executes the entire workflow sequentially in your PowerShell session, including the training required for method 3, and then tests all three methods on the 752 lines:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $runner = 'C:\Users\ammar\OneDrive\Desktop\final project\Diffusion\run_ar_mdiff_fusion_v2.ps1'
    $runArgs = @{
        Python = 'C:\Users\ammar\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe'
        Device = 'cuda'
        RunName = 'ar_md_fusion2_s54101'
        ARunName = 'ar1b_s54101'
        MDiffRunName = 'ma1b_s54101'
        TestCacheName = 'mdar_20261031_x01'
        PreviousHybridName = 'ar_md_h1_s54101'
        Epochs = 60
        Patience = 10
    }
    & $runner -Action All @runArgs
}
```

`All` performs `Check`, `Prepare`, `Calibrate`, `TrainShared`, `PrepareShared`, `CalibrateShared`, `Test`. It stops on an execution/integrity error. This explicit historical comparison reports raw results even when development preservation checks fail; it does not modify the original validation gates or promote/replace any model.

Scoring can take considerably longer than the old string selector. MDiff scoring evaluates every active character and the first EOS with that position masked. Eight independent masked cases are batched at a time. Completed lines are cached on disk, and progress is printed every ten lines. Method 3 defaults to at most 60 epochs with patience 10; the best raw validation checkpoint is selected before fusion calibration. No accuracy or runtime improvement is assumed.

## Stop and resume

Use Ctrl+C to stop. Preparation/test scoring saves each completed line atomically. Training saves each completed epoch, optimizer state and selection state. An interrupted epoch restarts from the last completed epoch when resumed; work in that incomplete epoch is replayed. Uncommitted attempt directories are preserved.

To continue the entire workflow, paste the block above again with its last line changed to:

```powershell
& $runner -Action All @runArgs -Resume
```

Completed preparation, calibration and test stages are verified and reused. A fresh `All -Resume` also works when no training header exists yet. If shared training has already completed, it is verified and skipped. The suite retains the latest committed optimizer checkpoint and selected best model, plus epoch histories and validation predictions. It removes only its own obsolete committed state/model files after committing a newer checkpoint; it never deletes primary checkpoints.

Do not change code, parent models, dataset, environment, Epochs or Patience under an existing RunName. Those values are sealed in Check. Use a new name for a changed experiment. Inference uses CUDA/BF16 when `Device='cuda'`; there is no silent CPU fallback.

## Run stages separately

Use the same `$runner` and `$runArgs` definitions within a script block:

```powershell
& $runner -Action Check @runArgs
& $runner -Action Prepare @runArgs
& $runner -Action Calibrate @runArgs
& $runner -Action TrainShared @runArgs
& $runner -Action PrepareShared @runArgs
& $runner -Action CalibrateShared @runArgs
& $runner -Action Test @runArgs
```

Only training needs an explicit resume flag when interrupted:

```powershell
& $runner -Action TrainShared @runArgs -Resume
```

After completion, display the saved table without rerunning inference:

```powershell
& $runner -Action Stats @runArgs
```

## Scoring and span rules

AR scoring uses shifted teacher forcing, a causal mask, and mean log probability over literal characters plus the first EOS. MDiff uses masked pseudo-log-likelihood: one active position is hidden in each scoring case, the tail contains MASK rather than length-revealing padding, and the noise fraction is `1 / active_length`, matching the cloze training objective. PAD and repeated EOS are excluded from both aggregates. MDiff pseudo-likelihood is not a normalized sequence probability. A fully visible correction-head copying score is never substituted for it.

For candidate `c`, the fused score is `(1-alpha)*AR_mean_logp(c) + alpha*MD_mean_logp(c)`. A different candidate is accepted only if its gain over the original AR candidate is strictly greater than the configured margin. The unchanged AR candidate remains available. This rule cannot guarantee preservation on unseen data.

Span proposals use deterministic Levenshtein alignment in logical Unicode order. Zero-width insertions and deletions are represented explicitly. At most six disagreement spans, each no longer than eight characters on either side, are considered. Candidates include individual replacements and one combination of the two shortest eligible spans, bounded by a ten-candidate limit and capacity 144 including EOS. Complete-line image-conditioned rescoring chooses among them. This version implements span splicing; it does not silently claim the optional new masked-regeneration editor was trained or evaluated.

Each method independently chooses alpha from `[0.25, 0.5, 0.75]` and margin from `[0, 0.05, 0.15]` on 120 development validation lines. The grouped 30-line audit is excluded from this grid search. Selection minimizes character edits, word edits, negative exact count and switch count, then prefers the larger margin and smaller alpha. A selected configuration that fails preservation is reported honestly rather than hidden by a replacement model.

All 150 validation labels were already used during primary checkpoint selection. Shared-head checkpoint selection also uses these 150 labels; its audit is therefore excluded only from the subsequent fusion-grid search. The 752-line historical set has also been used in development. Neither partition is a fresh generalization benchmark. No test reference or test error participates in head training, checkpoint selection, proposal construction or fusion-grid calibration.

## Shared training recipe

Canonical TRAIN: 2,103 lines, no test examples. Backbone: selected AR epoch 93 encoder and visual projection, frozen in eval mode; their complete parameter/buffer hash must match before training, after every epoch, and when loading the selected model. MDiff head: selected epoch 177 decoder, embeddings, time and output layers. Adapter: `Linear(256,256) -> GELU -> Linear(256,256)` with a zero-initialized final layer, added residually to AR visual memory.

The inherited MDiff loss remains masked CE/correction CE plus tail EOS supervision. AdamW uses learning rate 1e-4, weight decay .01, accumulation 16, gradient clipping 1, two warmup epochs, cosine decay to .1 of initial LR, existing TRAIN image augmentation and BF16 on CUDA. No AR weights or backbone parameters are optimized. Epoch ordering and RNG are reseeded by epoch for replay after interruption. Selection uses raw MDiff validation character edits, word edits, negative exact count, then earliest epoch. The fused shared result and raw shared result are both reported.

## Output and verification

- `check.json`, `source/`, `roles.json`: immutable experiment/input/source identities.
- `work/frozen_validation/`, `work/shared_validation/`: per-line development score caches.
- `calibration/frozen/`, `calibration/shared/`: score-grid trials, frozen settings, full development/audit results and seals.
- `shared/`: new training history, latest resume state and selected adapted head.
- `work/frozen_historical_752/`, `work/shared_historical_752/`: resumable historical score caches.
- `historical_752/report.json`, `report.txt`, `predictions.jsonl`, `seal.json`: final comparison, configuration/selection identity, all predictions and preservation diagnostics.
- `Diffusion/drive_all/logs/ar_md_fusion2_s54101/`: timestamped PowerShell logs.

The report includes exact lines lost and gained against AR, CER/WER, S/I/D, coverage/EOS, scoring time and peak CUDA memory. Frozen candidate scoring is shared between methods 1 and 2, so reported pool time is an amortized preparation cost, not a falsely independent latency estimate for each method. Original cached model timings are not presented as freshly measured timings for the new methods.

Code verification command (CPU/synthetic fixtures; no real OCR model runs):

```powershell
& $runArgs.Python -m pytest -q 'C:\Users\ammar\OneDrive\Desktop\final project\.codex-pretrained-selection\tests\test_ar_mdiff_fusion_v2.py'
```

Research rationale and primary publications: `C:\Users\ammar\OneDrive\Desktop\final project\research\2026-10-31_ar_mdiff_hybrid_research\FINDINGS.md`.
