# Bounded implementation verification — 29 October 2026

**Historical snapshot:** the status and remaining-work statements below refer to
29 October preparation. The geometry fix, completed training and later results
are recorded in the [current experiment README](experiments/arabic_ar_mdiff/README.md).

The implementation and commands are in [MDIFF_ARABIC_V1.md](MDIFF_ARABIC_V1.md).
**Accuracy readiness has not passed. Do not start primary training yet.**
No long primary training, historical test inference or final evaluation was run.
No existing model, comparator, split lock or checkpoint was replaced.

## Passed mechanics

- 48 regression tests passed; one optional legacy transformers/ViT test skipped
  because transformers is absent. The new SVTR experiment does not require it.
- Actual Windows PowerShell 5.1.26100.9711: all action routes and explicit
  `Train -Resume` dry-routed; documentation blocks parsed; Setup and an existing
  immutable Check executed successfully. Native stderr with exit 0 was accepted;
  native exit 7 was rejected. The command launcher itself defaults to PowerShell
  7.6.5, so a separate explicit Windows PowerShell 5.1 process was used to verify.
- Exact official encoder transfer: 164 tensors, 11,707,970 elements, zero missing
  encoder keys. English decoder/head tensors explicitly skipped. Both primary
  arms share the complete model and encoder initialization hashes.
- Canonical 2103/150/752 IDs/counts, manifest/lock hashes, TRAIN/validation image,
  pixel and literal label bindings verified. Zero TRAIN/validation OOV, sequence
  overflow or image-limit issues. No test inference or test-derived vocabulary.
- Final Check and Smoke pass for `ma1_s54101` and `ar1_s54101`. Smoke tests real
  BF16 forward/backward with nonzero gradients in encoder, visual projection,
  decoder, embeddings, output and timestep module, plus finite backward on the
  widest TRAIN image (489 pixels at height 32). Peak smoke allocation was
  189,922,304 bytes, not a full training memory measurement.
- Deterministic repeated decoding passed. Diffusion executes three rounds with
  **six** actual calls (denoise/correction), checked by trace; untrained AR uses
  up to 144 causal calls. These call counts are not recognition accuracy.
- Synthetic tests check causality/padded memory, literal Arabic spaces/newlines,
  EOS/PAD, actual correction of a confidently wrong visible token, exact-line
  damage hidden by equal totals, failed prediction denominators, immutable paired
  reports, drift detection and stopping before test-data access on a failed gate.

## Bounded optimizer probes — not validation results

Both use four fixed TRAIN examples, 35 reference characters, 7 words, seed 54101,
120-update/60-second limits, with fresh initializations and no augmentation.
These are diagnostic memorization results, not estimates of OCR quality.

| Probe | Completed updates | Seconds | Normal CER | Exact | Blank CER | Shuffled CER | Readiness |
|---|---:|---:|---:|---:|---:|---:|---|
| MDiff | 120 | 36.00 | 48.57% | 0/4 | 65.71% | 105.71% | failed |
| Matched AR | 120 | 10.66 | 0.00% | 4/4 | 0.00% | 108.57% | failed |

Diffusion missing-token CER was 45.71%, replaced-token CER 40.00%, and correct
state CER 42.86%. It damaged all four correct input lines in this short probe.
All outputs had EOS; finite-training and time-budget checks passed. Recovery,
preservation and normal-image accuracy thresholds failed.

AR memorized the four examples but also produced four exact outputs for blank
images with their original geometry. Image widths/positional extents remain
available in this diagnostic; that can supply shortcuts on four memorized lines.
The required normal-versus-blank advantage failed. This is not accepted as a
successful image-dependency check, and the criterion was not relaxed.

The 12-update FP16 scaled probe overflowed before the bounded BF16 checks.
Actual hardware BF16 support was verified and the frozen recipe uses BF16 with
no gradient scaling. No nonfinite-gradient check was disabled.

The `ck03` probes retain their exact historical source snapshots. A subsequent
small lifecycle change prevents resume beyond an already satisfied stopping
condition and adds regression assertions. Final primary-name Check/Smoke bind
the final executable sources. Probe reports are not copied into passing primary
readiness artifacts. No primary training directories exist yet.

## Still required

Run the supplied 1200-update/900-second Overfit commands under the final primary
run names. Stop and return the reports if any condition fails. Longer fitting
may or may not pass; in particular, more memorization does not guarantee that
the blank-image dependency failure disappears. Do not start Train just because
teacher-forced loss is low. Do not run Test until both completed arms pass the
true frozen validation report and all Python hash/condition checks.

Long training stability, real validation scores, selected checkpoints, a passing
promotion gate, historical test comparison and compatible interrupted-run resume
on a real long run remain unverified. Resume argument routing and stop-boundary
logic are tested, but no long training interruption was manufactured.

Machine-readable evidence, exact report paths and hashes are saved at
`research/2026-10-29_transformer_diffusion_ocr/mdiff_arabic_v1_verification.json`.
The primary checks/source snapshots/logs live under `Diffusion/drive_all/checks`
and `Diffusion/drive_all/logs`. Frozen baselines were only read/hashed.

## Scope reconnaissance

No applicable AGENTS.md was found in the project/repository or parent locations.
Repository history: 30 commits, 30 July–14 September 2026, two contributors,
detached checkout with many pre-existing dirty/untracked files. Historical
hotspots include pipeline contract tests and training scripts. Relevant existing
sources were inspected; no broad refactor or unrelated history cleanup was
performed. All experiment code is in new versioned files.
