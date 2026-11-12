# Marwan: matched MDiff on KHATT + children

Owner: **@marwansakas**. Implement the diffusion counterpart of our completed best AR experiment. This is the next research step, followed by a possible new span hybrid. The AR experiment is complete; its training should not be repeated.

## Current result and artifacts

The best AR uses all **9,497 annotated KHATT TRAIN lines + 2,807 children TRAIN lines = 12,304 examples**. It selected epoch 40, stopped at 50 with patience 10, and achieved **29.9192% CER / 65.7358% WER / 185 of 752 exact lines** on the historical comparison. Children validation: **26.7223% CER / 60.5996% WER / 37 of 150 exact lines**. See [RESULTS.json](RESULTS.json), [RESULTS.csv](RESULTS.csv) and [ARTIFACTS.json](ARTIFACTS.json).

The inference checkpoints are committed with Git LFS in [arabic_ocr_best](../arabic_ocr_best/README.md): the new AR and both components of the old span merge. Run `git lfs pull` and the artifact verification script before using them. The old 48.77%-CER span merge is an algorithm/configuration plus an old AR/MDiff pair, not another independent neural checkpoint.

## Implement one matched experiment

Create a **new sibling package and entrypoint**, for example `thesis/khatt_mdiff_v1/` and `thesis/scripts/run_khatt_mdiff_v1.py`. Do not change the completed `khatt_ablation_v1`, `mdiff_arabic_v1`, `ar_mdiff_fusion_v2` source packages, their original launchers/tests, existing checkpoints, selections or validation gates. Their raw bytes are recorded in completed runs. Reuse their model and preprocessing through imports where appropriate.

The new model must be `Recognizer(..., arm='mdiff')`, with the following controls matched to the winning AR:

| Control | Required value |
|---|---|
| TRAIN | Exact existing `khatt100_ours.jsonl`, 12,304 rows, each visited once per epoch |
| Validation | Exact existing 150 children lines; no KHATT validation |
| Historical evaluation | Exact original 752 children lines, explicit user-run Test only |
| Vocabulary | Exact existing KHATT `tokenizer.json`; no rebuilding from validation/test |
| Initialization | Official pretrained SVTRv2-small encoder; fresh Arabic head; seed 54101 |
| Architecture | Capacity 144, visual projection 256, 3 decoder layers, 8 heads, dropout 0.1 |
| Images | RGB height 32, max width 2,196, alignment 4; aspect-preserving and no truncation |
| Training | Maximum 100 epochs; validation patience 10 |
| Optimizer | AdamW; decoder LR 3e-4, encoder LR 1e-5; weight decay 0.01 |
| Budget | Batch 1, gradient accumulation 16, two decoder calls per training example |
| Schedule | Warmup 5 epochs, cosine floor 0.1; encoder frozen for first 5 epochs |
| Other | Clip 1; same augmentation cycle and sample ordering seeds; fixed BN statistics; encoder gradient checkpointing |
| Precision | BF16 on the original CUDA machine; record any different hardware/environment explicitly |
| Selection | Min validation character edits, then word edits, then most exact lines, then earliest epoch |

Use the existing MDiff objective, not a new architecture: structured masked denoising and visible-token correction, equal weights 0.5 each, plus 0.1 tail-EOS loss. Production generation is image-only, with the existing three denoise/correction rounds (six decoder passes). Do not provide reference length, target tokens, or test labels to inference. Keep `no_harakat_v1`: remove only U+064B–U+0652; preserve other code points, logical order and whitespace.

**Do not initialize training from the new child-fine-tuned AR checkpoint or old MDiff epoch 177.** The controlled experiment must use the same official encoder initialization and fresh head as AR. The saved AR initialization SHA-256 is `40e18c0934a7f289748eb666dc478132e06b09331dac249ed61added5d0bc557`; the encoder-only hash is `c6fa52b3592b18971c80aa10c2f5db5a9c9cb3c73b302e595e03eb801ee6bfeb`. With the same dependency versions, seed, tokenizer and constructor, confirm the matching initial tensors before training. `arm` changes the loss/attention behavior, not the parameter shapes. If a different platform changes initialization numerics, document and resolve that difference rather than claiming exact equivalence.

Official encoder assets: `Diffusion/drive_all/pretrained/mdiff_s_v1/{best.pth,config.yml}`. Their expected hashes are in `ARTIFACTS.json`; the pinned Hugging Face revision and upstream source are documented in `thesis/mdiff_arabic_v1/vendor/NOTICE.md`.

## Data transfer and portability

The repository is public. Training images, annotations, per-line predictions and school/notebook sample identifiers are supplied **privately**, outside Git. Ammar's prepared training-input archive is `Diffusion/handoffs/khatt_mdiff_marwan_20261110/training_inputs_private.zip` on Ammar's machine. Ask Ammar to share it through the team's private channel; the GitHub issue does not publish it. The archive contains the exact merged TRAIN and children validation inputs, tokenizers/manifests, original official encoder files, and `path_remapping.json`. It deliberately excludes the 752 historical test labels/images; Ammar can run final historical evaluation on his existing data.

The prepared archive is **465,903,172 bytes**, SHA-256 `8b386932914f6a8124b253e2a3b27eff815a4decba2ea5ea20273c61a0ac7ec0`. Its internal transfer manifest SHA-256 is `93f60e50886a418fe05c666839ab3b5e183ac16130e9c22af05a97e6328e9d60`. ZIP CRC and all 15,418 included source-file hashes were checked during preparation. Historical test data are absent.

The existing manifests contain absolute Windows paths under Ammar's project. Your runner should accept explicit repository, project/artifact and input-root arguments. Verify the original manifest bytes/hashes before applying a recorded prefix remapping from `path_remapping.json` to the extracted image/label locations; verify each resolved file's stored hash. Do not silently regenerate splits, rename IDs, shuffle dataset membership, merge KHATT validation, add future Drive lines, or replace the tokenizer to accommodate another machine. A newly derived portable manifest needs its own hash and a recorded parent hash.

The archive's `transfer_manifest.json` identifies every included file and hash. The full local experiment's provenance and reference checkpoint hashes are listed in `ARTIFACTS.json`. Machine-specific absolute paths and old source binding IDs are evidence about the original AR run, not paths that should be copied into new code. Create a new MDiff binding/run name for the new machine and implementation; retain the parent AR/data provenance.

## Deliverables and acceptance

- Implement `Check`, `Train`, `ValidationStats`, explicit `Test` and `Stats` actions, with exact PowerShell commands for Ammar. Avoid an automatic Train-to-Test chain. Keep historical test references inaccessible to preparation, training and validation.
- Provide a CPU synthetic integration check using the real `Recognizer.generate()` return contract `(token_ids, metadata)`; earlier AR validation failed because the pair was not unpacked. Check complete EOS/denominators, unknown/overlong targets and image limits.
- Test resume with model + optimizer + RNG + early-stopping state and no extension past patience/max epochs. Save only the selected best and latest resume state, with per-epoch history/predictions retained. Consider saving completed training work before validation so a validation error does not lose an entire long epoch.
- Record source bytes, assets, environment, dataset identities, initialization, training examples/updates and inference settings. Reject material drift and explain the changed field. Avoid binding unrelated globally installed packages as mandatory training dependencies; retain the observed package inventory for audit.
- Run bounded verification and send the implementation/commands for review. **Do not launch long training or final model evaluation automatically**; Ammar runs them unless he separately authorizes you to launch them.
- Once trained, preserve the best MDiff with its tokenizer/config/hashes using Git LFS. Compare against new AR 29.92%, old MDiff 53.84%, and historical span 48.77% on the same 752 lines. Report CER, WER, exact lines, Sub/Ins/Del, coverage/EOS failures and damage to previously exact lines. A model need not be better merely because it is diffusion; retain an unsuccessful result.

## Follow-up hybrid

If the new MDiff has useful complementary errors, pair it with the **new AR epoch-40 checkpoint** from Git LFS and reuse the short-span merge proposal/scoring idea. Both need the new KHATT vocabulary. Recalibrate the new pair's alpha/margin on children validation only and bind a separate hybrid run. Do not reuse the old selector parameters as though they had been calibrated for the new models, share features across independently fine-tuned encoders, or tune on the 752 lines. This hybrid is a later comparison, not part of the first matched MDiff training deliverable.

## Interpretation

The 752 lines are historical development-used data, not a fresh held-out test. The 100% mixed AR also added 704 children TRAIN lines and changed vocabulary/schedule versus the old models; its improvement cannot be attributed solely to KHATT. The old AR/span models and their preserved tokenizers remain historical controls. The 10% KHATT trial was intentionally abandoned after epoch 16; its best epoch 15 is excluded from the four-run comparison and should not be restarted.
