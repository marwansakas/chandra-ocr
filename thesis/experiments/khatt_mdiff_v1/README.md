# MDiff on 100% KHATT + children’s handwriting

The selected MDiff checkpoint reached **41.98% character error rate (CER)** on the 752 historical children’s lines. This is an **11.85 percentage point** improvement over the earlier MDiff (22.0% relative). The Transformer trained on the same mixture remains the strongest standalone model at **29.92% CER**. [Verified aggregate results](RESULTS.json)

## Training and selection

This was a single diffusion run on the exact training mixture used by the winning Transformer: **9,497 KHATT lines and 2,807 children’s lines**, shuffled together each epoch. Checkpoint selection used the same **150 children’s validation lines**. The dataset manifests, tokenizer, preprocessing and pretrained encoder initialization match the [AR experiment](../khatt_ablation_v1/README.md).

| Setting | Value |
|---|---|
| Visual encoder | Pretrained SVTRv2-small with stage depths 3, 6 and 3 |
| Decoder | Three Transformer blocks, width 256, eight heads, dropout 0.1, fresh Arabic weights, shared KHATT tokenizer |
| Input and output | Aspect-preserving RGB resize to height 32, width cap 2,196, patch alignment 4, output capacity 144 |
| MDiff objective | 0.5 masked denoising + 0.5 visible-token correction + 0.1 end-of-sequence loss |
| Inference | Three denoise/correction rounds (six decoder passes), from the image and masked tokens |
| Optimizer | AdamW, encoder LR `1e-5`, decoder LR `3e-4`, weight decay 0.01, gradient clipping at 1 |
| Schedule | Batch 1, accumulate 16, seed 54101, original augmentation cycle, five warm-up epochs, encoder frozen for the first five epochs, cosine decay with a 10% floor |
| Stopping | Maximum 100 epochs, patience 10, CUDA BF16 |

The best checkpoint was **epoch 73**. It scored **40.79% CER**, **83.51% WER**, and **27/150 exact (18.00%)** on validation. Training stopped at **epoch 83**. Selection minimized validation character edits, then word edits, then favored more exact lines and the earlier epoch. [Training runner](../../scripts/run_khatt_mdiff_v1.py)

## Historical comparison on 752 lines

| Model | CER ↓ | WER ↓ | Exact ↑ | Sub | Ins | Del | No EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer, 100% KHATT + ours | **29.92%** | **65.74%** | **185 (24.60%)** | 2,247 | 786 | 1,004 | 0 |
| Earlier MDiff | 53.84% | 91.87% | 100 (13.30%) | 5,031 | 596 | 1,637 | 0 |
| Earlier span merge | 48.77% | 83.28% | 151 (20.08%) | 4,160 | 743 | 1,678 | 0 |
| **New MDiff, 100% KHATT + ours** | **41.98%** | **85.21%** | **137 (18.22%)** | 3,622 | 906 | 1,137 | 0 |

The new MDiff made fewer substitutions and deletions than earlier MDiff and produced 37 more exact lines. Compared with the earlier span merge, it made fewer character edits but more word edits and got 14 fewer lines exact. Compared with the matched Transformer, it improved the character edit count on **173 lines**, worsened **363**, and tied **216**. MDiff got **30 lines exactly right that the Transformer missed**, while the Transformer got **78 lines exactly right that MDiff missed**. All four models completed all 752 lines.

The comparison uses the project’s historical development lines. The old-to-new MDiff comparison includes changes to training data, vocabulary and schedule. The comparison between the two models trained on the full mixture is the matched AR versus MDiff result.

## Checkpoint and reproduction

The [epoch-73 checkpoint](../arabic_ocr_best/checkpoints/mdiff_khatt100_ours_e073.pt) is stored through Git LFS. It uses the exact [KHATT tokenizer](../arabic_ocr_best/tokenizers/khatt.json) and the image settings in the [checkpoint manifest](../arabic_ocr_best/manifest.json).

```powershell
git lfs pull --include="thesis/experiments/arabic_ocr_best/checkpoints/*.pt"
python -X utf8 thesis/scripts/check_best_arabic_ocr_artifacts.py
```

Use `Recognizer(len(tokenizer), capacity=144, arm='mdiff')`, load `saved['model']` strictly, call `.eval()`, and unpack `(token_ids, metadata)` from `generate(..., steps=3)`. For the original run layout, place the archived `runners/run_khatt_mdiff_v1.ps1` under the project’s `Diffusion` folder. The wrapper expects the repository in the `.codex-pretrained-selection` checkout and the existing KHATT and children’s data, manifests, pretrained assets and cached comparison predictions at their recorded `Diffusion` paths. The original manifests contain local absolute paths. The Python runner and PowerShell wrapper preserve their original bytes for checkpoint resume. [Launcher archive](runners/run_khatt_mdiff_v1.ps1)

From the project root, with the original inputs in place:

```powershell
$runner = Join-Path (Get-Location) 'Diffusion/run_khatt_mdiff_v1.ps1'
& $runner -Action Check -Device cuda
& $runner -Action Train -Device cuda -Resume
# After training completes:
& $runner -Action Test -Device cuda
```

`-Resume` restores optimizer and random state. A saved pre-validation checkpoint lets validation restart without repeating its training epoch. The test action prints the historical table and saves a CSV and JSON report. The published results contain aggregate scores and provenance hashes.

## Next experiments

Both follow-up experiments below have been completed. The span hybrid using MDiff reached **29.19% historical CER**, and the hybrid using the `3e-4` fine-tune reached **28.93%**. [Fine-tune and hybrid results, decisions and checkpoints](../khatt_finetune_hybrids_v1/README.md)

1. **Try a new span hybrid.** Use the epoch-40 mixed Transformer as the default and this epoch-73 MDiff as a source of short, aligned correction candidates. Both use the same tokenizer. Carry over the existing [span proposal and image-conditioned scoring code](../../ar_mdiff_fusion_v2), and calibrate the acceptance margin and score weighting for this new pair using children’s validation. Freeze the settings before the comparison on the same 752 historical lines. Compare against both standalone models and the earlier span merge. Report CER, WER, exact matches and how many of the Transformer’s exact lines the hybrid loses as well as gains.
2. **Finish the KHATT-to-children Transformer fine-tunes.** Both fine-tunes start from the KHATT-only epoch-34 checkpoint and train on the same 2,807 children’s lines, with a fresh optimizer, encoder LR `1e-5`, maximum 100 epochs and patience 10. The only controlled difference is the decoder LR, `1e-4` versus `3e-4`. Compare both validation-selected checkpoints on the same 752 historical lines against the mixed Transformer at 29.92% CER. This measures whether sequential adaptation improves on training with the mixture. Use the full comparison to decide whether a fine-tuned AR is worth testing as the base of another hybrid.
