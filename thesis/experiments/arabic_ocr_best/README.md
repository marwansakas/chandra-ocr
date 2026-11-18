# Selected Arabic OCR checkpoints

Seven SHA-verified model checkpoints are committed with **Git LFS** together with their exact tokenizers and configuration. [Checkpoint manifest](manifest.json)

| Artifact | Role | Selected epoch | Historical CER | Exact lines |
|---|---|---:|---:|---:|
| `transformer_khatt100_ours_e040.pt` | Best aggregate AR recognizer; 100% KHATT + children | 40 | 29.92% | 185/752 |
| `mdiff_khatt100_ours_e073.pt` | Matched MDiff recognizer trained on 100% KHATT + children | 73 | 41.98% | 137/752 |
| `transformer_khatt_only_e034.pt` | Starting checkpoint for both children-only fine-tunes | 34 | 73.84% | 14/752 |
| `transformer_finetune_lr1e4_e029.pt` | Children’s fine-tune with decoder LR `1e-4` | 29 | 30.25% | 173/752 |
| `transformer_finetune_lr3e4_e052.pt` | Children’s fine-tune with decoder LR `3e-4` and source of span corrections | 52 | 31.16% | 183/752 |
| `span_merge_ar_e093.pt` | AR component of the previous span merge | 93 | — | — |
| `span_merge_mdiff_e177.pt` | MDiff component of the previous span merge | 177 | — | — |

The preserved span merge uses the `span_merge_*` checkpoints, `tokenizers/original.json`, the existing `thesis/ar_mdiff_fusion_v2` implementation, and the configuration in `manifest.json` (alpha 0.75, margin 0, bounded short-span proposals). The combination achieved **48.77% CER, 83.28% WER and 151/752 exact lines**. This is distinct from the older `span_hybrid_verifier_v4`, which achieved 55.91% CER.

Use `tokenizers/khatt.json` for the KHATT models and fine-tunes, and `tokenizers/original.json` for the old span pair. Run each checkpoint with its own encoder weights.

## Download and verify

```powershell
git lfs install
git lfs pull --include="thesis/experiments/arabic_ocr_best/checkpoints/*.pt"
python -X utf8 thesis/scripts/check_best_arabic_ocr_artifacts.py
```

Verification checks the downloaded file sizes and SHA-256 values. Each checkpoint is about 60 MB. The seven files total 423,757,507 bytes.

For inference, use `Recognizer(vocab_size, capacity=144, arm='ar' or 'mdiff')` from `thesis/mdiff_arabic_v1/model.py`, load `saved['model']` strictly, and use the checkpoint's declared tokenizer and image configuration. Call `.eval()` and unpack `token_ids, metadata = model.generate(...)`. The KHATT pair uses a width cap of 2,196 and the old span pair uses 640. Both use RGB height 32, patch alignment 4 and aspect-preserving per-image resizing. The original experiment runners verify their locally bound input manifests and source histories.

## Completed KHATT span hybrids

The new hybrids use `transformer_khatt100_ours_e040.pt` as the base. Pair it with `mdiff_khatt100_ours_e073.pt` for **29.19% historical CER**, or `transformer_finetune_lr3e4_e052.pt` for **28.93% historical CER**. Use the frozen score weighting and acceptance margins for each pair. [Fine-tune and hybrid comparison](../khatt_finetune_hybrids_v1/README.md) · [Frozen settings](../khatt_finetune_hybrids_v1/HYBRIDS.json)

All reported scores use the historical development-used 752 children lines, not a fresh untouched test set. Aggregate improvements have not established zero damage to previously correct lines or automatic model promotion.
