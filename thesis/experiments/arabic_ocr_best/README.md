# Selected Arabic OCR checkpoints

Three SHA-verified model checkpoints are committed with **Git LFS**, together with their exact tokenizers and [manifest](manifest.json). Dataset images, labels and optimizer states are not included.

| Artifact | Role | Selected epoch | Historical CER | Exact lines |
|---|---|---:|---:|---:|
| `transformer_khatt100_ours_e040.pt` | Best aggregate AR recognizer; 100% KHATT + children | 40 | 29.92% | 185/752 |
| `span_merge_ar_e093.pt` | AR component of the previous span merge | 93 | — | — |
| `span_merge_mdiff_e177.pt` | MDiff component of the previous span merge | 177 | — | — |

**Span merge has no separate neural weights.** Its preserved artifact consists of the last two checkpoints, `tokenizers/original.json`, the existing `thesis/ar_mdiff_fusion_v2` implementation, and the configuration in `manifest.json` (alpha 0.75, margin 0, bounded short-span proposals). The combination achieved **48.77% CER, 83.28% WER and 151/752 exact lines**. This is distinct from the older `span_hybrid_verifier_v4`, which achieved 55.91% CER.

The new Transformer uses `tokenizers/khatt.json`; the old span pair uses `tokenizers/original.json`. These vocabularies differ. Never load a checkpoint with the other tokenizer or feed features from one independently fine-tuned encoder to the other decoder.

## Download and verify

```powershell
git lfs install
git lfs pull --include="thesis/experiments/arabic_ocr_best/checkpoints/*.pt"
python -X utf8 thesis/scripts/check_best_arabic_ocr_artifacts.py
```

A roughly 130-byte checkpoint file is an LFS pointer, not the downloaded weights. Verification checks full file sizes and SHA-256 values. Each actual checkpoint is about 60 MB. The three files total 181,316,935 bytes.

For inference, use `Recognizer(vocab_size, capacity=144, arm='ar' or 'mdiff')` from `thesis/mdiff_arabic_v1/model.py`, load `saved['model']` strictly, and use the checkpoint's declared tokenizer and image configuration. Call `.eval()` and unpack `token_ids, metadata = model.generate(...)`. New AR image cap is 2,196; old span-pair cap is 640; all use RGB height 32, patch alignment 4 and aspect-preserving per-image resizing. These files do not require the original machine's dataset paths for loading the model, but the original full experiment runners also verify their locally bound datasets and source histories.

## Next hybrid

Marwan's task is the [matched KHATT MDiff experiment](../khatt_ablation_v1/MDIFF_HANDOFF.md). Its resulting MDiff should use the **new KHATT tokenizer**, allowing it to pair with `transformer_khatt100_ours_e040.pt`. Preserve the old span model as a historical comparator. Recalibrate the new pair's scorer/selector on children validation before historical evaluation; swapping checkpoint paths into the old sealed run is not a valid new hybrid.

All reported scores use the historical development-used 752 children lines, not a fresh untouched test set. Aggregate improvements have not established zero damage to previously correct lines or automatic model promotion.
