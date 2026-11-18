# KHATT fine-tuning and span hybrids

The **mixed Transformer + `3e-4` fine-tune hybrid** had the lowest character error rate (CER) in this comparison at **28.93%**. The **mixed Transformer + MDiff hybrid** reached **29.19%**, and the mixed Transformer alone reached **29.92%**. Both hybrids improved on the mixed Transformer for CER, word error rate (WER) and total exact matches. [Aggregate results](RESULTS.json)

## Transformer fine-tuning

Both trials started from the **KHATT-only Transformer at epoch 34** with a fresh optimizer. Each used the same **2,807 children’s training lines** and **150 children’s validation lines**. The controlled difference was decoder base LR (`1e-4` vs. `3e-4`). Both used encoder base LR `1e-5`, maximum 100 epochs, patience 10, the shared tokenizer, and the same preparation and training schedule. Best epochs were selected on validation by character edits, then word edits, then exact matches and epoch order.

| Decoder LR | Selected epoch | Stopped at | Validation CER | Validation WER | Exact / 150 |
|---|---:|---:|---:|---:|---:|
| `1e-4` | 29 | 39 | 26.72% | 60.60% | 49 |
| `3e-4` | 52 | 62 | 28.85% | 59.31% | 43 |

The `1e-4` fine-tune gave the better standalone historical CER (**30.25% vs. 31.16%**). The `3e-4` fine-tune gave the better WER (**63.30% vs. 64.80%**), and it got **47 lines exactly right that the mixed Transformer missed**, compared with **40** for `1e-4`. We chose `3e-4` as the first source of span corrections for those reasons. That choice drew on the 752-line historical results. The `1e-4` checkpoint remains available for a future comparison as a correction source. [Saved checkpoints](../arabic_ocr_best/manifest.json)

## Hybrid setup and validation

The base for both variants was the **mixed Transformer at epoch 40**. The correction sources were **MDiff epoch 73** and **`3e-4` fine-tune epoch 52**. The runner aligned each source’s saved transcription against the base. It proposed bounded replacements of short disagreement spans and scored each candidate against the image using both component models. The AR scorers use mean token log probabilities and the MDiff scorer masks each token before scoring it. Each selected replacement had to exceed the base transcription by the calibrated score margin. [Hybrid runner](../../scripts/run_khatt_span_hybrids_v1.py)

The calibration grid searched correction-source weights and acceptance margins on the 150 children’s validation lines. Both variants found settings with lower CER, lower WER, full EOS coverage and zero previously exact base lines lost. The historical evaluation used the frozen settings below. [Frozen settings](HYBRIDS.json)

| Model | CER | WER | Exact / 150 | Sub | Ins | Del | No EOS | Exact gained | Exact lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mixed Transformer e40 | 26.72% | 60.60% | 37 | 379 | 104 | 157 | 0 | — | — |
| Mixed MDiff e73 | 40.79% | 83.51% | 27 | 639 | 168 | 170 | 0 | — | — |
| `3e-4` fine-tune e52 | 28.85% | 59.31% | 43 | 411 | 75 | 205 | 0 | — | — |
| Transformer + MDiff | 26.10% | 59.74% | 40 | 370 | 102 | 153 | 0 | 3 | 0 |
| Transformer + `3e-4` fine-tune | **25.30%** | **56.96%** | **45** | 355 | 94 | 157 | 0 | 8 | 0 |

| Variant | Correction-source weight | Acceptance margin |
|---|---:|---:|
| Transformer + MDiff | 0.50 | 0.10 |
| Transformer + `3e-4` fine-tune | 0.25 | 0.05 |

## Historical comparison on 752 lines

Exact gains and losses are relative to the mixed Transformer. The rows combine the two completed comparison reports with the same line membership and metric policy.

| Model | Epoch | CER | WER | Exact / 752 | Sub | Ins | Del | No EOS | Exact gained | Exact lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KHATT-only Transformer | 34 | 73.84% | 109.63% | 14 | 5,103 | 3,168 | 1,692 | 0 | — | — |
| Mixed Transformer | 40 | 29.92% | 65.74% | 185 | 2,247 | 786 | 1,004 | 0 | — | — |
| Earlier span merge | — | 48.77% | 83.28% | 151 | 4,160 | 743 | 1,678 | 0 | — | — |
| Mixed MDiff | 73 | 41.98% | 85.21% | 137 | 3,622 | 906 | 1,137 | 0 | — | — |
| `1e-4` fine-tune | 29 | 30.25% | 64.80% | 173 | 2,139 | 648 | 1,294 | 0 | 40 | 52 |
| `3e-4` fine-tune | 52 | 31.16% | **63.30%** | 183 | 2,166 | 654 | 1,385 | 1 | 47 | 49 |
| Transformer + MDiff | 40 + 73 | 29.19% | 64.73% | 194 | 2,167 | 770 | 1,002 | 0 | 13 | 4 |
| Transformer + `3e-4` fine-tune | 40 + 52 | **28.93%** | 63.41% | **201** | 2,166 | 737 | 1,000 | 0 | 20 | 4 |

The mixed Transformer remains the best standalone model by CER. The `3e-4` hybrid has the strongest combined CER and exact-line result, while `3e-4` standalone has the lowest WER. Both hybrids each lost four exact lines on the historical comparison despite zero losses in calibration. These results measure performance on historical development lines, including lines used to choose `3e-4` as the initial donor. A direct donor comparison with `1e-4` remains open.

## Weights and reproduction

The checkpoint bundle includes the **KHATT-only starting point e34**, **fine-tune e29**, and **fine-tune e52** through Git LFS alongside the previously published **mixed Transformer e40** and **mixed MDiff e73**. Each hybrid consists of its component checkpoints and frozen settings. [Checkpoint bundle](../arabic_ocr_best/README.md)

The archived PowerShell launchers run from the original project layout. Put them under `Diffusion`, with the repository in `.codex-pretrained-selection` and the original dataset manifests and cached baseline predictions at their recorded paths. Use the bundle for checkpoint transfer and the local run artifacts for the original comparison runner.

```powershell
$finetune = Join-Path (Get-Location) 'Diffusion/run_khatt_ar_finetune_v1.ps1'
& $finetune -Action TrainAll -Device cuda -Resume
& $finetune -Action Test -Device cuda

$hybrid = Join-Path (Get-Location) 'Diffusion/run_khatt_span_hybrids_v1.ps1'
& $hybrid -Action Calibrate -Device cuda
& $hybrid -Action Test -Device cuda
```
