# Transformer and diffusion for handwriting OCR: evidence review

Research date: 29 October 2026. This is a literature and source-code review, not an independent reproduction of published scores. No OCR model, runner, checkpoint, package environment, or gate was changed for this review.

## What the evidence supports

There are successful Transformer–diffusion recognizers. There are also successful recognizers trained with diffusion-generated handwriting. These are different methods and their results must be separated.

The clearest direct-recognition CER evidence found is GryphOne, on handwritten mathematics. MDiff4STR and IPAD provide particularly relevant controlled recognition ablations, but report word accuracy rather than CER. The strongest ordinary handwriting CER augmentation examples below mostly use CNN/recurrent recognizers, so they are not proof of a Transformer–diffusion recognition decoder.

I did not verify a released model demonstrating low CER on Arabic children's full handwritten lines with our Paddle-conditioned architecture. This is a boundary of this search, not a claim that no such work exists.

## 1. Direct recognition: a Transformer predicts text through diffusion

### GryphOne — closest released handwriting diffusion recognizer with CER

Sources: [paper v2](https://arxiv.org/html/2602.03370v2), Tables 1, 3, 4; [official code](https://github.com/JG1VPP/GryphOne); [released weights](https://github.com/JG1VPP/GryphOne/releases/tag/ECCV2026).

It uses a pretrained DINO ViT-S/8 image encoder and five Transformer decoder blocks. Masked text is refined repeatedly while attending to the image. Two independently masked views share a symmetric mutual-learning objective. Random remasking permits reconsideration beyond a confidence-only policy. Its symbol-aware tokenization specifically represents visible mathematical symbols and LaTeX structure.

Training uses approximately 230,000 handwritten plus 400,000 synthetic MathWriting samples, 60 epochs, AdamW, learning rate 1e-4, and reported batch size 32. Baselines are reimplemented in a matched setting; CROHME is evaluated without dataset-specific fine-tuning. Ten inference runs measure decoding variability, not ten independently trained models.

| MathWriting test comparison | CER % |
|---|---:|
| ICAL matched baseline | 6.03 |
| GryphOne, confidence mask-predict variant | 6.29 |
| GryphOne, one diffusion step | 6.02 |
| GryphOne, 10 steps | 5.55 |
| GryphOne, 50 steps | 5.51 |

The one-step versus multi-step comparison isolates inference refinement more cleanly than comparing unrelated systems. The paper explicitly identifies small-data training as a limitation. These are mathematical expressions represented as LaTeX, not Arabic text-line CER.

### MDiff4STR — strongest match to our masking and confident-error problems

Sources: [paper](https://arxiv.org/html/2512.01422v1), especially Table 2; [official implementation](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/mdiff4str); [weights/configs/logs](https://huggingface.co/topdu/MDiff4STR).

An SVTRv2 visual encoder conditions a six-block Transformer diffusion decoder. Training mixes mask patterns resembling inference and adds wrong visible tokens. A separate correction loss supervises valid tokens, including visible errors, alongside masked denoising. This addresses both training/inference mismatch and confidently wrong letters becoming permanent.

The English real-data recipe uses filtered Union14M, 40 epochs, batch 1,024 on four RTX 3090s, AdamW at 5e-4, and a one-cycle schedule. Chinese training uses 100 epochs. Evaluation covers common scene-text benchmarks, Union14M challenges and Chinese recognition.

| Matched paper ablation | Common word accuracy % | Union14M word accuracy % | Occluded scene text (OST) word accuracy % |
|---|---:|---:|---:|
| Autoregressive baseline | 96.88 | 87.34 | 81.03 |
| Vanilla diffusion, block low-confidence decoding | 96.42 | 85.42 | 79.93 |
| Structured-mask training, without replacement | 96.98 | 87.09 | 81.92 |
| Full method, with replacement | 97.30 | 88.44 | 84.25 |

Chinese results are reported separately in the paper. All entries here are word accuracy and must never be converted into CER.

### IPAD — diffusion contribution separated from ordinary iterative decoding

Sources: [paper](https://arxiv.org/html/2312.11923), Table 7; [public implementation](https://github.com/Xiaomeng-Yang/IPAD).

IPAD combines a ViT with a time-conditioned parallel Transformer decoder, masked categorical diffusion training, and five easy-first decoding iterations. Table 7 separates a one-pass baseline, iterative decoding without diffusion, and diffusion training.

| Table 7 configuration | English synthetic-trained accuracy % | English real-trained accuracy % | Chinese accuracy % |
|---|---:|---:|---:|
| Baseline | 92.00 | 95.35 | 80.56 |
| Easy-first decoding, no diffusion | 92.47 | 95.85 | 82.78 |
| IPAD | 93.25 | 96.39 | 84.08 |

The English schedules are 50 epochs on synthetic data or 150 on real data; Chinese training uses pretraining followed by fine-tuning. Images are cropped words, typically 32×128, with short maximum sequence lengths. This is not a full-line Arabic spacing experiment and the table does not provide CER.

The source contains a working training entry point and a five-iteration configuration, although its README has stale command references. No explicit license was detected in the pinned repository; publicly readable code is not automatically licensed for unrestricted reuse.

### DiffusionSTR — why sequence extent and token presence matter

Source: [paper](https://arxiv.org/abs/2306.16707), Tables 2–3. No official public implementation was verified.

A 12-layer ViT and six-block Transformer decoder perform discrete diffusion recognition. A second head identifies positions containing characters, addressing text extent rather than treating every output slot alike. Removing that head gives 63.4% average word accuracy; adding it gives 91.8%.

Training uses MJ+ST synthetic scene words, 20 epochs, batch 384, four A100s and AdamW at 1e-4. Results average four experiments. Its step ablation is revealing: 100 steps gives 17.6% word accuracy, 500 gives 86.1%, 1,000 gives 91.8%, and 4,000 drops slightly to 91.3%. More inference steps are not universally better.

The reported metric is word accuracy, not CER. Our code already has a character-presence auxiliary head; merely adding a similarly named head would not be a new contribution or a reproduction of this method.

## 2. Diffusion as training-data generation

These generators do not have a standalone recognition CER. The before/after comparison is the same downstream recognizer trained without/with generated images.

| Work and downstream setting | Without generated images: CER % | With generated images: CER % | What the result actually tests |
|---|---:|---:|---|
| CTIG-DM, IAM | 7.06 | 5.19 | CRNN trained on real plus diffusion handwriting |
| CTIG-DM, IAM, also geometric augmentation | 7.06 | 4.67 | Diffusion plus an additional augmentation intervention |
| DiffusionPen, IAM, repeated-run appendix | 5.16 ± 0.01 | 4.78 ± 0.07 | CNN–LSTM/CTC recognition, mean/std over three runs |
| DiffBrush, IAM, supplemental Table 8 | 5.78 | 4.62 | Real plus synthetic line images for a recurrent recognizer |
| Ricoh image-to-text Transformer | Not isolated | 6.24 | Transformer trained on real and mixed synthetic data; no causal diffusion-only ablation |

**CTIG-DM.** [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Conditional_Text_Image_Generation_With_Diffusion_Models_CVPR_2023_paper.pdf), Table 3. Its generator conditions on text, image and style, using condition dropout. AdamW uses 1e-4; diffusion batches are 256 on V100 hardware. Synthetic-data quantity is varied up to roughly one million samples, and recognition is measured on real held-out IAM/RIMES data. RIMES CER similarly changes from 4.05% to 2.53%, or 1.98% with additional geometric augmentation. No author-verified public repository was established; an independent similarly named implementation should not be presented as official.

**DiffusionPen.** [ECCV paper](https://arxiv.org/abs/2409.06065), main Table 3 and appendix Table C1; [official repository](https://github.com/koninik/DiffusionPen). Text and five writer-style examples condition latent diffusion. The style encoder is pretrained separately; the generator trains for 1,000 epochs, batch 320, A100, and samples with 50 DDIM steps. Generated samples augment real IAM training. The headline table gives 5.14→4.71 CER, but its baseline is taken from an earlier recognition study. The appendix's repeated matched runs, 5.16→4.78, are the safer comparison. Neither result is a Transformer text-diffusion decoder.

**DiffBrush.** [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Dai_Beyond_Isolated_Words_Diffusion_Brush_for_Handwritten_Text-Line_Generation_ICCV_2025_paper.pdf) and [supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Dai_Beyond_Isolated_Words_ICCV_2025_supplemental.pdf), Table 8; [official repository](https://github.com/dailenson/DiffBrush). Its generator targets whole handwriting lines. Multi-scale word/line content discriminators help preserve the supplied text. Training uses 800 epochs and eight RTX 4090s; sampling uses 50 DDIM steps. The real-plus-synthetic recognition result is 5.78→4.62 CER and 21.76→16.86 WER. A separately reported synthetic-data evaluation metric, DCER, is not this result. At the inspected commit, generation code and weights are available, but a full training entry point is absent.

**Ricoh.** [Industrial research report](https://jp.ricoh.com/-/Media/Ricoh/Sites/jp_ricoh/technology/techreport/47/pdf/RTR47a16.pdf). This is an actual image-to-text Transformer trained with real and generated handwriting, using CTC/attention alignment objectives. The experiment combines roughly 47,000 real and 600,000 synthetic samples, with 4,300 real validation samples, reporting 6.24% CER. Synthetic sources include rendered fonts, GANs and diffusion. Diffusion outputs are manually filtered. No Transformer-only CER or isolated diffusion contribution is supplied, so this cannot be used to manufacture a before/after result.

## 3. Closely related cursive-script evidence, with weaker reproducibility

[Diffusion-Enhanced NAT–BART for Urdu word recognition](https://doi.org/10.1109/ACCESS.2026.3721734), also [listed by the authors' university](https://arrow.tudublin.ie/engschelero/26/), is especially relevant to cursive scripts. It uses generated word images to train a Neighborhood Attention Transformer plus pretrained BART, rather than diffusing text during recognition. The accessible publisher text reports handwritten test CER 0.26→0.049 (26%→4.9%) and combined printed/handwritten/synthetic CER 0.016 (1.6%). Approximately 12,000 real word crops are manually checked; generated images are filtered for readability.

I could verify the publication and read publisher-indexed method/results text, but did not obtain an independently inspectable full PDF or an official implementation. Some prose is internally inconsistent about baseline table references and whether test error is below training error. Writer-disjoint splitting and the exact synthetic-data partition procedure remain unverified. These are reasons to treat its large gains as a research lead, not our reproduction target or an expected Arabic result.

## 4. Relevant editing work that is not diffusion

[LevOCR](https://arxiv.org/abs/2209.03594), [official code](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/LevOCR), explicitly predicts deletions, inserted placeholder counts, and replacement characters through a visual/text Transformer. Optimal edit actions come from Levenshtein dynamic programming. Training mixes synthetic corruption with states derived from model predictions, then jointly optimizes visual and edit losses after pretraining.

Its visual-only accuracy rises from 90.65% to 92.79% after two editing rounds, reaching 92.81% after three. Those are word accuracies, not CER. Importantly, its identity-input probe still returns only 96.91% exact accuracy when given the correct text: approximately 3.09% of correct inputs are damaged. Even a successful learned editor does not automatically preserve correct lines. This supports measuring damage explicitly; it does not justify applying an arbitrarily huge penalty until all edits disappear.

## 5. Repository availability and practical reuse

Star counts are API snapshots from this research date, not quality scores. Parent-repository stars are labelled accordingly. Source trees were inspected without installing or executing their code.

| Repository | Stars | License checked | Availability and relevance |
|---|---:|---|---|
| [OpenOCR / MDiff4STR](https://github.com/Topdu/OpenOCR) | 1,485 | Apache-2.0 | Direct text diffusion; training/evaluation, configs, weights and logs. Best first source for a character recognition adaptation. |
| [GryphOne](https://github.com/JG1VPP/GryphOne) | 2 | MIT | Direct handwriting diffusion; train/infer code, configs and released weights. Mathematics-specific representation needs adaptation. |
| [One-DM](https://github.com/dailenson/One-DM) | 576 | MIT | Handwriting generation from one style example; training/fine-tuning/inference code. An augmentation option, not a text recognizer. |
| [DiffusionPen](https://github.com/koninik/DiffusionPen) | 111 | MIT | Handwriting generation, style training and diffusion training. An augmentation option. |
| [DiffBrush](https://github.com/dailenson/DiffBrush) | 196 | MIT | Line generation and weights; inspected release lacks complete training entry point. |
| [LevOCR inside AdvancedLiterateMachinery](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/LevOCR) | 1,834, parent | Apache-2.0 | Visual/text editing with training scripts; not diffusion. |

IPAD has four stars and public training code, but no explicit license was detected. ABINet has a non-commercial custom license; it should not be described as unrestricted MIT software.

### Important implementation checks

1. **OpenOCR decoding selection:** the inspected base YAML sets `sampler_step: 3`, but leaves the decoder mode flags false. `MDiffDecoder.forward()` then chooses a single parallel pass. Setting a step count alone is insufficient to activate iterative inference. The published Hugging Face configuration also needs explicit decoding-mode verification. The README's requested `develop` branch returned 404 during this review. These are integration/reproduction issues to resolve, not proof the paper's experiments used the wrong mode.
2. **OpenOCR sequence policy:** the English configuration uses `use_space_char: false` and a word-oriented length limit. Reusing it unchanged would invalidate our full-line spacing task. The checkpoint config enables `train_all_layer: true`; training configuration and decoder sampling dimensions must be matched.
3. **GryphOne encoder and loss:** the released encoder actually loads `facebook/dino-vits8`. The loss code implements two cross-entropy paths plus symmetric KL with detached peer targets. These are more concrete references than merely giving a model an analogous name.
4. **GryphOne CER:** metric code sums edit distances and divides by summed reference lengths. This is an aggregate edit-rate calculation, with mathematics tokenization. The Arabic text policy still needs to remain explicit.
5. **IPAD:** actual `scripts/train_ipad.sh` and `train.py` exist despite stale README references. Its inspected model config uses five inference iterations. Public code without a license needs clarification before reuse.

Pinned commits and exact fetched files are recorded in `source_manifest.json`, `implementation_manifest.json` and each repository's `tree.json`. Selected PDFs, extracted text, and rendered table checks are retained alongside this report.

## 6. What differs from our current implementation

The frozen v2.2 aligned checkpoint config specifies a small patch encoder: two encoder blocks, three diffusion decoder blocks, width 128, four heads, capacity 110, and Paddle text conditioning with pointer copying. The repair checkpoint retains that base and already adds replacement corruption, span/drop conditioning, a clean-denoising term, and tail supervision. Its presence auxiliary head is also already present.

Local sources:

- [Frozen parent configuration](<C:/Users/ammar/OneDrive/Desktop/final project/Diffusion/drive_all/baselines/agreement_refinement_v2_2_frozen_20260918/aligned_checkpoint-0014/config.json>)
- [Repair configuration](<C:/Users/ammar/OneDrive/Desktop/final project/Diffusion/drive_all/runs/diffusion_repair_r2_s52043/checkpoint-0044/config.json>)
- [Prototype and reproduction notes](<C:/Users/ammar/OneDrive/Desktop/final project/.codex-pretrained-selection/thesis/DIFFUSION.md>)

| Difference | Consequence or testable concern |
|---|---|
| Small, locally trained image encoder versus pretrained DINO or stronger OCR visual encoders | The diffusion decoder may not receive sufficiently informative visual features. This is a hypothesis supported by the architecture gap, not a causal diagnosis from parameter count alone. |
| Roughly 2,100 fit lines versus hundreds of thousands of expressions or millions of scene words | More local epochs do not create the visual diversity supplied by large pretraining sets. Transfer or carefully validated extra data is more defensible than expecting 200 epochs to fix weak evidence. |
| Our repair revises a selected masked subset, while other paths can directly correct visible wrong tokens | Confidently wrong characters can survive. A replacement-noise probability alone does not establish equivalent correction training and inference. |
| Our current verifier chooses bounded patches after generation | It can exploit existing candidates but cannot supply arbitrary missing text absent from those candidates. Better candidate generation remains necessary. |
| Most published high accuracies use cropped words or mathematical expressions | Full Arabic lines, spaces, connected glyphs and children's spelling need a separate evaluation. English word preprocessing is unsuitable. |
| Repeated historical validation and test inspection | The old test is now a development-used benchmark. It is useful for comparison, but cannot serve as fresh evidence of generalization after more tuning. |

The v4 verifier's historical CER improvement, 58.56→55.91, is meaningful on those recorded outputs. It does not establish that our base diffusion recognizer is now strong, and it does not make CER values from IAM or MathWriting directly comparable.

## 7. A research-driven next experiment, not another gate adjustment

My recommendation is to keep v2.2 and v4 frozen as comparators and make the next development target a proper image-conditioned diffusion recognizer with a strong pretrained visual encoder. MDiff4STR is the most relevant licensed character-recognition implementation to adapt. GryphOne supplies an independent handwritten-recognition reference and useful ablation designs. Neither is a ready-to-run Arabic child-line checkpoint.

The sequence below is a proposed experiment design, not something already run or a claim from any one paper:

1. **Establish that the image encoder supports recognition.** Use the same transferred encoder, Arabic vocabulary, training images and normalization for an autoregressive or CTC control and the diffusion decoder. This determines whether diffusion improves an adequate visual baseline.
2. **Verify a small training subset before a long run.** Confirm the model can overfit genuine image/text pairs and recover controlled missing/replaced characters without length explosions. Audit EOS, padding, spaces, truncation and RTL ordering. Include image ablation/shuffling to test image dependence rather than Paddle copying or language completion.
3. **Reproduce one coherent diffusion recipe.** Start with MDiff4STR-style mask/noise/correction objectives and a verified iterative sampler. Do not simultaneously add word merging, scorers, multiple noise schemes and arbitrary penalties. Random-remasking mutual learning is a separate later ablation, not an extra unchecked ingredient.
4. **Use predeclared development comparisons.** Same data, encoder initialization and compute budget; baseline versus diffusion; then correction objective on/off and an economical 1/3/5/10-step inference comparison. Select checkpoints on fixed validation CER, with predeclared WER/exact-line constraints and damage reports. A 200-epoch maximum and patience 30 can be reasonable local limits; they are not evidence that a broken objective will eventually improve.
5. **Separate adaptation from a fresh final evaluation.** Group at least by source page/notebook and preferably writer where identities exist. No need to restart the full nested-fold project. Reserve genuinely unseen material if available, and train generators/styles only on the training side. If no untouched data exists, label further results as development results.
6. **Report outcomes transparently.** Aggregate unclamped CER, WER, exact lines gained/lost, substitutions/insertions/deletions, whitespace errors, output-length/EOS failures, changes per line, and runtime/memory. Use multiple training seeds when feasible and confidence intervals respecting writer/page grouping. CER already combines substitution, insertion and deletion counts; adding them as separate bonuses double-counts edits.

For synthetic augmentation, first audit Arabic glyph joining, dots, spacing, and literal label fidelity. Never normalize a child's unusual spelling into a dictionary word. Compare equal synthetic sample budgets with ordinary augmentation. The [Quo Vadis study](https://openaccess.thecvf.com/content/ICCV2025W/VisionDocs/papers/Pippi_Quo_Vadis_Handwritten_Text_Generation_for_Handwritten_Text_Recognition_ICCVW_2025_paper.pdf) provides useful counterevidence: style fidelity alone does not predict HTR improvement, and aggressive filtering can discard useful diversity or prevent convergence.

This research supports improving visual evidence and denoising training before adding more selector complexity. It does not supply a justified promise of low single-digit CER on our data.
