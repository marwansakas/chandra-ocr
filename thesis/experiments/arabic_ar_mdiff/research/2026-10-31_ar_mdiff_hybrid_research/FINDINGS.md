# Research: combining the trained Arabic AR and MDiff recognizers

Research date: 2026-10-31. This work inspected source code, checkpoint metadata/encoder tensors and existing reports, and reviewed primary publications. No recognition, training, calibration sweep or new model evaluation was executed. Proposals below are adaptations to investigate, not demonstrated improvements on this dataset.

## What the current result means

The historical comparison contains 752 lines, 13,493 reference characters and 2,793 reference words.

| System | CER % | WER % | Exact lines | Character edits | Word edits |
|---|---:|---:|---:|---:|---:|
| AR transformer | 51.40 | 84.21 | 136 | 6,936 | 2,352 |
| MDiff | 53.84 | 91.87 | 100 | 7,264 | 2,566 |
| Current whole-line hybrid | 51.13 | 84.57 | 123 | 6,899 | 2,362 |

Relative to AR, the hybrid removes 37 character errors (0.274 percentage points), adds 10 word errors, loses 17 previously exact lines and gains 4. Its edit-count change is substitutions +11, insertions unchanged, deletions -48. This is a tradeoff, not a clear overall improvement. Aggregate results do not establish the cause of each error.

The current selector uses 14 features describing output strings, EOS flags and image aspect ratio. It has no image embeddings, decoder probabilities or token uncertainty. It chooses one entire existing output; it cannot construct a line from the better portions of both. Importing incorrect text together with a useful correction is a plausible explanation for collateral damage, but would require a separately defined diagnostic to confirm.

The saved validation complementarity report shows some usable diversity: MDiff has fewer character errors on 50 lines, AR on 58, and they tie on 42. A reference-dependent whole-line oracle would reduce validation CER from AR's 47.47% to 42.67%, with 34 exact lines instead of 29. This is an upper bound for those two saved candidates, not an implementable result or a forecast. These validation labels were already used during development and checkpoint selection.

## Encoder correction

The models share the SVTRv2 architecture and pretrained initialization, but their selected checkpoints do **not** share final encoder parameters. Both started with encoder hash `c6fa52b3592b18971c80aa10c2f5db5a9c9cb3c73b302e595e03eb801ee6bfeb`. The final AR encoder hashes to `4ac4264133c860c4f1c070416fb134c2b02e3c4dfe9b47509e7fdc15674e4d8c`; MDiff hashes to `fa48882969185b0f9597e952f48e08166b7e2eca87fc27ffa5a692f301d63704`. Hash conventions match the initialization calculation. Full checkpoint hashes were checked against their saved selections; details are in `encoder_identity.json`.

Consequently, frozen-model fusion must cache each model's own visual memory. Feeding AR visual memory directly into the trained MDiff decoder, or averaging their encoder weights, would introduce an unvalidated model change.

## Recommended first experiment: both models score the same candidates

**Published evidence.** Wick et al. combine CTC evidence with sequence-to-sequence decoding for handwritten text lines, penalizing inconsistent beam paths. This supports using complementary image-conditioned decoder evidence rather than a string-only selector; their CTC method is not directly our MDiff method. [Rescoring Sequence-to-Sequence Models for Text Line Recognition with CTC-Prefixes](https://arxiv.org/abs/2110.05909).

Salazar et al. score masked language models using pseudo-log-likelihood: hide each token before scoring it, then aggregate conditional scores. They demonstrate hypothesis rescoring for speech recognition and translation, not Arabic OCR. [Masked Language Model Scoring](https://aclanthology.org/2020.acl-main.240/).

**Proposed adaptation.** Keep both checkpoints frozen. Start with exactly the existing AR and MDiff strings. For each candidate:

1. Compute AR teacher-forced log probability, including EOS and excluding padding, against AR's image features.
2. Compute an image-conditioned MDiff masked pseudo-score: hide a candidate token before asking MDiff to predict it, repeat over active positions, and include a defined EOS/boundary check. The noise-state convention must match the trained masked objective. Treat masking pattern and time input as part of the fixed recipe.
3. Combine the two scores with explicit length treatment and a small, predefined development-only weight/margin grid. Retain the AR candidate in the pool.

Do not score a fully visible candidate with the correction head and call that likelihood: copying the supplied token can inflate the score. MDiff pseudo-likelihood is also not a normalized sequence probability and is not directly comparable to AR log probability. Their scales and length effects require calibration. A high score is not proof of correctness.

After the two-candidate control, add a small AR beam, for example five candidates, and compare against AR beam search alone. This distinguishes better search from a benefit attributable to MDiff. Neither scoring stage requires weight updates, although better adaptation could eventually require training. It does require fresh GPU forward passes: existing prediction caches contain no logits.

Exact token-by-token masked scoring costs roughly one decoder pass per candidate token. Cache separate visual memories, score in small batches, and measure actual latency/peak CUDA memory. It could be much slower than today's selector; no speedup is claimed.

## Best actual merge: local alternatives followed by whole-line verification

**Published evidence.** LevOCR refines text using visual features and learned insertion/deletion operations, allowing sequence length to change. Its experiments concern scene text recognition; they do not establish performance on our full-line Arabic data. [Levenshtein OCR](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3820_ECCV_2022_paper.php).

SpecRef starts masked diffusion from an AR draft and selectively masks uncertain tokens. It also documents damage to already-correct tokens. Its evaluations are code/reasoning/language benchmarks, not OCR. [Speculative Refinement](https://aclanthology.org/2026.gem-main.33/).

**Proposed adaptation.** First use character alignment to identify short disagreement spans in logical Unicode order. Preserve the AR output as one complete candidate, and propose a bounded number of alternatives that substitute only an aligned MDiff span. Alignment must represent insertions and deletions explicitly; equal character positions across different strings are not a valid correspondence. Include boundary/space alternatives deliberately and enforce the original normalization and capacity rules.

Score each resulting **whole line** using both models as above. A useful span can then survive without importing the rest of MDiff's line. Candidate enumeration must be bounded; unrestricted combinations grow exponentially and introduce many implausible strings. Verification against the image remains necessary because a splice can disrupt word boundaries or context.

A later, separate extension can mask those suspect spans in the AR draft and ask MDiff to regenerate them with its own visual features. Use disagreement plus uncertainty rather than uncertainty alone, since confident predictions can still be wrong. Preserve outside-span tokens explicitly after every refinement step: the current MDiff sampler includes an all-visible correction pass that can otherwise alter them. Agreement is a heuristic anchor, not a correctness guarantee.

The existing `initial` input is only a diagnostic hook, not a validated local editor. Variable-length repair, EOS handling, aligned anchors and a realistic AR-draft corruption distribution need their own implementation and checks. If frozen MDiff cannot handle these draft states, train an adapter or refinement head on training-only drafts. Do not assume the training-free SpecRef result transfers to our network.

## Higher-cost option: actually share the encoder during training

PARSeq demonstrates that a recognizer trained with permutation language modeling can support AR decoding and bidirectional refinement using shared parameters. It is an architectural/training precedent, not permission to remove the causal mask from our already-trained AR model. [Scene Text Recognition with Permuted Autoregressive Sequence Models](https://arxiv.org/abs/2207.06966).

One controlled new experiment would freeze the selected AR encoder and AR head, then train an MDiff head plus a small adapter to operate on that encoder's visual features. This creates genuine encoder sharing while preserving a fixed AR baseline. A later joint-training variant could optimize causal AR, masked reconstruction and conservative correction objectives together. Compare it separately: joint gradients can reduce useful diversity or hurt the stronger decoder. This is more work than frozen rescoring and should follow evidence that complementary decoder evidence helps.

A recent controlled Arabic HTR study compares CTC, attention and their hybrids with a common encoder architecture and protocol. Its conclusions caution that decoder combinations do not automatically fix shared visual/boundary errors. Its datasets, backbone and metric procedures differ from ours; headline scores are not directly comparable. [A controlled study of CTC, attention, and hybrid models for Arabic handwritten text-line recognition](https://link.springer.com/article/10.1007/s44443-026-00970-6).

## Concrete order and acceptance evidence

My recommendation is to build the image-conditioned scoring layer first, then bounded span proposals. Use separate ablations rather than combining all changes at once:

| Variant | What it establishes |
|---|---|
| Existing AR greedy | Fixed baseline |
| AR beam only | Gain from broader AR search |
| Existing two strings, dual scoring | Whether MDiff adds useful verification evidence |
| AR beam plus MDiff, dual scoring | Candidate diversity beyond AR search |
| Bounded span alternatives, dual scoring | Benefit from combining parts of both predictions |
| Local masked regeneration | Benefit beyond splicing existing text; implement only after earlier evidence |

Report CER, WER, exact lines, exact lines lost/gained relative to AR, S/I/D, spacing errors, coverage/EOS, latency and peak CUDA memory. Use a preservation probe containing correct AR drafts. Lower CER alone must not conceal the failure seen here; define acceptable WER and exact-line behavior before choosing settings. A development preservation result cannot guarantee zero damage on unseen lines.

Use training-derived predictions from genuinely held-out groups or new development data for learning/calibration where feasible. Notebook/page grouping is only a proxy when writer identities are unverified. The existing 150-line validation and 752-line historical comparison have both been used for development; neither is now a fresh final benchmark. Do not select thresholds from the historical test errors. Preserve literal children's spelling and the current `no_harakat_v1` policy; a dictionary or external language model is not part of this proposal.

The research deliverable changes no frozen source, checkpoint, selector, gate or experiment output. Any implementation should be isolated as a new version. Training/testing commands should be supplied for the user to execute, following the user's stated preference.
