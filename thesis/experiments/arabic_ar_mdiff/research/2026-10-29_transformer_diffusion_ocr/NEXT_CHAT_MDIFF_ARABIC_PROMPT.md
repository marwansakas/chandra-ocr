Implement and prepare our next Arabic handwriting diffusion recognition experiment. Finish the implementation, perform bounded verification, and give me the exact PowerShell commands to run training, validation, and the gated comparison. Do not stop at a proposal. Do not launch the long training or final evaluation yourself; I will run those commands and paste the results back.

## Objective and boundaries

Our main research method is Transformer-based diffusion recognition of Arabic children's handwritten lines. Paddle is a frozen comparator and potentially a future conditioning source; this task is not another Paddle fine-tuning project.

Adapt one coherent MDiff4STR-style recognition method with a strong pretrained visual encoder. Establish a matched autoregressive Transformer baseline using the same visual encoder, Arabic vocabulary, data, preprocessing, and comparable training budget. We need to distinguish improvement from the visual encoder from improvement contributed by diffusion.

Prepare these two primary arms first. A diffusion-without-correction ablation can be supported as a separately named, predefined follow-up; do not automatically run a large sweep. Keep word merging, new verifiers, scorer ensembles, image synthesis, and GryphOne mutual learning for later. Do not pile unrelated changes into this experiment.

MDiff4STR's published checkpoints are English/Chinese, with no verified official Arabic checkpoint. This is an Arabic adaptation, not an already proven Arabic solution. Do not promise its published word accuracy or single-digit CER on our dataset. Do not call an adaptation a faithful reproduction unless the differences are documented.

## Workspace and existing evidence

Windows project root:
C:\Users\ammar\OneDrive\Desktop\final project

Code repository:
C:\Users\ammar\OneDrive\Desktop\final project\.codex-pretrained-selection

Research report to read first:
C:\Users\ammar\OneDrive\Desktop\final project\research\2026-10-29_transformer_diffusion_ocr\FINDINGS.md

That research folder contains primary PDFs, extracted text, table checks, pinned repository trees, selected source files, source_manifest.json and implementation_manifest.json. Read the relevant methods and code, and verify any upstream information you need to use. Do not repeat the whole literature search.

Also inspect applicable AGENTS.md files, the working tree status, and these local sources before editing:

- .codex-pretrained-selection\thesis\DIFFUSION.md
- .codex-pretrained-selection\thesis\diffusion\model.py
- .codex-pretrained-selection\thesis\diffusion\repair_diffusion.py
- .codex-pretrained-selection\thesis\diffusion\images.py
- .codex-pretrained-selection\thesis\diffusion\tokenizer.py
- .codex-pretrained-selection\thesis\diffusion\metrics.py
- .codex-pretrained-selection\thesis\scripts\train_diffusion_line_ocr_model.py
- Diffusion\run_diffusion_repair_r2.ps1
- Diffusion\run_span_hybrid_verifier_v4.ps1
- Diffusion\run_span_hybrid_verifier_v4_1.ps1
- Diffusion\drive_all\runs\span_v4_1_development_diagnosis_20261025\diagnosis.json
- adversarial_arabic_handwriting_ocr_audit.md, if present; distinguish historical concerns from problems already fixed.

Authoritative existing data locations, to verify rather than recreate:

- Diffusion\drive_all\ground_truth_lines
- Diffusion\drive_all\splits_70_05_25_v3
- Diffusion\drive_all\paddle_70_05_25_v3
- Diffusion\drive_all\baselines\agreement_refinement_v2_2_frozen_20260918
- Diffusion\drive_all\runs\span_hybrid_verifier_v4_s53004
- Diffusion\drive_all\runs\diffusion_repair_r2_s52043\checkpoint-0044

The historical split has 2,103 train, 150 validation and 752 test lines; verify IDs and counts from manifests. Compare models on identical IDs, images, references and text policy. A basename or line count alone is insufficient to establish matching data.

## What happened previously

Our frozen v2.2 hybrid and v4 verifier are historical comparators. Reported historical 752-line results:

Model                         CER %   WER %  Exact %   Sub   Ins   Del
paddle                        59.39   92.84     6.52  1463   104  6446
frozen_v2_2                   58.56   93.02     6.25  2704   111  5086
span_hybrid_verifier_v4       55.91   93.09     6.25  2626   117  4801

These are context from an already inspected benchmark. Recalculate only from verified matching artifacts when an authorized evaluation runs; do not invent a fresh reproduction claim.

The stronger penalties in v4.1 did not pass the development gate and fell back to the existing output. Identical fallback scores are not evidence of a better recognizer.

Our original diffusion base uses a small patch encoder: two encoder blocks, three decoder blocks, width 128. R2 already adds replacement corruption, conditioning span dropout, clean-denoising/tail supervision, and an existing presence head. Simply adding similarly named features again is not an evidence-based improvement. Inspect actual training and inference behavior.

## Research references and integration traps

Primary references:

- MDiff4STR paper: https://arxiv.org/html/2512.01422v1
- Official code: https://github.com/Topdu/OpenOCR
- Published weights/configs/logs: https://huggingface.co/topdu/MDiff4STR
- GryphOne, supporting handwriting reference: https://arxiv.org/html/2602.03370v2 and https://github.com/JG1VPP/GryphOne

MDiff4STR uses a strong visual encoder, structured corruption, wrong-visible-token replacement, masked denoising and a separate correction objective. Its central findings concern mismatch between training and inference and confidently wrong characters escaping remasking. Its main tables report word accuracy, not CER. Diffusion image generation is a different method from text diffusion recognition.

The prior source review found concrete integration issues to check:

1. In the inspected OpenOCR decoder, leaving all inference-mode flags false selects one parallel pass even when sampler_step is 3. Explicitly verify the active iterative path and count actual decoder passes. Setting a number in YAML does not prove diffusion refinement runs.
2. The English configuration disables spaces and limits text to 25 characters. Both are unsuitable for our full-line Arabic task.
3. Label generation's sample count/train_all_layer settings must match the decoder's sample_k dimensions. Match checkpoint-specific configuration, not a mixture of incompatible examples.
4. The README referenced a develop branch that returned 404 in the previous review. Pin a real commit and a reproducible checkpoint/config pair; do not blindly repeat that command.
5. Encoder transfer must actually load compatible pretrained weights. Log source, revision, hashes and exact loaded/skipped parameters. Do not silently substitute a random encoder or load incompatible English output heads as Arabic heads.

Prefer a practical SVTRv2 encoder transfer from official weights. If that cannot be done cleanly, explain the concrete compatibility issue and choose one justified, licensed pretrained encoder. Keep that choice fixed across both primary arms. Preserve third-party licenses and attribution. Document adaptations and CPU/GPU memory implications.

## Arabic text and image rules

- Recognize complete lines, including spaces, punctuation, digits and literal children's spelling. Do not silently turn this into cropped-word recognition.
- Use logical Unicode order. Do not reverse labels just because Arabic is RTL. Any image-coordinate transformation must be explicit, consistent, and verified.
- Keep the existing no_harakat_v1 evaluation policy: remove only U+064B–U+0652. Preserve tatweel, spelling, punctuation and whitespace. Do not introduce additional Unicode normalization, character folding, whitespace collapse or dictionary correction.
- Inspect the existing tokenizer and label-file newline policy. Exclude only the permitted terminal file newline; do not broadly strip meaningful content.
- Use an explicit approved vocabulary or construct it from training data only. Report unsupported characters; do not silently drop them or replace all unknowns with an apparently valid answer.
- Adapt embeddings/output heads to Arabic and special tokens correctly. Account explicitly for PAD, MASK, EOS, loss masks and decoded length.
- Choose sequence capacity and image geometry from training-side evidence with a documented margin. Preserve aspect ratio, mask padded image regions, and handle pretrained positional embeddings correctly. Do not squash long lines into a word-sized rectangle.
- Report overlength and image-limit problems. Do not silently truncate ground truth, discard difficult validation examples, or shrink evaluation coverage until scores improve.

## Model and experiment requirements

The first diffusion arm should recognize from the image without depending on Paddle guesses. This establishes whether the recognizer itself works. Keep Paddle conditioning as a later separately identified ablation; do not replace this experiment with another copy gate.

Implement actual mask diffusion denoising and correction of wrong visible characters. Include correct visible characters in the appropriate supervision so preservation is learned. Document the noising schedule, timestep handling, structured masks, replacement process, objectives, EOS/length handling and inference update rules.

Use the same pretrained encoder initialization, data and augmentations for the matched autoregressive control. Save actual encoder initialization hashes for both. Keep training-budget accounting explicit; merely matching epoch count does not establish equal compute. Label architectural or parameter-count differences honestly.

Default long-run schedule: maximum 200 epochs, early stopping patience 30. Choose justified learning rates, warmup, regularization and encoder fine-tuning schedule once before the run. Save per-epoch validation metrics and enough checkpoints to verify selection. Do not silently train only a handful of epochs or choose the final epoch because it is latest.

Target native Windows and the RTX 4050. Verify available VRAM rather than assuming it. Use small batches, gradient accumulation, suitable mixed precision and workers=0 initially if needed. Record effective batch size and any deviations from the paper's much larger hardware budget.

Add these bounded diagnostics before long training:

- Round-trip tokenizer, spaces, EOS/PAD, variable lengths and representative Arabic labels.
- Real minibatch forward/backward with finite losses and nonzero gradients in intended modules.
- Verify pretrained weights really loaded and both arms share the intended initialization.
- Verify multi-step inference actually executes the intended updates; deterministic decoding settings should reproduce outputs when applicable.
- A configurable tiny training-subset overfit check, with a written success criterion and strict update/time budget. Report exact match and CER, not teacher-forced loss alone.
- Recovery of controlled missing/replaced characters using training-derived examples, and preservation when input tokens are already correct.
- Normal-image versus shuffled/blank-image diagnostics on training-derived examples. Treat these as dependency checks, not perfect proofs of grounding or excuses to tune on audit errors.

Do not launch full training or final evaluation during implementation. Short smoke checks are authorized. Provide the command for the longer overfit diagnostic if it exceeds the bounded verification budget. If a readiness check fails, fix the implementation where justified; never weaken the check just to print passed=true.

## Evaluation, exact-line protection and test discipline

Use aggregate, unclamped CER:
CER = (substitutions + insertions + deletions) / total reference characters.

CER already includes those three edit types. Show their breakdown, but do not add separate rewards for fewer deletions that double-count the same edits or conceal increased insertions/substitutions. WER and exact-line outcomes are important secondary measures.

For every model, expose raw predictions and raw metrics. If an optional compatibility fallback is offered, score it as a separate named output, alongside the rejected raw model. Do not make a weak model appear successful by reporting only unchanged v2.2/v4 output.

Checkpoint selection and eligibility must be separate:

- Select the best development checkpoint by a fixed, written validation CER rule, with deterministic secondary tie-breakers.
- Assess promotion using a policy frozen before the long run. Default conservative policy: strictly fewer character edits and no more word edits than frozen v4 on matched validation IDs, plus zero baseline-exact lines damaged among the exact lines from v4 or v2.2. Record every condition separately.
- Explicitly count baseline-exact lines lost and new exact lines gained. A matching total exact count can hide damage; do not report only the net count.
- My priority is to heavily discourage damaging correct answers. Use correct-state training supervision and the preservation eligibility check. Do not use validation/test truth to determine inference actions, and do not make arbitrary giant loss weights the central new method.
- A model can improve CER and still fail promotion. Report that distinction. Do not alter the policy after seeing which epoch almost passes.
- No automatic promotion or replacement of any existing best model.

The 150-line validation and historical 752-line test have been inspected repeatedly. They are development-used benchmarks. No random reshuffle makes them untouched again.

Preserve canonical manifests for historical comparison. Audit page/notebook/writer overlap, and use a fixed training-derived holdout for internal debugging if needed, keeping groups together. Do not restart the nested Paddle fold project; I explicitly stopped that direction. If genuinely untouched grouped material exists, reserve it under a new immutable manifest for a later final evaluation. If not, state that limitation.

Do not use test references for training, vocabulary discovery, checkpoints, inference settings, thresholds, early stopping, candidate selection or troubleshooting. The implementation verification must not run test inference. After validation, the Test action must verify a true frozen gate, hashes and selected checkpoint before doing the requested historical comparison. A failed gate means STOP. Historical evaluation must be labelled exploratory/development-used, not a fresh final generalization result.

Compare Paddle, frozen v2.2, frozen v4, the matched Transformer baseline and the raw Arabic MDiff adaptation on identical samples. Cache baselines without modifying them. For any newly reserved evaluation material, generate comparator predictions with frozen model/policy artifacts and verify provenance; old predictions on other IDs are not a valid comparison.

Reports must include:

- CER, WER, exact-line accuracy, substitution/insertion/deletion counts and reference denominators.
- Exact lines gained/lost, improved/worsened/unchanged lines, and whitespace-specific diagnostics.
- Expected/completed/failed/missing lines, EOS/length failures, per-line runtime, and peak GPU memory where measurable.
- Correct failure accounting; never drop failed predictions from the metric denominator to improve the score.
- Selected epoch, checkpoint path/hash, sampler mode/steps, seed, environment, dataset/split fingerprints, and whether test data was used.
- Machine-readable metrics and per-line predictions joined by stable sample_id, plus a readable table.
- Confidence intervals respecting groups when feasible. Start with one documented primary seed; leave confirmation seeds as explicit follow-ups, not an automatic large run.

## Native Windows environment and artifact rules

The previous runner default is:
C:\Users\ammar\AppData\Local\Microsoft\WindowsApps\python.exe

Verify sys.executable, Python version, Torch/CUDA and package compatibility. Native Windows GPU training previously worked. Do not switch to WSL or reinstall Paddle as a detour. Do not assume Paddle's gpu:0 device spelling is Torch's cuda spelling.

Preserve the working environment. Avoid global upgrades or uninstallations. If dependencies require a separate environment, provide a pinned, isolated setup with its exact Python path and a Setup command. Freeze versions after setup. If environment verification later differs, show which relevant packages changed and offer an explicit new environment/run; do not silently bypass hashes or repeatedly refreeze a changed experiment as if unchanged.

Work is already present and the code repository may be dirty. Preserve unrelated changes, existing checkpoints, reports, split locks and frozen baselines. Add a clearly versioned experiment/runner and immutable run directories. Verify a proposed run name is unused. Never overwrite a completed experiment. Resume must be explicit and verify data, configuration, code and initialization bindings; never silently restart or append incompatible training.

Bind reports to checkpoint/config/tokenizer, model initialization, code revision or source snapshot including uncommitted edits, dependency lock, seed, data manifests and selection policy. Keep paths short, use compact sample directory names, and create parent directories explicitly. We previously encountered Windows long-path output failures. Store full sample identity in manifests rather than long directory names.

## PowerShell interface and commands I want

Create one convenient PowerShell runner under Diffusion, with an explicit, discoverable interface. Prefer actions equivalent to Setup if needed, Check, Smoke, Overfit, Train, Validation, ValidationStats and Test. Document how the two model arms are selected, how each run is named, and how checkpoint selection works.

The preferred command shape is:

    & $runner -Action Check -Python $python -Device $device -RunName $runName

That illustrates the requested interface, not a claim a new runner already exists. Implement and verify the actual parameters before presenting commands as runnable.

Give me ordered, copy-pasteable Windows PowerShell blocks:

1. A common setup block defining the absolute project path, exact Python, runner, device, model arms and unused run names. It must work from PS C:\Users\ammar> with no assumptions about the current directory. Explain to rerun this block in a new shell.
2. Setup only if dependencies require it, followed by environment/data/initialization checks.
3. Smoke and tiny-overfit commands, with the readiness conditions explained.
4. The two long training commands, sequential on this GPU, with maximum epochs 200 and patience 30 explicitly visible. Prefer a runner that refuses long training when prerequisite readiness artifacts do not match.
5. Validation and ValidationStats commands exposing selected epochs, raw scores, damage and each gate condition.
6. A separately labelled Test block that checks the real frozen report's validation_passes_release_gate boolean and whose Python backend also enforces the gate. State to run it only after validation passes.
7. How to locate logs/reports and resume an interrupted compatible run, if resume is supported. Never invent a Resume action that has not been implemented.

Use ordinary PowerShell syntax, quoted paths and arrays/splatting where useful. Avoid fragile backtick continuation, Bash syntax, enormous inline Python commands, placeholder paths, or commands requiring me to edit the runner. Avoid deeply truncated Format-Table output. Default to Windows PowerShell 5.1-compatible syntax unless you explicitly verify a different shell requirement.

Check process exit codes reliably. Paddle/native libraries can write harmless diagnostics to stderr: do not treat every stderr line as fatal or hide real nonzero exit codes. Capture $LASTEXITCODE immediately for native Python, preserve/restore error preferences in try/finally where needed, and make the runner throw on failure. Do not rely on $? after unrelated output. Preserve UTF-8 for Arabic.

Show illustrative expected output, clearly labelled as format examples rather than actual experiment results:

Model                      Lines   CER %   WER %  Exact %   Sub   Ins   Del
paddle                       ...     ...     ...      ...   ...   ...   ...
frozen_v2_2                  ...     ...     ...      ...   ...   ...   ...
span_hybrid_verifier_v4      ...     ...     ...      ...   ...   ...   ...
transformer_arabic_control   ...     ...     ...      ...   ...   ...   ...
mdiff_arabic_raw             ...     ...     ...      ...   ...   ...   ...

Selected epoch: ... | Decoder: ... | Steps: ...
Exact lines lost/gained vs v4: ... / ...
Validation gate: ... | Test used: false
Report: absolute path

Use real report field names and real runner parameters in the final commands. Parse/check the runner syntax and CLI, and verify argument routing without launching long training or test. Explain whether each block retrains, resumes, generates predictions or only reads reports.

## Working style and completion criteria

Proceed autonomously with the authorized, reversible implementation. Use local evidence and sound defaults for routine choices. Ask a focused question only when an unresolved requirement genuinely blocks correct work, and explain the concrete reason. Do not repeatedly ask whether I want you to continue.

Give concise updates as you work, and distinguish facts, hypotheses and measured results. Be candid about flaws. Do not flatter a small or noisy gain, fabricate metrics, claim a test you did not run, or claim an Arabic checkpoint exists when it does not.

Keep the changes focused and reviewable. Write meaningful verification for metric accounting, length/space/EOS behavior, initialization, actual iterative updates, preservation reporting, bindings and gate enforcement. Do not spend time on cosmetic tests or a broad unrelated refactor.

Finish with: what changed and why; the controlled hypothesis being tested; exact files; what verification passed and what remains unverified; a small workflow diagram; the ordered PowerShell blocks; expected report format; and the explicit rule for whether I should proceed to evaluation. Return the runnable implementation and commands, not only this plan.
