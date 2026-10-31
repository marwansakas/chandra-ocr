# AR-default, MDiff-candidate line selector v1

**Completed results:** this selector reached CER 51.13%, WER 84.57% and 123/752
exact lines, losing 17 AR-exact lines and gaining four. See the
[current experiment README](experiments/arabic_ar_mdiff/README.md) for the full
comparison and the later dual-rescoring/span-merge experiments.

This is a new, separately calibrated v2.2-inspired hybrid. It selects one complete
existing line prediction; it does not merge words, run another diffusion pass,
retrain either recognizer, or reuse Paddle-specific gate coefficients. No base
model source, weights, tokenizer, checkpoint selection or gate report is changed.

The selected parents stay AR epoch 93 (`ar1b_s54101/m093.pt`) and MDiff epoch 177
(`ma1b_s54101/m177.pt`). Source, environment, initialization, data and checkpoint
bindings are verified through the original experiment before each action.

## Checks actually performed on saved validation outputs

All 150 canonical validation IDs are matched, with no new model inference.

| Quantity | Lines |
|---|---:|
| MDiff has fewer character errors | 50 |
| AR has fewer character errors | 58 |
| Equal character distance | 42 |
| MDiff exact, AR wrong | 5 |
| AR exact, MDiff wrong | 13 |
| Both exact | 16 |
| Both wrong | 116 |

AR: 1137 edits / 2395 characters = 47.47% CER, 29 exact lines.
MDiff: 1251 edits / 2395 characters = 52.23% CER, 21 exact lines.
An oracle choosing the lower-error complete line has 1022 edits = 42.67% CER,
34 exact lines. Potential headroom over AR is 115 edits / 4.80 percentage points.
**That reference-dependent oracle is an upper bound, not a usable model or a
hybrid result.** No oracle rule is used during actual selection.

## Method and calibration scope

AR is the default. A fresh ridge regression predicts normalized character-edit
gain for replacing AR with MDiff. It uses 14 fixed label-free features: output
lengths, disagreement ratio, common prefix/suffix, relative spaces/digits/
punctuation/uniqueness/repetition, EOS flags and image aspect ratio. The target
is `(AR edits - MDiff edits) / max(1, output lengths)`. Regularization is fixed at
10; coefficients and feature scaling are fitted afresh. This score is **not a
calibrated probability of correctness**.

No token-confidence fields were saved in the new base predictions. They are not
fabricated or filled with zeros. Confidence collection and agreement-anchored
second-pass decoding are later separate ablations, not silently folded into this
first line-selector experiment. Both parents share pretraining/data, so agreement
does not constitute independent correctness evidence.

The recognizers saw all 2103 TRAIN lines. Their saved TRAIN-derived debug holdout
was not excluded from recognizer training. Consequently it cannot provide unseen
primary-model predictions for honest OOF gate calibration. No nested Paddle
project or automatic recognizer retraining is started.

This practical first selector instead uses the **already development-used
150-line validation set** in fixed notebook/writer-group roles:

- 90 lines / 15 groups: fit regression coefficients.
- 30 lines / 4 groups: select threshold.
- 30 lines / 4 groups: audit the frozen gate once.

Roles use only group IDs and counts: largest groups first, balance target
60/20/20 counts, tie by SHA256(seed 54101 | group). No reference text or error
stratification chooses roles. Whole groups stay together among roles. They are
not verified writer identities. The base checkpoints were already selected on
all 150 labels, and the aggregate complementarity was inspected. Thus even the
gate audit is **reused development evidence**, not pristine generalization.
Reports explicitly separate role scores and label the mixed full-set table.

Threshold grid is frozen at 0, .025, .05, .1, .2, .4, 1, plus a reject-all control.
Calibration eligibility requires strictly fewer character edits than AR, no more
word edits, zero AR-exact lines lost, and complete valid coverage/EOS. Among
passing grid values choose lower character edits, fewer word edits, more exact
lines, fewer switches, then higher threshold. If none pass, retain the predefined
threshold-zero **raw candidate as rejected**. It is not hidden behind unchanged
AR scores. The audit never refits the gate or changes its threshold.

MDiff is selected only when valid and its score strictly exceeds the threshold.
Identical outputs retain AR. An unavailable MDiff retains AR; an unavailable AR
can use a valid MDiff. Ground truth is never read by feature extraction or the
selection function. No generated text, source-length oracle or character folding.

The new hybrid gate requires calibration, audit and combined AR-relative checks
to pass. The original v4/v2.2 preservation condition is reported separately as
`passes_original_v4_release_gate`; it is not renamed to conceal prior failures.
All scores are raw no_harakat_v1 aggregate unclamped CER/WER, with substitution,
insertion/deletion counts, denominators, exact gains/losses, failure accounting
and grouped descriptive intervals. There is no automatic model replacement.

## Interface and execution boundaries

Runner: `Diffusion/run_ar_mdiff_hybrid_v1.ps1`.
Sources: `thesis/ar_mdiff_hybrid_v1/{logic,workflow}.py` and
`thesis/scripts/run_ar_mdiff_hybrid_v1.py`.
Tests: `tests/test_ar_mdiff_hybrid_v1.py`.

Artifacts: `Diffusion/drive_all/hybrids/ar_md_h1_s54101`.
Native logs: `Diffusion/drive_all/logs/ar_md_h1_s54101`.
Completed outputs are immutable; no silent resume/restart. Changed source,
environment, parents or manifests require a new reviewed run name. No new
dependencies are needed. The numeric gate runs on CPU; `Device=cuda` preserves
the original primary-model environment binding and does not trigger GPU fitting.

- **Check** reads saved validation predictions, verifies parents, freezes code,
  recipe, roles and complementarity diagnostics. It does not fit or infer.
- **Fit** fits the small fresh gate on 90 development lines and calibrates its
  threshold on 30. It does not retrain or run either recognizer. User executes it.
- **Validation** applies the frozen selector to saved validation candidates,
  reports the 30-line audit and mixed 150-line table, then seals its report.
- **ValidationStats** verifies and reads the frozen report; no model runs.
- **Test** applies the frozen selector to the already cached historical 752-line
  AR/MDiff/Paddle/v2.2/v4 predictions from `mdar_20261031_x01`. It verifies cached
  checkpoint/split hashes, IDs, references and image bindings. No new recognizer
  inference is needed. Historical results remain exploratory/development-used.
  The default action stops before reading historical labels if the new hybrid
  validation gate fails. Explicit `-AllowExploratory` records an evaluation-only
  override; no gate boolean or original experiment is changed.

Per user preference, actual fitting and historical comparison are supplied as
commands and are **not executed by Codex**. Synthetic unit-test fitting is only
bounded verification of code behavior, not a real hybrid result.

## PowerShell 5.1 commands

Rerun definitions in a new shell. Paths work from PS C:\Users\ammar>.

```powershell
$ErrorActionPreference = 'Stop'
$project = 'C:\Users\ammar\OneDrive\Desktop\final project'
$python = 'C:\Users\ammar\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe'
$runner = Join-Path $project 'Diffusion\run_ar_mdiff_hybrid_v1.ps1'
$hybrid = @{
    Python = $python
    Device = 'cuda'
    RunName = 'ar_md_h1_s54101'
    ARunName = 'ar1b_s54101'
    MDiffRunName = 'ma1b_s54101'
    TestCacheName = 'mdar_20261031_x01'
}
& $runner -Action Check @hybrid
```

Fresh numeric selector fitting, followed by frozen development reporting:

```powershell
& $runner -Action Fit @hybrid
& $runner -Action Validation @hybrid
& $runner -Action ValidationStats @hybrid
```

Historical comparison only after reviewing the development report:

```powershell
& $runner -Action Test @hybrid
```

If the hybrid gate failed and you explicitly choose a descriptive historical
comparison anyway, use this **instead** of the preceding Test command:

```powershell
& $runner -Action Test @hybrid -AllowExploratory
```

Read the saved table without recomputing:

```powershell
$folder = Join-Path $project 'Diffusion\drive_all\hybrids\ar_md_h1_s54101'
Get-Content -LiteralPath (Join-Path $folder 'validation\report.txt') -Encoding UTF8
# After a historical comparison exists:
# Get-Content -LiteralPath (Join-Path $folder 'historical_test\report.txt') -Encoding UTF8
```

The runner captures LASTEXITCODE immediately, preserves error preferences and
UTF-8, accepts harmless native stderr with exit 0, and throws on nonzero exits.
`-DryRun` verifies argument routing without fitting, data access or test execution.
