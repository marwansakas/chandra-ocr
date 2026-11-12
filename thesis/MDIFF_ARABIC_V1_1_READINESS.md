# Readiness v1.1: remove the tiny-subset width shortcut

**Historical readiness fix:** the corrected runs subsequently completed. See the
[current experiment README](experiments/arabic_ar_mdiff/README.md) for selected
checkpoints, measured results and the later hybrid comparisons.

The v1 AR report `20261029T141853547292Z/report.json` is a **readiness failure**,
not a Python/CUDA crash: after 1200 updates normal and blank images both have
0% CER and 4/4 exact lines. Only `better_than_blank` failed. Shuffled CER is
108.57%. The paired v1 MDiff report `20261029T141645215676Z/report.json` passed:
normal/missing/replaced/correct CER 0%, 4/4 exact, no correct input damaged;
blank CER 14.29%, shuffled CER 108.57%. These are TRAIN-only memorization probes,
not validation results or evidence of generalization.

I initially selected four TRAIN examples without matching their geometry. Their
resized widths are 65,137,108,88, so even blank images have distinct encoder
extents. Shuffling also swapped geometry. The previous results therefore cannot
isolate glyph-content dependence from a width shortcut. No diagnostic checkpoint
was saved, so a causal intervention on those fitted weights cannot now be made;
the width explanation is plausible, not proven for that particular checkpoint.

The separately named runner `Diffusion/run_mdiff_arabic_v1_1.ps1` explicitly
selects `geometry_matched_v1`. It keeps the recognizer, initialization, vocabulary,
canonical data/preprocessing, primary training recipe, promotion policy and
**every readiness threshold unchanged**. It does not bypass the failed report.

The new protocol selects four distinct labels from TRAIN only. Eligible labels
still have 4..18 evaluation codepoints. Group by ceil(resized width / 4), select
the bucket with most distinct labels (tie lower width), then SHA256(seed+ID)
order, preferring distinct notebook/writer groups. No validation/test labels,
model scores, failed-prediction rankings or manual choice of easy examples enter
selection. The first new Check freezes the resulting manifest before fitting.

For this dataset the deterministic selection has four distinct notebook groups
and encoder input shape 1×3×32×68. Original aspect ratios remain intact, with the
same <=3px stride-alignment border. During the blank diagnostic all pixels and
encoder dimensions are **identical** across the four distinct labels. With
deterministic inference a width-only recognizer cannot produce four distinct
correct answers. Shuffling now swaps image content within the common geometry.
Smoke and Overfit verify this actual tensor invariant and record its hash/shape.

New tests cover deterministic TRAIN selection, distinct labels, identical blank
inputs, rejection of geometry mismatches and explicit CLI routing. Expected
readiness failures now print failed conditions/report path and exit 2, without a
misleading Python traceback. Nonzero exits still fail the PowerShell runner.

Old checks, reports, source snapshots and logs remain intact. The source binding
has changed, so **do not reuse `ma1_s54101`/`ar1_s54101` or their readiness flags**.
Use `ma1b_s54101`/`ar1b_s54101`. Both primary training directories were absent at
diagnosis. No long training or historical test inference is launched by this fix.

The old complete workflow is in MDIFF_ARABIC_V1.md; use the new runner and names
below. All primary training/selection/gate rules remain in force.

## Copy-paste commands, PowerShell 5.1

From any working directory (rerun definitions in a new shell):

```powershell
$ErrorActionPreference = 'Stop'
$project = 'C:\Users\ammar\OneDrive\Desktop\final project'
$python = 'C:\Users\ammar\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe'
$runner = Join-Path $project 'Diffusion\run_mdiff_arabic_v1_1.ps1'
$runRoot = Join-Path $project 'Diffusion\drive_all\runs'
$common = @{ Python = $python; Device = 'cuda' }
$mdiff = @{ Arm = 'mdiff'; RunName = 'ma1b_s54101'; PeerRunName = 'ar1b_s54101' }
$ar = @{ Arm = 'ar'; RunName = 'ar1b_s54101'; PeerRunName = 'ma1b_s54101' }
foreach ($name in @($mdiff.RunName, $ar.RunName)) {
    if (Test-Path -LiteralPath (Join-Path $runRoot $name)) { throw "Run name already used: $name" }
}

& $runner -Action Check @common @mdiff
& $runner -Action Check @common @ar
& $runner -Action Smoke @common @mdiff
& $runner -Action Smoke @common @ar
```

Run fresh TRAIN-only diagnostic fits. Neither reuses old overfit weights/reports.
Stop on failure and return the saved report; do not relax any condition.

```powershell
& $runner -Action Overfit @common @mdiff -OverfitUpdates 1200 -OverfitSeconds 900
& $runner -Action Overfit @common @ar -OverfitUpdates 1200 -OverfitSeconds 900
```

Only after **both new** readiness reports pass, long training may proceed:

```powershell
& $runner -Action Train @common @mdiff -Epochs 200 -Patience 30
& $runner -Action Train @common @ar -Epochs 200 -Patience 30
```

After both fits complete, freeze the paired validation comparison, then read it:

```powershell
& $runner -Action Validation @common @mdiff
& $runner -Action ValidationStats @common @mdiff
```

Historical Test remains blocked until the frozen paired validation gate passes:

```powershell
$reportPath = Join-Path (Join-Path $runRoot $mdiff.RunName) 'validation\report.json'
$report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($report.validation_passes_release_gate -isnot [bool] -or $report.validation_passes_release_gate -ne $true) {
    throw 'STOP: frozen validation gate did not pass'
}
& $runner -Action Test @common @mdiff
```
