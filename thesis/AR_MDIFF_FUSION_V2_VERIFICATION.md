# Fusion suite preparation verification — 2026-10-31

**Preparation-time snapshot:** the user subsequently completed the real runs.
Their results are recorded in the
[current experiment README](experiments/arabic_ar_mdiff/README.md); the verification
scope below describes what was executed while preparing the implementation.

Implemented as a separate package and runner. No actual OCR training, preparation inference, validation calibration or historical evaluation was executed. The real suite output directory `Diffusion/drive_all/hybrids/ar_md_fusion2_s54101` was not created during preparation.

Executed:

```text
python -m pytest -q tests/test_ar_mdiff_fusion_v2.py tests/test_ar_mdiff_hybrid_v1.py tests/test_mdiff_arabic_v1.py
79 passed in 27.78s
```

The tests use CPU toy models, synthetic records and temporary directories. They cover:

- Logical Unicode Levenshtein alignment, insertions/deletions, literal spaces, bounded span combinations and capacity limits.
- Causal AR shifting and the first-EOS denominator.
- MDiff masking of every scored token, including EOS; MASK tail, cloze noise fraction and equivalent masked-batch sizes.
- Cached-memory MDiff sampling matching the original sampler on a deterministic toy decoder.
- Frozen backbone parameters/buffers and nonzero adapter learning.
- Interrupted toy training resuming with optimizer/selection state; latest and selected checkpoint retention.
- Safe checkpoint cleanup restricted to this new shared run, with selected/external files protected.
- Per-line cache resume, tamper detection, atomic publication after interruption and OS lock release.
- Calibration subset isolation, audit denominators, exact-line damage and frozen settings.
- All three methods producing one comparison with exactly 752 synthetic lines per table row.
- No historical-reference access through preparation/calibration actions; All dispatch ordering.
- Dry-run parsing for every CLI action and no model execution in dry runs.
- Native Windows PowerShell 5.1 runner invocation; stderr with exit 0 accepted, exit 7 rejected.

Read-only integrity verification against the previously frozen bindings also passed:

| Existing artifact | Verified |
|---|---|
| AR `ar1b_s54101` | All 18 bound source files match; selected epoch-93 checkpoint hash matches |
| MDiff `ma1b_s54101` | All 18 bound source files match; selected epoch-177 checkpoint hash matches |
| Previous hybrid `ar_md_h1_s54101` | All 6 bound source files match |

Real CUDA forward passes, full-data scoring time, shared-head convergence and resulting CER/WER remain to be measured by the user-run commands in `AR_MDIFF_FUSION_V2.md`. No accuracy result is claimed from these code checks.
