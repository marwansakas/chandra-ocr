"""Validate and print a matched comparison of two diffusion OCR test runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .diffusion_stats_table import format_evaluation_table
else:
    from diffusion_stats_table import format_evaluation_table


VARIANT_CHANGES = {
    "dropout01": {"dropout": (0.0, 0.1)},
    "weightdecay1e4": {"weight_decay": (0.0, 0.0001)},
    "mildaug": {"augmentation": ("none", "mild_v1")},
    "combined": {
        "augmentation": ("none", "mild_v1"),
        "weight_decay": (0.0, 0.0001),
        "dropout": (0.0, 0.1),
    },
    "trocrviews": {"augmentation": ("none", "trocr_views_v1")},
    "trocrallviews": {
        "augmentation": ("none", "trocr_all_views_v1"),
        "early_stopping_patience": (30, 0),
    },
}

REPORT_ONLY_SOURCES = {
    "thesis/scripts/compare_diffusion_runs.py",
    "thesis/scripts/preview_diffusion_augmentation.py",
    "thesis/scripts/report_diffusion_fitting.py",
}

# The baseline predates this new policy. These files contain the policy branch,
# epoch-to-view routing, and console formatting needed to run it. All remaining
# captured training/inference sources still have to match byte-for-byte.
VARIANT_SOURCE_CHANGES = {
    "trocrviews": {
        "thesis/diffusion/images.py",
        "thesis/diffusion/runtime.py",
        "thesis/scripts/train_diffusion_line_ocr_model.py",
    },
    "trocrallviews": {
        "thesis/diffusion/images.py",
        "thesis/diffusion/runtime.py",
        "thesis/scripts/train_diffusion_line_ocr_model.py",
    },
}

# These settings define comparison conditions. Output paths and the one named
# experimental setting are handled separately.
COMPARISON_FIELDS = (
    "augmentation",
    "batch_size",
    "check_only",
    "checkpoint_every",
    "cpu_threads",
    "d_model",
    "dataset_root",
    "decoder_kind",
    "decoder_layers",
    "device",
    "diagnostics_every",
    "diffusion_steps",
    "early_stopping_patience",
    "encoder_layers",
    "eos_target",
    "exact_line_cer_band",
    "epochs",
    "evaluation_decoding",
    "full_mask_probability",
    "heads",
    "height",
    "ink_mode",
    "initialize_from",
    "learning_rate",
    "max_width",
    "monitor_train",
    "overfit_manifest",
    "overfit_samples",
    "patch_size",
    "resume",
    "sampling_steps",
    "save_traces",
    "seed",
    "selection_min_delta",
    "split_root",
    "text_policy",
    "train_loss_samples",
    "validation_loss_seed",
    "vit_checkpoint",
    "weight_decay",
    "workers",
    "dropout",
)


@dataclass(frozen=True)
class RunResult:
    label: str
    root: Path
    metadata: dict
    evaluation: dict


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def selection_suffix(selection: str) -> str:
    return "" if selection == "best_cer" else f"_{selection}"


def load_run(label: str, root: Path, selection: str = "best_cer") -> RunResult:
    metadata = read_json(root / "run.json")
    suffix = selection_suffix(selection)
    evaluation = read_json(root / f"test_evaluation_greedy_one_pass{suffix}" / "evaluation.json")
    if evaluation.get("partition") != "test":
        raise ValueError(f"{label} evaluation is not for TEST")
    if evaluation.get("dataset_fingerprint") != metadata.get("dataset_fingerprint"):
        raise ValueError(f"{label} evaluation dataset fingerprint differs from run.json")
    if evaluation.get("split_lock_sha256") != metadata.get("split_lock_sha256"):
        raise ValueError(f"{label} evaluation split hash differs from run.json")
    if set(evaluation.get("models", {})) != {"diffusion"}:
        raise ValueError(f"{label} evaluation must contain exactly the diffusion model")
    model = evaluation["models"]["diffusion"]
    if model.get("completed") != model.get("expected") or model.get("missing") or model.get("failed"):
        raise ValueError(f"{label} TEST evaluation is incomplete")
    pointer_name = {"best_cer": "best.json", "best_exact": "best_exact.json", "best_loss": "best_loss.json", "last": "last.json"}[selection]
    pointer = read_json(root / pointer_name)
    checkpoint_sha256 = hashlib.sha256((root / pointer["checkpoint"] / "model.pt").read_bytes()).hexdigest()
    summary = read_json(root / f"test_predictions_greedy_one_pass{suffix}" / "run_summary.json")
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{label} TEST evaluation does not use the recorded {selection} checkpoint")
    return RunResult(label, root, metadata, evaluation)


def validate_comparison(baseline: RunResult, candidate: RunResult, variant: str) -> None:
    if baseline.metadata.get("dataset_fingerprint") != candidate.metadata.get("dataset_fingerprint"):
        raise ValueError("Runs use different dataset fingerprints")
    if baseline.metadata.get("split_lock_sha256") != candidate.metadata.get("split_lock_sha256"):
        raise ValueError("Runs use different frozen split hashes")
    for field in ("source_revision", "versions", "python", "platform", "torch_cuda", "gpu"):
        if baseline.metadata.get("runtime", {}).get(field) != candidate.metadata.get("runtime", {}).get(field):
            raise ValueError(f"Runs use different recorded runtime {field}")
    ignored_sources = REPORT_ONLY_SOURCES | VARIANT_SOURCE_CHANGES.get(variant, set())
    baseline_hashes = {
        name: value
        for name, value in baseline.metadata.get("runtime", {}).get("source_hashes", {}).items()
        if name not in ignored_sources
    }
    candidate_hashes = {
        name: value
        for name, value in candidate.metadata.get("runtime", {}).get("source_hashes", {}).items()
        if name not in ignored_sources
    }
    if baseline_hashes != candidate_hashes:
        raise ValueError("Runs differ outside the source files required for the named experiment")

    expected = VARIANT_CHANGES[variant]
    baseline_args = baseline.metadata["arguments"]
    candidate_args = candidate.metadata["arguments"]
    differences = {
        field: (baseline_args.get(field), candidate_args.get(field))
        for field in COMPARISON_FIELDS
        if baseline_args.get(field) != candidate_args.get(field)
    }
    if differences != expected:
        raise ValueError(
            f"Comparison is not the expected {variant} experiment; "
            f"observed differences: {differences}"
        )


def comparison_evaluation(baseline: RunResult, candidate: RunResult, *additional: RunResult) -> dict:
    return {
        "partition": "test",
        "models": {
            baseline.label: baseline.evaluation["models"]["diffusion"],
            candidate.label: candidate.evaluation["models"]["diffusion"],
            **{run.label: run.evaluation["models"]["diffusion"] for run in additional},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate matched diffusion runs and print Letter/Word/Exact accuracy."
    )
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--candidate-variant", choices=sorted(VARIANT_CHANGES), required=True)
    parser.add_argument("--metric-policy", choices=("strict", "historical"), default="strict")
    parser.add_argument("--include-last", action="store_true", help="Print best-CER and last-checkpoint TEST rows for both runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_label = "current_baseline_best" if args.include_last else "current_baseline"
    candidate_label = f"{args.candidate_variant}_best" if args.include_last else args.candidate_variant
    baseline = load_run(baseline_label, args.baseline_run.resolve())
    candidate = load_run(candidate_label, args.candidate_run.resolve())
    validate_comparison(baseline, candidate, args.candidate_variant)
    additional = ()
    if args.include_last:
        additional = (
            load_run("current_baseline_last", args.baseline_run.resolve(), "last"),
            load_run(f"{args.candidate_variant}_last", args.candidate_run.resolve(), "last"),
        )
    print(format_evaluation_table(comparison_evaluation(baseline, candidate, *additional), args.metric_policy))
    source_note = (
        "all other captured sources match; the augmentation implementation/routing and console formatter are the recorded source changes"
        if args.candidate_variant in VARIANT_SOURCE_CHANGES
        else "source snapshots match"
    )
    budget_note = (
        "planned epochs match, with three candidate views and optimizer updates per crop per epoch"
        if args.candidate_variant == "trocrallviews"
        else "training budget matches"
    )
    print(
        "Matched provenance: same dataset fingerprint, frozen split, runtime, seed, architecture and decoding; "
        f"{budget_note}; {source_note}."
    )


if __name__ == "__main__":
    main()
