"""Print saved data, training and test statistics for a diffusion run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .diffusion_stats_table import format_evaluation_table
else:
    from diffusion_stats_table import format_evaluation_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, help="Folder containing stats/data_stats.json")
    parser.add_argument("--run", type=Path, help="Training run containing training_metrics.jsonl")
    parser.add_argument("--evaluation", type=Path, help="Evaluation directory containing evaluation.json")
    parser.add_argument("--format", choices=("table", "json"), default="table",
                        help="Saved evaluation table (default), or full JSON; without evaluation, prints JSON")
    parser.add_argument("--metric-policy", choices=("historical", "strict"), default="historical")
    args = parser.parse_args()
    if not any((args.pilot_root, args.run, args.evaluation)):
        parser.error("Supply --pilot-root, --run and/or --evaluation")
    report = {}
    if args.pilot_root:
        data = json.loads((args.pilot_root / "stats" / "data_stats.json").read_text(encoding="utf-8"))
        keys = ("selected_pages", "surya_detected_line_folders", "ground_truth_line_folders", "usable_labeled_lines",
                "training_lines", "test_lines", "capacity_including_eos", "vocabulary_size_with_specials", "test_oov_lines",
                "test_capacity_overflow", "pages_in_both_partitions", "downloaded_source_bytes")
        report["data"] = {key: data[key] for key in keys}
    if args.run:
        history = [json.loads(line) for line in (args.run / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
        report["training"] = {"completed_epochs_in_this_run": len(history), "mode": run["mode"],
                              "selection_metric": run["selection_metric"], "trainable_parameters": run["trainable_parameters"],
                              "last_epoch": history[-1] if history else None,
                              "loss_by_epoch": [{"epoch": row["epoch"], "loss": row["train_loss"]} for row in history]}
    if args.evaluation:
        result = json.loads((args.evaluation / "evaluation.json").read_text(encoding="utf-8"))
        report["test"] = {"partition": result["partition"], "models": result["models"]}
    if args.format == "table" and args.evaluation:
        print(format_evaluation_table(report["test"], args.metric_policy))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
