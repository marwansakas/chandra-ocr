"""Report validation-selected diffusion runs without conflating partitions."""
import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .diffusion_stats_table import format_evaluation_table
else:
    from diffusion_stats_table import format_evaluation_table


def compact(row):
    scores = row["metrics"]
    result = {"epoch": row["epoch"], "optimization_train_loss": round(row["train_loss"], 4),
              "validation_loss": round(row["validation_loss"], 4) if row.get("validation_loss") is not None else None,
              "validation_all_mask_loss": round(row["validation_all_mask_loss"], 4) if row.get("validation_all_mask_loss") is not None else None,
              "train_fixed_corruption_loss": round(row["train_fixed_corruption_loss"], 4) if row.get("train_fixed_corruption_loss") is not None else None,
              "train_all_mask_loss": round(row["train_all_mask_loss"], 4) if row.get("train_all_mask_loss") is not None else None,
              "cer_percent": round(100 * scores["strict"]["cer"], 3),
              "raw_cer_percent": round(100 * scores["raw_strict"]["cer"], 3),
              "exact_crops": scores["strict"]["exact_lines"], "no_eos": row["no_eos"],
              "mean_prediction_length": round(row["mean_prediction_length"], 2),
              "mean_reference_length": round(row["mean_reference_length"], 2)}
    if "full_mask_diagnostics" in row:
        for key in ("characters", "first_eos"):
            accuracy = row["full_mask_diagnostics"][key]["accuracy"]
            result[key + "_accuracy_percent"] = round(100 * accuracy, 3) if accuracy is not None else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, help="Optional completed test evaluation directory")
    parser.add_argument("--controls", type=Path, help="Optional controls JSON for the selected checkpoint")
    parser.add_argument("--format", choices=("table", "json"), default="table",
                        help="Saved evaluation table (default), or full JSON; without evaluation, prints JSON")
    parser.add_argument("--metric-policy", choices=("historical", "strict"), default="historical")
    parser.add_argument("--checkpoint-selection", choices=("best_cer", "best_loss", "last"), default="best_cer",
                        help="Checkpoint represented by optional evaluation and controls")
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    rows = [json.loads(s) for s in (args.run / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows or run["sampler"] != "greedy_one_pass":
        raise ValueError("Requires a fitting run with greedy epoch metrics")
    split_lock = json.loads((args.run / "splits" / "split_lock.json").read_text(encoding="utf-8"))
    pointer_name = {"best_cer": "best.json", "best_loss": "best_loss.json", "last": "last.json"}[args.checkpoint_selection]
    selected_path = args.run / pointer_name
    if not selected_path.exists() and args.checkpoint_selection == "best_cer":
        selected_path = args.run / "last.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected_model_sha256 = hashlib.sha256((args.run / selected["checkpoint"] / "model.pt").read_bytes()).hexdigest()
    validation_rows = [row for row in rows if row.get("metrics_partition") == "validation" and row.get("metrics")]
    report = {"run": args.run.name,
              "epoch_metrics_partition": "validation" if validation_rows else rows[-1].get("metrics_partition"),
              "partitions": {"train": split_lock["counts"]["train"], "validation": split_lock["counts"]["validation"], "test": split_lock["counts"]["test"],
                             "fixed_train_loss_subset": len(json.loads((args.run / "train_loss_subset.json").read_text(encoding="utf-8"))["sample_ids"]) if (args.run / "train_loss_subset.json").exists() else None},
              "settings": {"planned_epochs": run["arguments"]["epochs"], "dropout": run["model_config"]["dropout"],
                           **run["training_options"]},
              "completed_epochs": rows[-1]["epoch"], "optimizer_updates": rows[-1]["optimizer_updates"],
              "curve": [compact(r) for r in rows if r["epoch"] == 1 or r["epoch"] % 20 == 0 or r is rows[-1]],
              "final_epoch": compact(rows[-1]),
              "completion": {"state": "early_stopped" if rows[-1].get("early_stop") else "completed_max_epochs" if rows[-1]["epoch"] == run["arguments"]["epochs"] else "stopped_before_max_epochs",
                             "early_stop": rows[-1].get("early_stop")},
              "selected_checkpoint": {**selected, "model_sha256": selected_model_sha256,
                                      "requested_selection": args.checkpoint_selection,
                                      "reason": selected.get("selection_metric", "last_checkpoint")},
              "provenance": {"dataset_fingerprint": run["dataset_fingerprint"], "split_lock_sha256": run["split_lock_sha256"],
                             "tokenizer_sha256": hashlib.sha256((args.run / "splits" / "tokenizer.json").read_bytes()).hexdigest()}}
    if validation_rows:
        report["best_validation_cer"] = compact(min(validation_rows, key=lambda r: r["metrics"]["strict"]["cer"]))
        report["best_validation_loss"] = compact(min(validation_rows, key=lambda r: r["validation_loss"]))
    path = args.run / "final_fitting_diagnosis" / "summary.json"
    if path.exists():
        final = json.loads(path.read_text(encoding="utf-8"))
        report["final_replay_matches_epoch_metrics"] = (
            final["epoch"] == rows[-1]["epoch"] and final["strict_derived_targets"] == rows[-1]["metrics"]["strict"]
            and final["strict_raw_targets"] == rows[-1]["metrics"]["raw_strict"] and final["no_eos"] == rows[-1]["no_eos"]
            and final["training_subset_sha256"] == run["training_options"].get("training_subset_sha256"))
        report["final_checkpoint_sha256"] = final["checkpoint_sha256"]
    else:
        report["final_replay_matches_epoch_metrics"] = None
    if args.evaluation:
        evaluation = json.loads((args.evaluation / "evaluation.json").read_text(encoding="utf-8"))
        if evaluation["partition"] != "test" or evaluation["dataset_fingerprint"] != run["dataset_fingerprint"] or evaluation["split_lock_sha256"] != run["split_lock_sha256"]:
            raise ValueError("Test evaluation must match the run's frozen dataset and split")
        report["test"] = {"partition": evaluation["partition"], "models": evaluation["models"]}
    if args.controls:
        controls = json.loads(args.controls.read_text(encoding="utf-8"))
        if controls["dataset_fingerprint"] != run["dataset_fingerprint"] or controls["split_lock_sha256"] != run["split_lock_sha256"]:
            raise ValueError("Controls must match the run's frozen dataset and split")
        if controls["checkpoint_sha256"] != selected_model_sha256:
            raise ValueError("Controls must use the currently selected checkpoint")
        report["controls"] = controls
    if args.format == "table" and args.evaluation:
        print(format_evaluation_table(report["test"], args.metric_policy, f"{args.run.name}/{args.checkpoint_selection}"))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
