"""Compact final comparison for a suite of TRAIN memorization experiments."""
import argparse
import json
from pathlib import Path


def summarize(root):
    rows = []
    for name in ("baseline", "eos", "fullmask", "combined", "ink"):
        folder = root / name
        if not (folder / "run.json").is_file():
            rows.append({"variant": name, "status": "not_started"})
            continue
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        metric_path = folder / "training_metrics.jsonl"
        lines = metric_path.read_text(encoding="utf-8").splitlines() if metric_path.exists() else []
        if not lines:
            rows.append({"variant": name, "status": "no_completed_epoch"})
            continue
        last = json.loads(lines[-1])
        scores = last["metrics"]["strict"]
        row = {"variant": name, "status": "trained" if last["epoch"] == run["arguments"]["epochs"] else "incomplete",
               "epoch": last["epoch"], "updates": last.get("optimizer_updates"), "loss": last["train_loss"],
               "final_cer_percent": 100 * scores["cer"], "exact_crops": scores["exact_lines"],
               "raw_final_cer_percent": 100 * last["metrics"].get("raw_strict", scores)["cer"],
               "changed_training_labels": run.get("target_policy", {}).get("changed_training_labels", 0),
               "samples": scores["lines"], "text_policy": run["model_config"].get("text_policy", "literal")}
        if (folder / "final_diagnosis.json").is_file():
            report = json.loads((folder / "final_diagnosis.json").read_text(encoding="utf-8"))
            correct = report["conditions"]["correct"]
            diag = correct["full_mask_diagnostics"]
            row.update({"status": "evaluated", "three_sampling_seeds_mean_cer_percent": correct["mean_cer_percent"],
                        "three_sampling_seeds_mean_exact_percent": correct["mean_exact_line_accuracy_percent"],
                        "full_mask_character_accuracy_percent": 100 * diag["characters"]["accuracy"],
                        "full_mask_first_eos_accuracy_percent": 100 * diag["first_eos"]["accuracy"],
                        "shuffled_mean_cer_percent": report["conditions"].get("shuffled", {}).get("mean_cer_percent"),
                        "blank_mean_cer_percent": report["conditions"].get("blank", {}).get("mean_cer_percent")})
        rows.append(row)
    return {"suite": str(root.resolve()), "evaluation": "TRAIN memorization; original 120 test crops untouched",
            "variants": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Convenience summary; refreshed on each Stats call")
    args = parser.parse_args()
    report = summarize(args.suite_root)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
