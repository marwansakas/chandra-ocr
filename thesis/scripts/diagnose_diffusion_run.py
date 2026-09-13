"""Evaluate the explicit final checkpoint on its TRAIN memorization subset."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; refuses overwrite")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ink-mode", choices=("original", "dominant_soft"), help="Inference-only override, explicitly recorded")
    parser.add_argument("--image-controls", action="store_true", help="Also test cyclic shuffled and blank inputs")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from PIL import Image
    from thesis.diffusion.data import file_hash, json_bytes, load_frozen
    from thesis.diffusion.images import emphasize_dominant_ink
    from thesis.diffusion.metrics import aggregate
    from thesis.diffusion.runtime import Collator, LineDataset, full_mask_diagnostics, get_device, load_checkpoint, seed_everything
    from thesis.diffusion.tokenizer import training_text
    from thesis.diffusion.subsets import run_training_records

    torch.set_num_threads(2)
    seed_everything(42)
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    count = run["training_options"].get("overfit_samples")
    if not count:
        raise ValueError("This diagnostic is only for an explicit TRAIN memorization run")
    last = json.loads((args.run / "last.json").read_text(encoding="utf-8"))
    if last["epoch"] != run["arguments"]["epochs"]:
        raise ValueError("Finish the planned epochs before final evaluation")
    checkpoint = args.run / last["checkpoint"]
    device = get_device(args.device)
    model, tokenizer, image_config, _ = load_checkpoint(checkpoint, device)
    model.eval()
    saved_ink_mode = image_config.ink_mode
    if args.ink_mode:
        image_config = replace(image_config, ink_mode=args.ink_mode)
    partitions, lock, _ = load_frozen(args.run / "splits", args.dataset_root)
    source_records = run_training_records(args.run, run, partitions["train"])
    records = [{**r, "source_text": r["text"], "text": training_text(r["text"], model.config.text_policy)} for r in source_records]
    items = [LineDataset(args.dataset_root, records)[i] for i in range(count)]
    ink_audit = [{"sample_id": r["sample_id"], **emphasize_dominant_ink(img)[1]} for img, r in items]
    batch_size = run["training_options"]["batch_size"]
    collator = Collator(image_config, tokenizer, model.config.capacity, model.config.eos_target)
    seeds = [run["training_options"]["seed"] + 200000 + i for i in range(3)]
    conditions = {}
    with torch.inference_mode():
        for condition in (["correct", "shuffled", "blank"] if args.image_controls else ["correct"]):
            if condition == "shuffled":
                inputs = [(items[(i + 1) % count][0], r) for i, (_, r) in enumerate(items)]
            elif condition == "blank":
                inputs = [(Image.new("RGB", img.size, "white"), r) for img, r in items]
            else:
                inputs = items
            batches = [collator(inputs[i:i + batch_size]) for i in range(0, count, batch_size)]
            outcomes = []
            for seed in seeds:
                generator = torch.Generator(device=device).manual_seed(seed)
                predictions = []
                for pixels, widths, _, rows in batches:
                    ids, _ = model.sample(pixels.to(device), widths.to(device), steps=run["training_options"]["sampling_steps"], generator=generator)
                    for row, token_ids in zip(rows, ids.cpu().tolist()):
                        text, has_eos = tokenizer.decode(token_ids)
                        predictions.append({"sample_id": row["sample_id"], "reference": row["text"], "source_reference": row["source_text"], "prediction": text, "has_eos": has_eos})
                outcomes.append({"sampling_seed": seed,
                                 "strict": aggregate([(p["reference"], p["prediction"]) for p in predictions]),
                                 "raw_strict": aggregate([(p["source_reference"], p["prediction"]) for p in predictions]),
                                 "no_eos": sum(not p["has_eos"] for p in predictions), "predictions": predictions})
            conditions[condition] = {"mean_cer_percent": 100 * statistics.mean(r["strict"]["cer"] for r in outcomes),
                                     "mean_exact_line_accuracy_percent": 100 * statistics.mean(r["strict"]["exact_line_accuracy"] for r in outcomes),
                                     "full_mask_diagnostics": full_mask_diagnostics(model, batches, device), "outcomes": outcomes}
    report = {"evaluation": "same TRAIN memorization crops; not held-out test accuracy", "epoch": last["epoch"],
              "samples": count, "reference_characters": sum(len(r["text"]) for r in records),
              "sample_ids": [r["sample_id"] for r in records], "sampling_seeds": seeds,
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_hash(checkpoint / "model.pt"),
              "dataset_fingerprint": lock["dataset_fingerprint"], "model_config": model.config.as_dict(),
              "image_config": image_config.as_dict(), "saved_ink_mode": saved_ink_mode,
              "inference_ink_override": args.ink_mode, "ink_audit": ink_audit, "conditions": conditions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(report))
    print(json.dumps({"report": str(args.output), "samples": count, "epoch": last["epoch"],
                      "mean_cer_percent": {k: v["mean_cer_percent"] for k, v in conditions.items()}}, indent=2))


if __name__ == "__main__":
    main()
