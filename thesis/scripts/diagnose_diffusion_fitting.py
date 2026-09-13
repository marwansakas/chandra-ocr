"""Measure greedy TRAIN recognition and full-mask character/EOS prediction.

Reads the explicit final checkpoint. Writes only a new diagnostic directory;
does not update model weights, labels, frozen partitions, or existing reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new diagnostic directory: {args.output_dir}")
    import torch
    from thesis.diffusion.data import file_hash, json_bytes, load_frozen, resolve_relative
    from thesis.diffusion.decoding_diagnostics import greedy_decode
    from thesis.diffusion.metrics import aggregate
    from thesis.diffusion.runtime import Collator, LineDataset, full_mask_diagnostics, get_device, load_checkpoint, seed_everything
    from thesis.diffusion.tokenizer import training_text
    from thesis.diffusion.subsets import run_training_records

    torch.set_num_threads(2)
    seed_everything(42)
    device = get_device(args.device)
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    last = json.loads((args.run / "last.json").read_text(encoding="utf-8"))
    if last["epoch"] != run["arguments"]["epochs"]:
        raise ValueError("The planned training run has not finished")
    checkpoint = resolve_relative(args.run, last["checkpoint"])
    model, tokenizer, image_config, metadata = load_checkpoint(checkpoint, device)
    if model.config.decoder_kind != "diffusion":
        raise ValueError("This diagnostic requires a diffusion model")
    model.eval()
    partitions, lock, _ = load_frozen(args.run / "splits", args.dataset_root)
    if metadata["dataset_fingerprint"] != lock["dataset_fingerprint"]:
        raise ValueError("Checkpoint and evaluation dataset differ")
    if (metadata["training_options"] != run["training_options"] or metadata["epoch"] != last["epoch"]
            or metadata["split_lock_sha256"] != run["split_lock_sha256"]):
        raise ValueError("Checkpoint and run fitting settings differ")
    records = run_training_records(args.run, run, partitions["train"])
    records = [{**r, "source_text": r["text"], "text": training_text(r["text"], model.config.text_policy)} for r in records]
    collator = Collator(image_config, tokenizer, model.config.capacity, model.config.eos_target)
    loader = torch.utils.data.DataLoader(LineDataset(args.dataset_root, records),
                                         batch_size=run["training_options"]["batch_size"],
                                         shuffle=False, num_workers=0, collate_fn=collator)
    weight_hash = file_hash(checkpoint / "model.pt")
    predictions = []
    with torch.inference_mode():
        for pixels, widths, _, rows in loader:
            ids, _ = greedy_decode(model, pixels.to(device), widths.to(device), method="one_pass")
            for r, token_ids in zip(rows, ids.cpu().tolist()):
                text, has_eos = tokenizer.decode(token_ids)
                predictions.append({"sample_id": r["sample_id"], "reference": r["text"],
                                    "source_reference": r["source_text"], "prediction": text, "has_eos": has_eos})
            if len(predictions) % 200 == 0:
                print(f"Greedy predictions: {len(predictions)}/{len(records)}", flush=True)
        print("Scoring fully masked character and EOS positions...", flush=True)
        diagnostic = full_mask_diagnostics(model, loader, device)
    strict = aggregate([(p["reference"], p["prediction"]) for p in predictions])
    raw = aggregate([(p["source_reference"], p["prediction"]) for p in predictions])
    assert len(predictions) == len(records)
    assert diagnostic["characters"]["positions"] == sum(len(r["text"]) for r in records)
    assert diagnostic["first_eos"]["positions"] == len(records)
    assert file_hash(checkpoint / "model.pt") == weight_hash
    summary = {"partition": "train", "samples": len(records), "epoch": last["epoch"],
               "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": weight_hash,
               "dataset_fingerprint": lock["dataset_fingerprint"], "decoding": "greedy_one_pass",
               "training_subset_sha256": run["training_options"].get("training_subset_sha256"),
               "text_policy": model.config.text_policy, "strict_derived_targets": strict, "strict_raw_targets": raw,
               "no_eos": sum(not p["has_eos"] for p in predictions),
               "mean_prediction_length": statistics.mean(len(p["prediction"]) for p in predictions),
               "mean_reference_length": statistics.mean(len(p["reference"]) for p in predictions),
               "full_mask_diagnostics": diagnostic,
               "notes": "Every decoder input is MASK at fixed capacity and t=1. References select scoring positions only. No training or test-set inference.",
               "source_hashes": {str(p): file_hash(p) for p in (Path(__file__), Path(__file__).parents[1] / "diffusion/decoding_diagnostics.py", Path(__file__).parents[1] / "diffusion/runtime.py", Path(__file__).parents[1] / "diffusion/subsets.py")}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "predictions.jsonl").write_bytes(b"".join(json_bytes(p) for p in predictions))
    (args.output_dir / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
