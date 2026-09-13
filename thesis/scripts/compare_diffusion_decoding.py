"""Compare decoding on the final combined/ink TRAIN memorization checkpoints."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory, never overwrites results")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Results already exist: {args.output_dir}. Read comparison.json or choose another directory.")
    import torch
    from thesis.diffusion.data import file_hash, json_bytes, load_frozen, resolve_relative
    from thesis.diffusion.decoding_diagnostics import greedy_decode
    from thesis.diffusion.metrics import aggregate
    from thesis.diffusion.runtime import Collator, LineDataset, get_device, load_checkpoint, seed_everything
    from thesis.diffusion.tokenizer import training_text
    from thesis.diffusion.subsets import run_training_records

    torch.set_num_threads(2)
    seed_everything(42)
    device = get_device(args.device)
    prepared = []
    for variant in ("combined", "ink"):
        run_root = args.suite_root / variant
        run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
        last = json.loads((run_root / "last.json").read_text(encoding="utf-8"))
        if last["epoch"] != run["arguments"]["epochs"] or not run["training_options"].get("overfit_samples"):
            raise ValueError(f"{variant}: finish the planned TRAIN memorization run first")
        checkpoint = resolve_relative(run_root, last["checkpoint"])
        partitions, lock, _ = load_frozen(run_root / "splits", args.dataset_root)
        prepared.append((variant, run_root, run, checkpoint, run_training_records(run_root, run, partitions["train"]), lock))
    if [r["sample_id"] for r in prepared[0][4]] != [r["sample_id"] for r in prepared[1][4]]:
        raise ValueError("The models were trained on different memorization subsets")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    methods = (("ancestral_16", None, 3), ("greedy_one_pass", "one_pass", 1),
               ("greedy_random_reveal_16", "random_reveal", 3), ("greedy_confidence_16", "confidence", 1))
    with torch.inference_mode():
        for variant, run_root, run, checkpoint, records, lock in prepared:
            model, tokenizer, image_config, _ = load_checkpoint(checkpoint, device)
            model.eval()
            records = [{**r, "source_text": r["text"], "text": training_text(r["text"], model.config.text_policy)} for r in records]
            dataset = LineDataset(args.dataset_root, records)
            collator = Collator(image_config, tokenizer, model.config.capacity, model.config.eos_target)
            batch_size = run["training_options"]["batch_size"]
            batches = [collator([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
                       for start in range(0, len(dataset), batch_size)]
            outcomes = {}
            for name, method, repeats in methods:
                trials = []
                for repeat in range(repeats):
                    seed = run["training_options"]["seed"] + 200000 + repeat
                    generator = torch.Generator(device=device).manual_seed(seed)
                    predictions = []
                    for pixels, widths, _, rows in batches:
                        pixels, widths = pixels.to(device), widths.to(device)
                        if method is None:
                            ids, _ = model.sample(pixels, widths, steps=16, generator=generator)
                        else:
                            ids, _ = greedy_decode(model, pixels, widths, method=method, steps=16, generator=generator)
                        for r, row in zip(rows, ids.cpu().tolist()):
                            text, has_eos = tokenizer.decode(row)
                            predictions.append({"sample_id": r["sample_id"], "reference": r["text"], "source_reference": r["source_text"], "prediction": text, "has_eos": has_eos})
                    trials.append({"sampling_seed": seed if repeats > 1 else None,
                                   "strict": aggregate([(p["reference"], p["prediction"]) for p in predictions]),
                                   "raw_strict": aggregate([(p["source_reference"], p["prediction"]) for p in predictions]),
                                   "no_eos": sum(not p["has_eos"] for p in predictions), "predictions": predictions})
                outcomes[name] = trials
                row = {"model": variant, "decoding": name, "sampling_repeats": repeats,
                       "mean_cer_percent": 100 * statistics.mean(t["strict"]["cer"] for t in trials),
                       "mean_exact_crops": statistics.mean(t["strict"]["exact_lines"] for t in trials),
                       "samples": len(records), "no_eos_per_repeat": [t["no_eos"] for t in trials],
                       "cer_percent_per_repeat": [100 * t["strict"]["cer"] for t in trials]}
                summaries.append(row)
                print(json.dumps(row), flush=True)
            prior_path = run_root / "final_diagnosis.json"
            replay_matches = None
            if prior_path.exists() and run["training_options"]["sampling_steps"] == 16:
                prior = json.loads(prior_path.read_text(encoding="utf-8"))
                replay_matches = all(a["strict"] == b["strict"] for a, b in zip(outcomes["ancestral_16"], prior["conditions"]["correct"]["outcomes"]))
            report = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_hash(checkpoint / "model.pt"),
                      "dataset_fingerprint": lock["dataset_fingerprint"], "model_config": model.config.as_dict(),
                      "image_config": image_config.as_dict(), "reference_characters": sum(len(r["text"]) for r in records),
                      "original_sampler_matches_prior_metrics": replay_matches, "outcomes": outcomes,
                      "source_hashes": {str(p): file_hash(p) for p in (Path(__file__), Path(__file__).parents[1] / "diffusion/decoding_diagnostics.py")}}
            (args.output_dir / f"{variant}.json").write_bytes(json_bytes(report))
    report = {"evaluation": "same TRAIN memorization crops; no retraining, no held-out test evaluation",
              "definitions": {"ancestral_16": "Original categorical token sampling and random reveal, 16 passes",
                              "greedy_one_pass": "All MASK at t=1; argmax once; decode until predicted EOS",
                              "greedy_random_reveal_16": "Argmax characters, random reveal locations, 16 passes",
                              "greedy_confidence_16": "Argmax characters, deterministic confidence-ordered reveal, 16 passes"},
              "notes": "All methods receive images only at inference, with fixed capacity. Revealed tokens are retained. Random-reveal seeds are not paired reveal masks with the ancestral sampler. Deterministic methods are evaluated once.",
              "comparison": summaries}
    (args.output_dir / "comparison.json").write_bytes(json_bytes(report))
    print(f"\nSend this file: {args.output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
