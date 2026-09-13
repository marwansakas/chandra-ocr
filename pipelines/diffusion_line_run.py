"""Target-free line inference using an image-conditioned diffusion checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from line_runner_common import DEFAULT_INPUT, ROOT, find_line_images, prediction_dir, select_batch, write_prediction, write_run_files

sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint directory or run directory containing best.json/last.json")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, required=True, help="NEW prediction directory")
    parser.add_argument("--split-root", type=Path, help="Frozen split directory; omit to discover line images")
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--decoding", choices=("ancestral", "greedy_one_pass", "greedy_confidence"), default="ancestral",
                        help="Diffusion decoding rule: ancestral samples tokens; greedy rules use argmax without sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-provenance", action="store_true", help="Require checkpoint dataset/split/tokenizer to match --split-root")
    args = parser.parse_args()
    if args.steps < 1 or args.cpu_threads < 1:
        parser.error("--steps and --cpu-threads must be positive")
    args.input_root = args.input_root.resolve()
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)
    from thesis.diffusion.data import file_hash, load_frozen, resolve_relative
    if args.require_provenance and not args.split_root:
        parser.error("--require-provenance requires --split-root")
    records_by_path = {}
    split_hash = None
    split_lock = split_tokenizer = None
    if args.split_root:
        partitions, split_lock, split_tokenizer = load_frozen(args.split_root, args.input_root, verify_data=False)
        records_by_path = {resolve_relative(args.input_root, r["image_path"]): r for r in partitions[args.partition]}
        images = list(records_by_path)
        split_hash = file_hash(args.split_root / "split_lock.json")
    else:
        images = find_line_images(args.input_root)
    selected = select_batch(images, args.start_line, args.limit)
    if not selected:
        raise ValueError("No selected line images")
    if args.dry_run:
        print(json.dumps({"selected": len(selected), "images": [p.relative_to(args.input_root).as_posix() for _, p in selected]}, indent=2))
        return
    if args.output_root.exists():
        raise FileExistsError(f"Prediction output exists; choose a new --output-root: {args.output_root}")
    import torch
    from PIL import Image
    from thesis.diffusion.images import prepare_images
    from thesis.diffusion.decoding_diagnostics import greedy_decode
    from thesis.diffusion.runtime import checkpoint_path, get_device, load_checkpoint, runtime_info, seed_everything, verify_checkpoint_provenance
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    device = get_device(args.device)
    model, tokenizer, image_config, model_metadata = load_checkpoint(args.model, device)
    if args.require_provenance:
        verify_checkpoint_provenance(model_metadata, tokenizer, split_tokenizer, split_lock, split_hash)
    model.eval()
    model_name = "arabic_line_diffusion" if model.config.decoder_kind == "diffusion" else "matched_encoder_autoregressive"
    checkpoint = checkpoint_path(args.model)
    args.output_root.mkdir(parents=True, exist_ok=False)
    rows, started = [], time.perf_counter()
    for position, image_path in selected:
        destination = prediction_dir(args.output_root, args.input_root, image_path)
        sample_id = image_path.parent.relative_to(args.input_root).as_posix()
        row = {"line_position": position, "sample_id": sample_id,
               "image_path": image_path.relative_to(args.input_root).as_posix(), "image_path_base": "input_root",
               "prediction_path": (destination / "prediction.txt").relative_to(args.output_root).as_posix(),
               "prediction_path_base": "output_root", "model": model_name, "model_path": str(checkpoint),
               "steps": args.steps, "seed": args.seed + position, "status": "failed"}
        try:
            record = records_by_path.get(image_path)
            if record and file_hash(image_path) != record["image_sha256"]:
                raise ValueError("Image changed since dataset freeze")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            sample_started = time.perf_counter()
            with Image.open(image_path) as image:
                pixels, widths = prepare_images([image], image_config)
            generator = torch.Generator(device=device).manual_seed(args.seed + position)
            if args.decoding == "ancestral":
                ids, trace = model.sample(pixels.to(device), widths.to(device), steps=args.steps, generator=generator, return_trace=args.trace)
            else:
                method = "one_pass" if args.decoding == "greedy_one_pass" else "confidence"
                ids, trace = greedy_decode(model, pixels.to(device), widths.to(device), method=method,
                                           steps=args.steps, generator=generator, return_trace=args.trace)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            text, has_eos = tokenizer.decode(ids[0].tolist())
            row.update({"status": "completed", "text": text, "has_eos": has_eos,
                        "decoder_calls": (1 if args.decoding == "greedy_one_pass" else args.steps) if model.config.decoder_kind == "diffusion" else (len(text) + 1 if has_eos else model.config.capacity),
                        "preprocess_and_inference_seconds": time.perf_counter() - sample_started,
                        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
            write_prediction(destination, text, row)
            # Shared writer strips edge spaces; restore literal output for strict scoring.
            (destination / "prediction.txt").write_bytes((text + "\n").encode("utf-8"))
            if trace is not None:
                (destination / "trace.json").write_text(json.dumps({"tokens": tokenizer.tokens, "times": [1 - k / args.steps for k in range(args.steps + 1)] if model.config.decoder_kind == "diffusion" else None,
                                                                   "states": [step[0] for step in trace]}, ensure_ascii=False) + "\n", encoding="utf-8")
        except (OSError, ValueError, RuntimeError) as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "metadata.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps({"sample_id": sample_id, "status": row["status"], "error": row.get("error")}, ensure_ascii=False), flush=True)
    summary = {"model": model_name, "selected": len(selected), "completed": sum(r["status"] == "completed" for r in rows),
               "failed": sum(r["status"] == "failed" for r in rows), "skipped": 0,
               "no_eos": sum(r["status"] == "completed" and r.get("has_eos") is False for r in rows), "device": str(device), "batch_size": 1,
               "elapsed_seconds": time.perf_counter() - started, "runtime": runtime_info(), "seed": args.seed,
               "sampler": args.decoding if model.config.decoder_kind == "diffusion" else "autoregressive_greedy", "steps": args.steps,
               "checkpoint_sha256": file_hash(checkpoint / "model.pt"), "split_lock_sha256": split_hash,
               "input_root": str(args.input_root), "partition": args.partition if args.split_root else None,
               "model_metadata": model_metadata, "image_config": image_config.as_dict(),
               "timing_policy": "per sample: image read/resize/normalize, transfer, encoder and all decoder calls; excludes model load and output writes"}
    write_run_files(args.output_root, rows, summary)
    print(json.dumps({k: summary[k] for k in ("selected", "completed", "failed", "no_eos")}, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
