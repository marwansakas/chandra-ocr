"""Cache frozen TrOCR outputs, train a diffusion refiner, and evaluate held-out lines."""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from pretrained_ocr.metrics import normalize
from thesis.diffusion.data import audit_dataset, file_hash, freeze_line_split, freeze_splits, json_bytes
from thesis.diffusion.metrics import aggregate, correction_changes
from thesis.diffusion.runtime import get_device, runtime_info, seed_everything
from thesis.diffusion.tokenizer import CharacterTokenizer
from .cache import (DEFAULT_BASE, CachedCollator, CachedLines, frozen_data,
                    prepare_cache, read_json, save_json)
from .model import DiffusionRefiner, RefinerConfig


def summaries(rows, prediction):
    pairs = [(row["reference"], row[prediction]) for row in rows]
    return {"raw": aggregate(pairs),
            "arabic-light": aggregate([(normalize(a), normalize(b)) for a, b in pairs])}


def selection_score(metrics, normalization):
    value = metrics[normalization]["cer"]
    if value is None or not math.isfinite(value):
        raise ValueError("Selection partition has no scorable characters or a nonfinite CER")
    return value


def choose_strategy(baseline_cer, learned_cer, learned_mode):
    # Ties retain the cheaper original recognizer. Test labels never enter this decision.
    return learned_mode if learned_cer < baseline_cer else "baseline"


def prepare(args):
    if args.split_kind == "writer" and not args.writer_map:
        raise ValueError("Writer-isolated preparation requires --writer-map; filenames are not verified child IDs")
    writer_map = read_json(args.writer_map) if args.writer_map else None
    if writer_map is not None and not isinstance(writer_map, dict):
        raise ValueError("Writer map must be a JSON object mapping sample IDs to child IDs")
    audit = audit_dataset(args.dataset_root.resolve(), writer_map)
    if audit["issues"]:
        raise ValueError("Dataset audit needs review before splitting: " + json.dumps(audit["issues"][:20], ensure_ascii=False))
    freeze = freeze_splits if args.split_kind == "writer" else freeze_line_split
    lock = freeze(audit, args.split_root, seed=args.seed, train_ratio=args.train_ratio,
                  validation_ratio=args.validation_ratio)
    print(json.dumps({"counts": lock["counts"], "split_unit": lock["split_unit"], "capacity": lock["capacity"]}, indent=2))


@torch.no_grad()
def infer(model, dataset, tokenizer, device, *, batch_size, steps, seed):
    model.eval()
    batches = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=CachedCollator(tokenizer, model.config.capacity, targets=False))
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for visual, mask, draft, _, samples in batches:
        visual, mask, draft = visual.to(device), mask.to(device), draft.to(device)
        outputs, timings = {}, {}
        for mode in ("one_pass", "diffusion"):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            ids = model.sample(visual, mask, draft, steps=steps, mode=mode, generator=generator)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings[mode] = (time.perf_counter() - start) / len(samples)
            outputs[mode] = [tokenizer.decode(row) for row in ids.cpu().tolist()]
        for index, sample in enumerate(samples):
            record = sample["record"]
            row = {"sample_id": record["sample_id"], "reference": record["text"], "baseline": sample["draft"],
                   "base_has_eos": sample["base_has_eos"], "cached_base_seconds": sample["base_seconds"],
                   "capacity_overflow": len(record["text"]) + 1 > model.config.capacity,
                   "draft_clipped": len(sample["draft"]) + 1 > model.config.capacity,
                   "draft_unknown_codepoints": sorted(set(sample["draft"]) - set(tokenizer.characters)),
                   "target_unknown_codepoints": sorted(set(record["text"]) - set(tokenizer.characters))}
            for mode in outputs:
                row[mode], row[mode + "_has_eos"] = outputs[mode][index]
                row[mode + "_seconds"] = timings[mode]
            rows.append(row)
    return rows


@torch.no_grad()
def validation_loss(model, batches, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    for visual, mask, draft, targets, samples in batches:
        loss = model(visual.to(device), mask.to(device), draft.to(device), targets.to(device), generator=generator)
        total += float(loss) * len(samples)
        count += len(samples)
    return total / count


def save_weights(path, model, epoch):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "epoch": epoch}, temporary)
    temporary.replace(path)


def train(args):
    if args.output_dir.exists():
        raise FileExistsError("Run directory already exists; choose a new --output-dir")
    parts, lock, tokenizer = frozen_data(args.dataset_root, args.split_root)
    split_sha = file_hash(args.split_root / "split_lock.json")
    train_data = CachedLines(args.cache_root, "train", parts["train"], split_sha)
    validation_data = CachedLines(args.cache_root, "validation", parts["validation"], split_sha,
                                  expected_signature=train_data.base["signature"])
    if validation_data.visual_dim != train_data.visual_dim:
        raise ValueError("Train/validation visual dimensions differ")
    for partition in ("train", "validation"):
        for record in parts[partition]:
            tokenizer.encode(record["text"], lock["capacity"])
    device = get_device(args.device)
    seed_everything(args.seed)
    config = RefinerConfig(len(tokenizer), lock["capacity"], train_data.visual_dim,
                           d_model=args.d_model, heads=args.heads, layers=args.layers, dropout=args.dropout,
                           diffusion_steps=args.diffusion_steps, full_mask_probability=args.full_mask_probability,
                           draft_dropout=args.draft_dropout)
    model = DiffusionRefiner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    collator = CachedCollator(tokenizer, config.capacity)
    batches = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
                                         generator=torch.Generator().manual_seed(args.seed))
    validation_batches = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, collate_fn=collator)
    baseline_rows = [{"reference": row["record"]["text"], "baseline": row["draft"]} for row in validation_data]
    baseline_metrics = summaries(baseline_rows, "baseline")
    baseline_cer = selection_score(baseline_metrics, args.selection_normalization)
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    info = {"format_version": 1, "options": options, "split_lock_sha256": split_sha,
            "dataset_fingerprint": lock["dataset_fingerprint"], "base_signature": train_data.base["signature"],
            "base_source": train_data.base["source"], "split_unit": lock["split_unit"],
            "target_policy": "literal characters plus one EOS; PAD excluded from loss and replaced by MASK in decoder input",
            "trainable_parameters": sum(p.numel() for p in model.parameters()), "runtime": runtime_info(),
            "source_hashes": {p.name: file_hash(p) for p in source_files},
            "validation_baseline": baseline_metrics}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer.save(args.output_dir / "tokenizer.json")
    save_json(args.output_dir / "config.json", config.as_dict())
    save_json(args.output_dir / "run.json", info)
    best_cer, best_epoch, best_mode, stale = math.inf, None, None, 0
    print(f"Trainable parameters: {info['trainable_parameters']:,}; frozen TrOCR: cached", flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for visual, mask, draft, targets, samples in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = model(visual.to(device), mask.to(device), draft.to(device), targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(samples)
            count += len(samples)
        val_loss = validation_loss(model, validation_batches, device, args.seed + 1000)
        predictions = infer(model, validation_data, tokenizer, device, batch_size=args.batch_size,
                            steps=args.sampling_steps, seed=args.seed + 2000)
        metrics = {mode: summaries(predictions, mode) for mode in ("one_pass", "diffusion")}
        candidate = min(metrics, key=lambda mode: selection_score(metrics[mode], args.selection_normalization))
        candidate_cer = selection_score(metrics[candidate], args.selection_normalization)
        if candidate_cer < best_cer:
            best_cer, best_epoch, best_mode, stale = candidate_cer, epoch, candidate, 0
            save_weights(args.output_dir / "best.pt", model, epoch)
            save_json(args.output_dir / "best_validation.json", {"epoch": epoch, "metrics": metrics, "samples": predictions})
        else:
            stale += 1
        save_weights(args.output_dir / "last.pt", model, epoch)
        row = {"epoch": epoch, "train_loss": total / count, "validation_loss": val_loss,
               "validation": metrics, "seconds": time.perf_counter() - start}
        with (args.output_dir / "training_metrics.jsonl").open("ab") as handle:
            handle.write(json_bytes(row))
        print(f"Epoch {epoch:03d} | loss {total/count:.4f} | val loss {val_loss:.4f} | "
              f"val CER {100*candidate_cer:.2f}% ({candidate}) | {row['seconds']:.1f}s", flush=True)
        if args.patience and stale >= args.patience:
            break
    strategy = choose_strategy(baseline_cer, best_cer, best_mode)
    selection = {"strategy": strategy, "learned_mode": best_mode, "epoch": best_epoch,
                 "normalization": args.selection_normalization, "baseline_validation_cer": baseline_cer,
                 "learned_validation_cer": best_cer, "sampling_steps": args.sampling_steps,
                 "seed": args.seed + 2000, "batch_size": args.batch_size,
                 "policy": "lowest validation CER; one_pass wins learned-mode ties; baseline wins baseline ties"}
    save_json(args.output_dir / "selection.json", selection)
    names = ("config.json", "tokenizer.json", "run.json", "best.pt", "selection.json", "best_validation.json")
    save_json(args.output_dir / "completed.json", {"hashes": {name: file_hash(args.output_dir / name) for name in names}})
    print(json.dumps(selection, indent=2), flush=True)


def load_run(run_dir, device):
    complete = read_json(run_dir / "completed.json")
    for name in ("config.json", "tokenizer.json", "run.json", "best.pt", "selection.json", "best_validation.json"):
        if file_hash(run_dir / name) != complete["hashes"][name]:
            raise ValueError("Run artifact integrity mismatch: " + name)
    config = RefinerConfig(**read_json(run_dir / "config.json"))
    tokenizer = CharacterTokenizer.load(run_dir / "tokenizer.json")
    if len(tokenizer) != config.vocab_size:
        raise ValueError("Checkpoint vocabulary mismatch")
    model = DiffusionRefiner(config)
    weights = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights["model"], strict=True)
    return model.to(device).eval(), tokenizer, read_json(run_dir / "run.json"), read_json(run_dir / "selection.json")


def evaluate(args):
    if args.output_dir.exists():
        raise FileExistsError("Evaluation directory exists; choose a new output directory")
    parts, lock, frozen_tokenizer = frozen_data(args.dataset_root, args.split_root)
    device = get_device(args.device)
    model, tokenizer, run, selection = load_run(args.run_dir, device)
    split_sha = file_hash(args.split_root / "split_lock.json")
    if (split_sha != run["split_lock_sha256"] or lock["dataset_fingerprint"] != run["dataset_fingerprint"]
            or tokenizer.as_dict() != frozen_tokenizer.as_dict() or model.config.capacity != lock["capacity"]):
        raise ValueError("Checkpoint/frozen evaluation split mismatch")
    dataset = CachedLines(args.cache_root, args.partition, parts[args.partition], split_sha,
                          expected_signature=run["base_signature"])
    if dataset.visual_dim != model.config.visual_dim:
        raise ValueError("Checkpoint/cache visual dimension mismatch")
    rows = infer(model, dataset, tokenizer, device, batch_size=selection["batch_size"],
                 steps=selection["sampling_steps"], seed=selection["seed"])
    for row in rows:
        row["selected"] = row[selection["strategy"]]
        row["correction"] = correction_changes(row["reference"], row["baseline"], row["selected"])
    metrics = {mode: summaries(rows, mode) for mode in ("baseline", "one_pass", "diffusion", "selected")}
    report = {"partition": args.partition, "selection": selection, "metrics": metrics,
              "split_lock_sha256": split_sha, "base_signature": run["base_signature"],
              "checkpoint_sha256": file_hash(args.run_dir / "best.pt"), "split_unit": lock["split_unit"],
              "lines": len(rows), "capacity_overflow": sum(row["capacity_overflow"] for row in rows),
              "unknown_target_lines": sum(bool(row["target_unknown_codepoints"]) for row in rows),
              "base_without_eos": sum(not row["base_has_eos"] for row in rows),
              "draft_clipped_lines": sum(row["draft_clipped"] for row in rows),
              "correction": {key: sum(row["correction"][key] for row in rows) for key in rows[0]["correction"]},
              "mean_cached_base_seconds": sum(row["cached_base_seconds"] for row in rows) / len(rows),
              "mean_refiner_seconds": {mode: sum(row[mode + "_seconds"] for row in rows) / len(rows)
                                       for mode in ("one_pass", "diffusion")},
              "timing_note": "base measured during cache creation; refiner-only timings exclude cache IO and include first batch; not live end-to-end latency",
              "runtime": runtime_info()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_json(args.output_dir / "evaluation.json", report)
    (args.output_dir / "predictions.jsonl").write_bytes(b"".join(json_bytes(row) for row in rows))
    lines = ["# TrOCR diffusion refiner evaluation", "", f"Partition: {args.partition}; validation-selected strategy: {selection['strategy']}", "",
             "| Model | Raw CER | Raw WER | Arabic-light CER | Exact lines (raw) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    def percent(value):
        return "n/a" if value is None else f"{100 * value:.2f}%"
    for mode, result in metrics.items():
        raw = result["raw"]
        lines.append(f"| {mode} | {percent(raw['cer'])} | {percent(raw['wer'])} | {percent(result['arabic-light']['cer'])} | {raw['exact_lines']}/{len(rows)} |")
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare", help="Create a NEW three-way split; use existing frozen splits when available")
    preparation.add_argument("--dataset-root", type=Path, required=True)
    preparation.add_argument("--split-root", type=Path, required=True)
    preparation.add_argument("--split-kind", choices=("writer", "line"), default="writer")
    preparation.add_argument("--writer-map", type=Path)
    preparation.add_argument("--train-ratio", type=float, default=0.7)
    preparation.add_argument("--validation-ratio", type=float, default=0.1)
    preparation.add_argument("--seed", type=int, default=42)
    for name in ("cache", "train", "evaluate"):
        child = sub.add_parser(name)
        child.add_argument("--dataset-root", type=Path, required=True)
        child.add_argument("--split-root", type=Path, required=True)
        child.add_argument("--cache-root", type=Path, required=True)
        child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        child.add_argument("--cpu-threads", type=int, default=2)
        if name == "cache":
            child.add_argument("--base-model", default=DEFAULT_BASE)
            child.add_argument("--base-revision")
            source = child.add_mutually_exclusive_group(required=True)
            source.add_argument("--base-training-manifest", type=Path)
            source.add_argument("--external-base", action="store_true", help="Declare base never trained on project samples; not for your fine-tuned checkpoint")
            child.add_argument("--partition", nargs="+", choices=("train", "validation", "test"), default=["train", "validation"])
            child.add_argument("--batch-size", type=int, default=4)
            child.add_argument("--max-new-tokens", type=int, default=128)
            child.add_argument("--resume", action="store_true")
        else:
            child.add_argument("--output-dir", type=Path, required=True)
        if name == "train":
            child.add_argument("--epochs", type=int, default=30)
            child.add_argument("--batch-size", type=int, default=8)
            child.add_argument("--d-model", type=int, default=192)
            child.add_argument("--heads", type=int, default=6)
            child.add_argument("--layers", type=int, default=3)
            child.add_argument("--dropout", type=float, default=0.1)
            child.add_argument("--diffusion-steps", type=int, default=256)
            child.add_argument("--sampling-steps", type=int, default=8)
            child.add_argument("--full-mask-probability", type=float, default=0.25)
            child.add_argument("--draft-dropout", type=float, default=0.5)
            child.add_argument("--learning-rate", type=float, default=3e-4)
            child.add_argument("--weight-decay", type=float, default=0.01)
            child.add_argument("--patience", type=int, default=7)
            child.add_argument("--seed", type=int, default=42)
            child.add_argument("--selection-normalization", choices=("raw", "arabic-light"), default="arabic-light")
        if name == "evaluate":
            child.add_argument("--run-dir", type=Path, required=True)
            child.add_argument("--partition", choices=("validation", "test"), default="test")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
        return
    for key in ("cpu_threads", "batch_size", "max_new_tokens", "epochs", "d_model", "heads", "layers", "diffusion_steps", "sampling_steps"):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(key.replace("_", "-") + " must be positive")
    if args.command == "train":
        if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not math.isfinite(args.weight_decay)
                or args.weight_decay < 0 or args.patience < 0):
            parser.error("Invalid optimizer or patience setting")
    torch.set_num_threads(args.cpu_threads)
    {"cache": prepare_cache, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
