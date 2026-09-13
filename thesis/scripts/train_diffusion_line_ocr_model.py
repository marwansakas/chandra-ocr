"""Train the absorbing diffusion baseline on frozen train/validation partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW run directory; never overwrites a run")
    parser.add_argument("--resume", type=Path, help="Checkpoint to continue in a NEW output directory; --epochs is total epochs")
    parser.add_argument("--vit-checkpoint", type=Path, help="Optional LOCAL ViTModel + image processor directory; no network download")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--max-width", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--decoder-kind", choices=("diffusion", "autoregressive"), default="diffusion", help="Matched-encoder autoregressive control is a separate experiment")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=256)
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--evaluation-decoding", choices=("ancestral", "greedy_one_pass"), default="ancestral", help="Decoder used for epoch metrics and best-checkpoint selection")
    parser.add_argument("--eos-target", choices=("repeat", "single"), default=None, help="Default repeat; single excludes positions after the first EOS from loss")
    parser.add_argument("--full-mask-probability", type=float, default=None, help="Default 0; p mixes full-mask CE with weight p and original diffusion objective with weight 1-p")
    parser.add_argument("--ink-mode", choices=("original", "dominant_soft"), default=None, help="Optional conservative color emphasis, saved with checkpoint")
    parser.add_argument("--augmentation", choices=("none", "mild_v1", "trocr_views_v1", "trocr_all_views_v1"), default="none", help="Training-only image augmentation policy; validation and test stay deterministic")
    parser.add_argument("--epoch-log-format", choices=("json", "compact"), default="json", help="Console output only; complete epoch records are always saved to training_metrics.jsonl")
    parser.add_argument("--text-policy", choices=("literal", "no_harakat_v1"), default=None, help="Derived targets only: omit U+064B..U+0652; retain original labels and raw scores")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save periodic, best and final checkpoints; metrics still recorded each epoch")
    parser.add_argument("--diagnostics-every", type=int, default=0, help="Full-mask character/first-EOS/tail-EOS diagnostics on validation or overfit crops; 0 disables")
    parser.add_argument("--validation-loss-seed", type=int, default=300042, help="Fixed generator seed used for comparable validation/train denoising loss")
    parser.add_argument("--train-loss-samples", type=int, default=128, help="Fixed TRAIN-only subset size for evaluation-mode loss diagnostics")
    parser.add_argument("--early-stopping-patience", type=int, default=30, help="Validation-CER evaluations without a qualifying improvement; 0 disables")
    parser.add_argument("--selection-min-delta", type=float, default=0.0, help="Minimum strict-CER decrease that resets early-stop patience; absolute-best checkpoint still records every decrease")
    parser.add_argument("--exact-line-cer-band", type=float, default=0.005,
                        help="Absolute CER band for selecting the checkpoint with most exact validation lines; 0.005 is 0.5 percentage points")
    parser.add_argument("--save-traces", action="store_true", help="Include all sampling token states in saved predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overfit-samples", type=int, help="Diagnostic mode: fit/evaluate only the first N TRAIN records; never a validation result")
    parser.add_argument("--overfit-manifest", type=Path, help="Use these frozen TRAIN sample IDs instead of first N; requires --overfit-samples")
    parser.add_argument("--monitor-train", action="store_true", help="Line pilot only: score all TRAIN crops each epoch, retaining fixed-final checkpoint selection")
    parser.add_argument("--check-only", action="store_true", help="Verify inputs, dimensions and configuration without allocating a model or writing output")
    args = parser.parse_args()
    for key in ("epochs", "batch_size", "height", "max_width", "patch_size", "d_model", "heads", "encoder_layers", "decoder_layers", "diffusion_steps", "sampling_steps", "cpu_threads"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.workers < 0 or args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("Invalid workers, optimizer values or dropout")
    if args.overfit_samples is not None and args.overfit_samples < 1:
        parser.error("--overfit-samples must be positive")
    if args.overfit_manifest and not args.overfit_samples:
        parser.error("--overfit-manifest requires --overfit-samples")
    if args.resume and args.vit_checkpoint:
        parser.error("Resume already includes the visual encoder; omit --vit-checkpoint")
    if args.full_mask_probability is not None and not 0 <= args.full_mask_probability <= 1:
        parser.error("--full-mask-probability must be in [0, 1]")
    if args.checkpoint_every < 1 or args.diagnostics_every < 0 or args.train_loss_samples < 1 or args.early_stopping_patience < 0 or args.selection_min_delta < 0 or args.exact_line_cer_band < 0:
        parser.error("Invalid checkpoint or diagnostic interval")
    return args


def format_epoch_progress(row: dict) -> str:
    """Format the six human-facing epoch fields without dropping saved diagnostics."""
    strict = (row.get("metrics") or {}).get("strict") or {}

    def number(value, digits: int) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    cer = strict.get("cer")
    cer_text = "n/a" if cer is None else f"{100 * cer:.3f}%"
    exact = strict.get("exact_lines")
    exact_text = "n/a" if exact is None else str(exact)
    return (
        f"Epoch {int(row['epoch']):03d} | Train loss {number(row.get('train_loss'), 6)} | "
        f"Validation loss {number(row.get('validation_loss'), 6)} | CER {cer_text} | "
        f"Exact lines {exact_text} | Time {number(row.get('elapsed_seconds'), 1)}s"
    )


def main():
    args = parse_args()
    import torch
    from thesis.diffusion.data import file_hash, json_bytes, load_frozen
    from thesis.diffusion.images import (AugmentationConfig, ImageConfig, augmentation_seed,
                                         augmentation_view_index, augment_image, resized_width)
    from thesis.diffusion.metrics import aggregate
    from thesis.diffusion.model import DiffusionRecognizer, ModelConfig
    from thesis.diffusion.runtime import (ROOT, Collator, LineDataset, checkpoint_path, get_device, load_checkpoint,
                                          load_training_state, restore_rng, rng_state, runtime_info, save_checkpoint,
                                          seed_everything, seed_worker)
    from thesis.diffusion.runtime import all_mask_loss, fixed_corruption_loss, full_mask_diagnostics
    from thesis.diffusion.tokenizer import training_text, HARAKAT_V1
    from thesis.diffusion.subsets import read_subset
    from thesis.diffusion.decoding_diagnostics import greedy_decode
    from thesis.diffusion.selection import checkpoint_candidate, update_exact_line_frontier

    if args.output_dir.exists() and not args.check_only:
        raise FileExistsError(f"Run directory exists; choose a new --output-dir: {args.output_dir}")
    torch.set_num_threads(args.cpu_threads)
    device = get_device(args.device)
    seed_everything(args.seed)
    partitions, lock, tokenizer = load_frozen(args.split_root, args.dataset_root)
    split_hash = file_hash(args.split_root / "split_lock.json")
    line_pilot = lock.get("split_unit") == "line_pilot" and not args.overfit_samples
    if args.monitor_train and not line_pilot:
        raise ValueError("--monitor-train requires a line pilot without --overfit-samples")
    train_records = partitions["train"]
    validation_records = partitions["validation"]
    if args.monitor_train:
        validation_records = train_records
    if args.overfit_samples:
        if args.overfit_samples > len(train_records):
            raise ValueError("--overfit-samples exceeds frozen training records")
        train_records = (read_subset(args.overfit_manifest, train_records, lock["dataset_fingerprint"], split_hash, args.overfit_samples)
                         if args.overfit_manifest else train_records[:args.overfit_samples])
        validation_records = train_records

    encoder_source = None
    model = None
    if args.resume:
        model, saved_tokenizer, image_config, previous = load_checkpoint(args.resume, device)
        config = model.config
        if saved_tokenizer.as_dict() != tokenizer.as_dict() or config.capacity != lock["capacity"]:
            raise ValueError("Resume vocabulary/capacity differs from frozen data")
        encoder_source = previous.get("encoder_source")
        for name, saved in (("eos_target", config.eos_target), ("full_mask_probability", config.full_mask_probability), ("ink_mode", image_config.ink_mode), ("text_policy", config.text_policy)):
            if getattr(args, name) is not None and getattr(args, name) != saved:
                raise ValueError(f"Resume cannot change {name}; use a fresh run")
    else:
        vit_config = None
        image_config = ImageConfig(args.height, args.max_width, args.patch_size)
        if args.vit_checkpoint:
            from transformers import AutoImageProcessor, ViTConfig
            if not args.vit_checkpoint.is_dir():
                raise FileNotFoundError(args.vit_checkpoint)
            vit = ViTConfig.from_pretrained(args.vit_checkpoint, local_files_only=True)
            if vit.model_type != "vit":
                raise ValueError("Only a local ViTModel checkpoint is supported by this adapter")
            processor = AutoImageProcessor.from_pretrained(args.vit_checkpoint, local_files_only=True, use_fast=False)
            vit_config = vit.to_dict()
            image_config = ImageConfig(args.height, args.max_width, vit.patch_size, tuple(processor.image_mean), tuple(processor.image_std))
            encoder_source = {"path": str(args.vit_checkpoint), "files": {p.name: file_hash(p) for p in sorted(args.vit_checkpoint.iterdir()) if p.is_file()}}
        config = ModelConfig(len(tokenizer), lock["capacity"], args.d_model, args.heads, args.encoder_layers,
                             args.decoder_layers, image_config.patch_size, args.dropout, args.diffusion_steps,
                             "vit" if vit_config else "patch", vit_config, args.decoder_kind,
                             eos_target=args.eos_target or "repeat", full_mask_probability=args.full_mask_probability or 0.0,
                             text_policy=args.text_policy or "literal",
                             excluded_token_ids=tuple(sorted(tokenizer.ids[c] for c in HARAKAT_V1 if c in tokenizer.ids)) if args.text_policy == "no_harakat_v1" else ())
        image_config.ink_mode = args.ink_mode or "original"
    if args.evaluation_decoding == "greedy_one_pass" and config.decoder_kind != "diffusion":
        raise ValueError("greedy_one_pass evaluation requires a diffusion model")
    def derive(records):
        return [{**r, "source_text": r["text"], "text": training_text(r["text"], config.text_policy)} for r in records]
    train_records, validation_records = derive(train_records), derive(validation_records)
    # Only train/validation control capacity and preprocessing choices; test is untouched here.
    for name, records in (("train", train_records), ("validation", validation_records)):
        for record in records:
            try:
                tokenizer.encode(record["text"], config.capacity)
                resized_width(record["width"], record["height"], image_config)
            except ValueError as exc:
                raise ValueError(f"{name}/{record['sample_id']}: {exc}") from exc
    if args.augmentation != "none":
        # Geometry changes with the virtual safety canvas, so verify every
        # TRAIN crop before a long run. Validation/test never receive it.
        from PIL import Image
        augmentation = AugmentationConfig(args.augmentation)
        for record in train_records:
            with Image.open(args.dataset_root / record["image_path"]) as image:
                source = image.convert("RGB")
            epochs_to_check = (1, 2, 3) if args.augmentation in ("trocr_views_v1", "trocr_all_views_v1") else (1,)
            for augmentation_epoch in epochs_to_check:
                view_index = augmentation_view_index(augmentation_epoch) if args.augmentation in ("trocr_views_v1", "trocr_all_views_v1") else None
                augmented, _ = augment_image(
                    source,
                    augmentation,
                    augmentation_seed(record["sample_id"], augmentation_epoch, args.seed),
                    view_index=view_index,
                )
                try:
                    resized_width(*augmented.size, image_config)
                except ValueError as exc:
                    raise ValueError(f"augmentation/train/{record['sample_id']}/view-{augmentation_epoch}: {exc}") from exc
    train_loss_count = min(args.train_loss_samples, len(train_records))
    train_loss_records = sorted(random.Random(args.seed + 400000).sample(train_records, train_loss_count),
                                key=lambda row: row["sample_id"])
    training_options = {key: getattr(args, key) for key in ("seed", "batch_size", "learning_rate", "weight_decay", "sampling_steps", "overfit_samples", "workers", "validation_loss_seed", "train_loss_samples", "early_stopping_patience", "selection_min_delta", "augmentation")}
    training_views_per_sample = 3 if args.augmentation == "trocr_all_views_v1" else 1
    training_examples_per_epoch = len(train_records) * training_views_per_sample
    if training_views_per_sample > 1:
        training_options["training_views_per_sample"] = training_views_per_sample
        training_options["training_examples_per_epoch"] = training_examples_per_epoch
    training_options["train_loss_effective_samples"] = train_loss_count
    if args.monitor_train:
        training_options["monitor_train"] = True
    if args.overfit_manifest:
        training_options["training_subset_sha256"] = file_hash(args.overfit_manifest)
    if args.evaluation_decoding != "ancestral":
        training_options["evaluation_decoding"] = args.evaluation_decoding
    checkpoint_selection_policy = None if line_pilot or args.overfit_samples else {
        "absolute_best": "lowest validation strict CER",
        "near_tie": "most exact validation lines within the CER band; ties use lower CER then earlier epoch",
        "exact_line_cer_band": args.exact_line_cer_band,
        "exact_line_cer_band_percentage_points": 100 * args.exact_line_cer_band,
        "test_used_for_selection": False,
    }
    if args.resume and (previous["split_lock_sha256"] != split_hash or previous["training_options"] != training_options):
        raise ValueError("Resume split or training settings changed; retain original settings for reproducible continuation")
    if args.check_only:
        print(json.dumps({"checked": True, "model": config.as_dict(), "image": image_config.as_dict(), "partitions": lock["counts"],
                          "training_samples": len(train_records), "training_views_per_sample": training_views_per_sample,
                          "training_examples_per_epoch": training_examples_per_epoch,
                           "evaluation_samples": len(validation_records), "checkpoint_selection_policy": checkpoint_selection_policy,
                           "evaluation_decoding": args.evaluation_decoding, "training_options": training_options,
                           "train_loss_subset": {"samples": len(train_loss_records), "seed": args.seed + 400000},
                          "mode": "overfit" if args.overfit_samples else "fixed_epochs_line_pilot" if line_pilot else "validation"}, indent=2))
        return
    if model is None:
        model = DiffusionRecognizer(config)
        if args.vit_checkpoint:
            from transformers import ViTModel
            pretrained, loading = ViTModel.from_pretrained(args.vit_checkpoint, local_files_only=True,
                                                          add_pooling_layer=False, output_loading_info=True)
            if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
                raise ValueError(f"Incomplete pretrained ViT load: {loading}")
            model.encoder.vit.load_state_dict(pretrained.state_dict(), strict=True)
            encoder_source["loading_info"] = loading
            encoder_source["loaded_tensors"] = sorted(pretrained.state_dict())
            del pretrained
        model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.resume:
        state = load_training_state(args.resume)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        restore_rng(state["rng"])
        if args.epochs <= start_epoch:
            raise ValueError(f"Resume completed {start_epoch} epochs; --epochs must be larger")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.split_root, args.output_dir / "splits")
    if args.overfit_manifest:
        shutil.copy2(args.overfit_manifest, args.output_dir / "training_subset.json")
    provenance = runtime_info()
    for relative in provenance["source_hashes"]:
        destination = args.output_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    run = {"runtime": provenance, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "training_options": training_options, "split_lock_sha256": split_hash,
           "epoch_metrics_partition": "train" if args.monitor_train or args.overfit_samples else "validation",
           "dataset_fingerprint": lock["dataset_fingerprint"], "encoder_source": encoder_source,
           "model_config": config.as_dict(), "image_config": image_config.as_dict(),
           "objective": "mixture: (1-p)*uniform-time masked CE/t + p*full-mask CE" if config.full_mask_probability else "uniform-time masked CE/t",
            "target_policy": {"text": config.text_policy, "eos": config.eos_target,
                             "changed_training_labels": sum(r["source_text"] != r["text"] for r in train_records),
                              "source_files": "immutable; derived targets preserve tatweel, spelling and internal line breaks"},
            "training_augmentation": AugmentationConfig(args.augmentation).as_dict(),
           "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "mode": "training_subset_overfit_diagnostic" if args.overfit_samples else "fixed_epochs_line_pilot" if line_pilot else "validation_selected_training",
           "resume_checkpoint": str(checkpoint_path(args.resume)) if args.resume else None,
           "sampler": ("greedy_one_pass" if args.evaluation_decoding == "greedy_one_pass" else "absorbing_ancestral_categorical") if config.decoder_kind == "diffusion" else "autoregressive_greedy", "precision": "float32",
           "selection_metric": "strict CER on training subset" if args.overfit_samples else "fixed final epoch; test never evaluated during training" if line_pilot else "strict CER on validation",
           "checkpoint_selection_policy": checkpoint_selection_policy}
    (args.output_dir / "run.json").write_bytes(json_bytes(run))
    (args.output_dir / "training_targets.jsonl").write_bytes(b"".join(json_bytes({"sample_id": r["sample_id"], "source_text": r["source_text"], "target_text": r["text"], "policy": config.text_policy}) for r in train_records))
    (args.output_dir / "train_loss_subset.json").write_bytes(json_bytes({"partition": "train", "seed": args.seed + 400000,
        "sample_ids": [r["sample_id"] for r in train_loss_records], "dataset_fingerprint": lock["dataset_fingerprint"],
        "split_lock_sha256": split_hash, "purpose": "fixed evaluation-mode loss diagnostics; never checkpoint selection"}))
    training_collator = Collator(image_config, tokenizer, config.capacity, eos_target=config.eos_target,
                                 augmentation=AugmentationConfig(args.augmentation), augmentation_seed_base=args.seed)
    evaluation_collator = Collator(image_config, tokenizer, config.capacity, eos_target=config.eos_target)
    validation_loader = torch.utils.data.DataLoader(LineDataset(args.dataset_root, validation_records), batch_size=args.batch_size,
                                                    collate_fn=evaluation_collator, num_workers=args.workers, worker_init_fn=seed_worker,
                                                    generator=torch.Generator().manual_seed(args.seed + 100000))
    train_loss_loader = torch.utils.data.DataLoader(LineDataset(args.dataset_root, train_loss_records), batch_size=args.batch_size,
                                                     collate_fn=evaluation_collator, num_workers=args.workers, worker_init_fn=seed_worker,
                                                     generator=torch.Generator().manual_seed(args.seed + 500000))
    best_cer = float("inf")
    best_loss = float("inf")
    selection_candidates = []
    epochs_without_cer_improvement = 0
    if args.resume:
        # Keep the previous best eligible; a worse resumed epoch must not replace it.
        parent_run = checkpoint_path(args.resume).parent
        if not line_pilot and (parent_run / "best.json").is_file():
            parent_best = json.loads((parent_run / "best.json").read_text(encoding="utf-8"))
            best_path = checkpoint_path(parent_run)
            _, _, _, best_metadata = load_checkpoint(best_path)
            if best_metadata["epoch"] > start_epoch:
                raise ValueError("Parent best is later than resumed checkpoint; resume the latest checkpoint")
            shutil.copytree(best_path, args.output_dir / "inherited-best")
            best_cer = parent_best["strict_cer"]
            inherited_best = {**parent_best, "checkpoint": "inherited-best"}
            (args.output_dir / "best.json").write_bytes(json_bytes(inherited_best))
            if checkpoint_selection_policy is not None:
                state_path = parent_run / "checkpoint_selection.json"
                parent_candidates = json.loads(state_path.read_text(encoding="utf-8"))["candidates"] if state_path.is_file() else []
                if not parent_candidates:
                    strict = best_metadata.get("metrics", {}).get("metrics", {}).get("strict", {})
                    parent_candidates = [checkpoint_candidate(parent_best["checkpoint"], parent_best.get("epoch", best_metadata["epoch"]),
                                                                best_cer, strict.get("exact_lines", 0))]
                copied = {parent_best["checkpoint"]: "inherited-best"}
                for item in parent_candidates:
                    if item["epoch"] > start_epoch:
                        raise ValueError("Parent checkpoint-selection candidate is later than resumed checkpoint; resume the latest checkpoint")
                    source_name = item["checkpoint"]
                    destination_name = copied.get(source_name)
                    if destination_name is None:
                        destination_name = f"inherited-selection-{item['epoch']:04d}"
                        shutil.copytree(parent_run / source_name, args.output_dir / destination_name)
                        copied[source_name] = destination_name
                    selection_candidates.append({**item, "checkpoint": destination_name})
        # Loading the inherited model constructs modules and consumes RNG state.
        restore_rng(state["rng"])
    training_dataset = LineDataset(args.dataset_root, train_records, views_per_sample=training_views_per_sample)
    updates_per_epoch = (len(training_dataset) + args.batch_size - 1) // args.batch_size
    for epoch in range(start_epoch + 1, args.epochs + 1):
        training_collator.epoch = epoch
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size,
                                                  shuffle=True, collate_fn=training_collator, num_workers=args.workers,
                                                  worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train()
        started, total_loss, seen = time.perf_counter(), 0.0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for pixels, widths, targets, records in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(pixels.to(device), widths.to(device), targets.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(records)
            seen += len(records)
        model.eval()
        validation_loss = validation_all_mask_loss = train_fixed_loss = train_all_mask_loss = None
        if validation_records:
            validation_loss = fixed_corruption_loss(model, validation_loader, device, args.validation_loss_seed)
            validation_all_mask_loss = all_mask_loss(model, validation_loader, device)
            train_fixed_loss = fixed_corruption_loss(model, train_loss_loader, device, args.validation_loss_seed)
            train_all_mask_loss = all_mask_loss(model, train_loss_loader, device)
        generator = torch.Generator(device=device).manual_seed(args.seed + 200000)
        predictions = []
        for pixels, widths, _, records in validation_loader:
            if args.evaluation_decoding == "greedy_one_pass":
                tokens, trace = greedy_decode(model, pixels.to(device), widths.to(device), method="one_pass", return_trace=args.save_traces)
            else:
                tokens, trace = model.sample(pixels.to(device), widths.to(device), steps=args.sampling_steps, generator=generator, return_trace=args.save_traces)
            for index, (ids, record) in enumerate(zip(tokens.cpu().tolist(), records)):
                text, has_eos = tokenizer.decode(ids)
                predictions.append({"sample_id": record["sample_id"], "reference": record["text"], "text": text,
                                    "source_reference": record["source_text"],
                                    "has_eos": has_eos, **({"trace_token_ids": [step[index] for step in trace]} if trace else {})})
        pairs = [(p["reference"], p["text"]) for p in predictions]
        scores = {"strict": aggregate(pairs), "raw_strict": aggregate([(p["source_reference"], p["text"]) for p in predictions]),
                  "historical_normalized": aggregate(pairs, historical=True)} if pairs else None
        row = {"epoch": epoch, "train_loss": total_loss / seen, "metrics": scores,
               "validation_loss": validation_loss, "validation_loss_schedule": {"seed": args.validation_loss_seed, "type": "fixed_uniform_diffusion_corruption"} if validation_loss is not None else None,
               "validation_all_mask_loss": validation_all_mask_loss, "train_fixed_corruption_loss": train_fixed_loss,
               "train_all_mask_loss": train_all_mask_loss,
               "metrics_partition": run["epoch_metrics_partition"],
               "evaluation_decoding": run["sampler"],
               "optimizer_updates": epoch * updates_per_epoch,
               "mean_prediction_length": sum(len(p["text"]) for p in predictions) / len(predictions) if predictions else None,
               "mean_reference_length": sum(len(p["reference"]) for p in predictions) / len(predictions) if predictions else None,
               "no_eos": sum(not p["has_eos"] for p in predictions), "elapsed_seconds": time.perf_counter() - started,
               "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        if args.diagnostics_every and validation_records and (epoch % args.diagnostics_every == 0 or epoch == args.epochs):
            row["full_mask_diagnostics"] = full_mask_diagnostics(model, validation_loader, device)
        name = f"checkpoint-{epoch:04d}"
        cer = scores["strict"]["cer"] if scores else None
        previous_best_cer = best_cer
        is_best = not line_pilot and cer is not None and cer < best_cer
        qualifying_cer_improvement = is_best and cer < previous_best_cer - args.selection_min_delta
        is_best_loss = not line_pilot and validation_loss is not None and validation_loss < best_loss - args.selection_min_delta
        if qualifying_cer_improvement:
            epochs_without_cer_improvement = 0
        elif not line_pilot and cer is not None:
            epochs_without_cer_improvement += 1
        stopping = (not line_pilot and args.early_stopping_patience > 0 and cer is not None
                    and epochs_without_cer_improvement >= args.early_stopping_patience)
        if stopping:
            row["early_stop"] = {"reason": "validation_strict_cer_patience", "patience": args.early_stopping_patience,
                                  "epochs_without_improvement": epochs_without_cer_improvement,
                                  "selection_min_delta": args.selection_min_delta}
        selected_exact = None
        is_selection_candidate = False
        if checkpoint_selection_policy is not None and cer is not None:
            if is_best:
                best_cer = cer
            current_candidate = checkpoint_candidate(name, epoch, cer, scores["strict"]["exact_lines"])
            selection_candidates, selected_exact, is_selection_candidate = update_exact_line_frontier(
                selection_candidates, current_candidate, best_cer, args.exact_line_cer_band)
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs or is_best or is_best_loss or is_selection_candidate or stopping:
            save_checkpoint(args.output_dir / name, model, tokenizer, image_config, metadata={**run, "epoch": epoch, "metrics": row},
                            training_state={"epoch": epoch, "optimizer": optimizer.state_dict(), "rng": rng_state()})
            if predictions:
                (args.output_dir / f"validation-{epoch:04d}.jsonl").write_bytes(b"".join(json_bytes(p) for p in predictions))
            (args.output_dir / "last.json").write_bytes(json_bytes({"checkpoint": name, "epoch": epoch}))
        with (args.output_dir / "training_metrics.jsonl").open("ab") as handle:
            handle.write(json_bytes(row))
        if is_best:
            (args.output_dir / "best.json").write_bytes(json_bytes({"checkpoint": name, "epoch": epoch, "strict_cer": cer,
                "selection_metric": f"absolute_lowest_{run['epoch_metrics_partition']}_strict_cer"}))
        if selected_exact is not None:
            selection_report = {
                "checkpoint": selected_exact["checkpoint"], "epoch": selected_exact["epoch"],
                "strict_cer": selected_exact["strict_cer"], "exact_lines": selected_exact["exact_lines"],
                "selection_metric": "most_exact_validation_lines_within_cer_band",
                "absolute_best_strict_cer": best_cer, "cer_band": args.exact_line_cer_band,
                "cer_band_percentage_points": 100 * args.exact_line_cer_band,
                "tie_breakers": ["lower_strict_cer", "earlier_epoch"],
            }
            (args.output_dir / "best_exact.json").write_bytes(json_bytes(selection_report))
            (args.output_dir / "checkpoint_selection.json").write_bytes(json_bytes({
                "absolute_best_strict_cer": best_cer, "cer_band": args.exact_line_cer_band,
                "candidates": selection_candidates, "selected": selection_report,
            }))
        if is_best_loss:
            best_loss = validation_loss
            (args.output_dir / "best_loss.json").write_bytes(json_bytes({"checkpoint": name, "epoch": epoch, "validation_loss": validation_loss,
                "selection_metric": "fixed_validation_denoising_loss", "schedule_seed": args.validation_loss_seed}))
        print(format_epoch_progress(row) if args.epoch_log_format == "compact" else json.dumps(row, ensure_ascii=False), flush=True)
        if stopping:
            break


if __name__ == "__main__":
    main()
