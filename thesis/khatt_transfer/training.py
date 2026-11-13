"""Two-stage training around the existing SVTRv2/MDiff recognizer."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

from thesis.diffusion.data import file_hash
from thesis.diffusion.images import ImageConfig, prepare_images, resized_width
from thesis.diffusion.metrics import aggregate
from thesis.diffusion.tokenizer import CharacterTokenizer
from thesis.mdiff_arabic_v1.model import Recognizer, load_encoder, tensor_hash
from .data import load_prepared, read_json, write_json

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, value)
    os.replace(temp, path)


def save_checkpoint(path, value):
    value = dict(value, model_sha256=tensor_hash(value["model"]))
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def load_checkpoint(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("version") != 1 or value.get("arm") != "mdiff":
        raise ValueError("Expected a khatt_transfer v1 MDiff checkpoint")
    if tensor_hash(value["model"]) != value["model_sha256"]:
        raise ValueError("Checkpoint model hash mismatch")
    if CharacterTokenizer(value["tokenizer"]["characters"]).as_dict() != value["tokenizer"]:
        raise ValueError("Checkpoint tokenizer policy mismatch")
    return value


def check_records(rows, tokenizer, config, *, labels=True):
    image_config = ImageConfig(height=32, max_width=config["max_width"], patch_size=4)
    for row in rows:
        try:
            if labels:
                tokenizer.encode(row["text"], config["capacity"], eos_fill=False)
            resized_width(row["width"], row["height"], image_config)
        except ValueError as error:
            raise ValueError(f"{row['sample_id']}: {error}") from error


def batch(root, rows, tokenizer, config, device, *, labels):
    images = []
    for row in rows:
        path = root / row["image_path"]
        if file_hash(path) != row["image_sha256"]:
            raise ValueError("Image changed during the run: " + row["sample_id"])
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    pixels, widths = prepare_images(images, ImageConfig(height=32, max_width=config["max_width"], patch_size=4))
    targets = torch.tensor([tokenizer.encode(row["text"], config["capacity"], eos_fill=False) for row in rows],
                           dtype=torch.long, device=device) if labels else None
    return pixels.to(device), widths.to(device), targets


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()


def select_device(name):
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; run inside a GPU allocation or explicitly use --device cpu")
    return device


@torch.no_grad()
def evaluate_rows(model, root, rows, tokenizer, config, device, steps):
    model.eval()
    predictions, pairs = [], []
    for row in rows:
        # Image-only batch construction: reference tokens/length never reach generation.
        pixels, widths, _ = batch(root, [row], tokenizer, config, device, labels=False)
        with amp(device):
            ids, details = model.generate(pixels, widths, steps=steps)
        text, has_eos = tokenizer.decode(ids[0].cpu().tolist())
        predictions.append(dict(sample_id=row["sample_id"], prediction=text, has_eos=has_eos,
                                decoder_passes=details["decoder_passes"], reference=row["text"],
                                reference_oov=sorted(set(row["text"]) - set(tokenizer.characters)),
                                reference_over_capacity=len(row["text"])+1 > config["capacity"]))
        pairs.append((row["text"], text))
    scores = aggregate(pairs, no_harakat_v1=True)
    scores.update(expected=len(rows), completed=len(predictions),
                  no_eos=sum(not p["has_eos"] for p in predictions),
                  reference_oov_lines=sum(bool(p["reference_oov"]) for p in predictions),
                  reference_over_capacity_lines=sum(p["reference_over_capacity"] for p in predictions))
    return scores, predictions


def metric_key(scores):
    return (scores["char_distance"], scores["word_distance"], -scores["exact_lines"])


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_all(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def source_hashes():
    paths = set((ROOT / "thesis/khatt_transfer").glob("*.py"))
    paths.update((ROOT / "thesis/mdiff_arabic_v1/vendor").glob("*.py"))
    paths.add(ROOT / "thesis/mdiff_arabic_v1/model.py")
    paths.update((ROOT / "thesis/diffusion").glob("*.py"))
    paths.add(ROOT / "thesis/scripts/common.py")
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def environment(device):
    return dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__),
                numpy=np.__version__, pillow=importlib.metadata.version("Pillow"),
                device=str(device), cuda=torch.version.cuda,
                precision="bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32",
                gpu=torch.cuda.get_device_name() if device.type == "cuda" else None)


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [field for key in sorted(set(left) | set(right))
                for field in differences(left.get(key), right.get(key), prefix + str(key) + ".")]
    return [] if left == right else [prefix.rstrip(".")]


@contextmanager
def run_lock(folder):
    # Lock is held by the OS; a killed process does not leave a stale exclusive claim.
    folder.parent.mkdir(parents=True, exist_ok=True)
    with (folder.parent / (folder.name + ".lock")).open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def build_config(args, tokenizer, parts, parent):
    if parent:
        config = parent["config"]
        for key in ("capacity", "dim", "layers", "dropout", "max_width"):
            value = getattr(args, key)
            if value is not None and value != config[key]:
                raise ValueError("Checkpoint transfer/resume cannot change " + key)
        if tokenizer.as_dict() != parent["tokenizer"]:
            raise ValueError("Checkpoint tokenizer differs: retain the KHATT tokenizer and character IDs")
        return config
    return dict(capacity=args.capacity or max(144, max(len(r["text"])+1 for r in parts["train"])),
                dim=args.dim or 256, layers=args.layers or 3,
                dropout=.1 if args.dropout is None else args.dropout, max_width=args.max_width or 4096)


def make_model(config, tokenizer, device):
    return Recognizer(len(tokenizer), capacity=config["capacity"], arm="mdiff", dim=config["dim"],
                      layers=config["layers"], dropout=config["dropout"]).to(device)


def train(args):
    output = args.output.resolve()
    with run_lock(output):
        _train(args, output)


def _train(args, output):
    metadata, tokenizer, parts = load_prepared(args.prepared)
    root = args.prepared.resolve()
    data_hash = file_hash(root / "prepared.json")
    device = select_device(args.device)
    seed_all(args.seed, args.cpu_threads)
    if args.resume:
        state = load_checkpoint(output / "last.pt")
        run = read_json(output / "run.json")
        parent = state
    else:
        if output.exists():
            raise FileExistsError(f"Choose a new output directory, or use --resume: {output}")
        state = None
        parent = load_checkpoint(args.init_from) if args.init_from else None
        if metadata["stage"] == "children" and parent is None:
            raise ValueError("Children fine-tuning requires --init-from KHATT_RUN/best.pt")
        if metadata["stage"] == "khatt" and parent is not None:
            raise ValueError("KHATT pretraining starts with a fresh MDiff head; omit --init-from")
        if parent and metadata["source"]["init_checkpoint_sha256"] != file_hash(args.init_from):
            raise ValueError("Children snapshot was prepared for a different initialization checkpoint")
        if parent and any(r["pixels_sha256"] in set(parent["seen_pixels"]) for rows in parts.values() for r in rows):
            raise ValueError("Training/validation images overlap the parent stage")
        if not parent and not (args.encoder_checkpoint or args.random_init):
            raise ValueError("Supply --encoder-checkpoint from setup-assets, or explicitly use --random-init")
        if parent and (args.encoder_checkpoint or args.random_init):
            raise ValueError("Fine-tuning already loads all model weights; omit encoder/random initialization flags")
    config = build_config(args, tokenizer, parts, parent)
    for rows in parts.values():
        check_records(rows, tokenizer, config)
    fine = metadata["stage"] == "children"
    options = dict(epochs=args.epochs, patience=args.patience, seed=args.seed, cpu_threads=args.cpu_threads,
                   batch_size=args.batch_size, accumulation=args.accumulation, steps=args.steps,
                   decoder_lr=args.decoder_lr if args.decoder_lr is not None else (3e-5 if fine else 3e-4),
                   encoder_lr=args.encoder_lr if args.encoder_lr is not None else (1e-6 if fine else 1e-5),
                   freeze_epochs=args.freeze_epochs if args.freeze_epochs is not None else (0 if fine else 5),
                   warmup_epochs=args.warmup_epochs, weight_decay=args.weight_decay)
    binding = dict(prepared_sha256=data_hash, config=config, options=options, sources=source_hashes(),
                   environment=environment(device))
    if args.resume:
        changed = differences(binding, run["binding"]) + differences(binding, state["binding"])
        if changed:
            raise ValueError("Resume binding changed: " + ", ".join(sorted(set(changed))) + "; use original settings/environment")
        if args.init_from and file_hash(args.init_from) != run["parent_checkpoint_sha256"]:
            raise ValueError("Resume initialization differs from original stage")
    model = make_model(config, tokenizer, device)
    if parent:
        model.load_state_dict(parent["model"], strict=True)
    encoder_source = None
    if not parent and args.encoder_checkpoint:
        from thesis.mdiff_arabic_v1.artifacts import ASSETS
        if file_hash(args.encoder_checkpoint) != ASSETS["best.pth"]:
            raise ValueError("Expected the pinned official MDiff4STR-small visual checkpoint")
        encoder_source = load_encoder(model.encoder, args.encoder_checkpoint)
    groups = [dict(params=list(model.encoder.parameters()), lr=options["encoder_lr"]),
              dict(params=[p for name, p in model.named_parameters() if not name.startswith("encoder.")],
                   lr=options["decoder_lr"])]
    optimizer = torch.optim.AdamW(groups, weight_decay=options["weight_decay"])
    seen_pixels = sorted({r["pixels_sha256"] for rows in parts.values() for r in rows}
                         | (set(parent["seen_pixels"]) if parent else set()))
    lineage = (state.get("lineage", []) if args.resume else
               parent.get("lineage", []) + [dict(checkpoint_sha256=file_hash(args.init_from),
                    model_sha256=parent["model_sha256"], stage=parent["stage"],
                    prepared_sha256=parent["binding"]["prepared_sha256"])] if parent else [])
    base = dict(version=1, arm="mdiff", stage=metadata["stage"], config=config,
                tokenizer=tokenizer.as_dict(), binding=binding, seen_pixels=seen_pixels, lineage=lineage)
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        run = dict(binding=binding, stage=metadata["stage"],
                   parent_checkpoint_sha256=file_hash(args.init_from) if args.init_from else None,
                   parent_stage=parent["stage"] if parent else None,
                   optimizer_reset=True, encoder_source=encoder_source,
                   initialization="KHATT model weights" if parent else "official visual encoder + fresh MDiff head" if args.encoder_checkpoint else "random",
                   interpretation="new run, not a reproduction of Ammar's joint KHATT/children AR experiment")
        atomic_json(output / "run.json", run)
        state = dict(base, model=model.state_dict(), epoch=0, best=None, best_epoch=None,
                     stale=0, history=[], optimizer=optimizer.state_dict(), rng=rng_state())
        # Stage zero is evaluated on THIS stage's validation, never parent's scores.
        scores, predictions = evaluate_rows(model, root, parts["validation"], tokenizer, config, device, args.steps)
        state.update(best=scores, best_epoch=0, rng=rng_state())
        save_checkpoint(output / "best.pt", dict(base, model=model.state_dict(), epoch=0, metrics=scores))
        save_checkpoint(output / "last.pt", state)
        atomic_json(output / "validation-0000.json", dict(metrics=scores, predictions=predictions))
        atomic_json(output / "best.json", dict(epoch=0, metrics=scores))
    else:
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        # Reconstruct small derived summaries after a crash between atomic commits.
        atomic_json(output / "best.json", dict(epoch=state["best_epoch"], metrics=state["best"]))
        atomic_json(output / "history.json", state["history"])
    pending_path = output / "trained.pt"
    if (output / "complete.json").exists():
        if file_hash(output / "best.pt") != read_json(output / "complete.json")["best_checkpoint_sha256"]:
            raise ValueError("Completed stage's selected checkpoint hash changed")
        print("Stage already complete; no further optimizer updates.")
        return
    while state["epoch"] < args.epochs and state["stale"] < args.patience:
        epoch = state["epoch"] + 1
        if pending_path.exists():
            pending = load_checkpoint(pending_path)
            if pending["binding"] != binding:
                raise ValueError("Pending epoch binding mismatch")
            if pending["epoch"] == epoch:
                model.load_state_dict(pending["model"], strict=True)
                optimizer.load_state_dict(pending["optimizer"])
                restore_rng(pending["rng"])
                loss_value = pending["train_loss"]
            elif pending["epoch"] <= state["epoch"]:
                pending_path.unlink()
                continue
            else:
                raise ValueError("Pending epoch is not the next committed epoch")
        else:
            frozen = epoch <= options["freeze_epochs"]
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(not frozen)
            factor = min(1., epoch/max(1, args.warmup_epochs)) * (
                .1 + .9*.5*(1+math.cos(math.pi * max(0, epoch-args.warmup_epochs)/max(1, args.epochs-args.warmup_epochs))))
            for group, lr in zip(optimizer.param_groups, (options["encoder_lr"], options["decoder_lr"])):
                group["lr"] = lr * factor
            model.train()
            order = list(parts["train"])
            random.Random(args.seed + epoch).shuffle(order)
            loss_sum = 0.
            window_size = args.batch_size * args.accumulation
            for start in range(0, len(order), window_size):
                window = order[start:start + window_size]
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, len(window), args.batch_size):
                    rows = window[offset:offset + args.batch_size]
                    pixels, widths, targets = batch(root, rows, tokenizer, config, device, labels=True)
                    with amp(device):
                        loss, _ = model.loss(pixels, widths, targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    (loss * len(rows)/len(window)).backward()
                    loss_sum += float(loss.detach()) * len(rows)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            loss_value = loss_sum/len(order)
            # Commit completed training BEFORE validation; a failed validation need not repeat an epoch.
            save_checkpoint(pending_path, dict(base, model=model.state_dict(), epoch=epoch,
                            optimizer=optimizer.state_dict(), rng=rng_state(), train_loss=loss_value))
        scores, predictions = evaluate_rows(model, root, parts["validation"], tokenizer, config, device, args.steps)
        improved = metric_key(scores) < metric_key(state["best"])
        if improved:
            save_checkpoint(output / "best.pt", dict(base, model=model.state_dict(), epoch=epoch, metrics=scores))
        row = dict(epoch=epoch, train_loss=loss_value, validation=scores,
                   examples=len(parts["train"]), optimizer_updates=math.ceil(len(parts["train"])/(args.batch_size*args.accumulation)))
        state = dict(base, model=model.state_dict(), epoch=epoch,
                     optimizer=optimizer.state_dict(), rng=rng_state(),
                     best=scores if improved else state["best"], best_epoch=epoch if improved else state["best_epoch"],
                     stale=0 if improved else state["stale"]+1, history=state["history"]+[row])
        atomic_json(output / f"validation-{epoch:04d}.json", dict(metrics=scores, predictions=predictions))
        save_checkpoint(output / "last.pt", state)
        pending_path.unlink(missing_ok=True)
        atomic_json(output / "best.json", dict(epoch=state["best_epoch"], metrics=state["best"]))
        atomic_json(output / "history.json", state["history"])
        print(json.dumps(row), flush=True)
    atomic_json(output / "complete.json", dict(epoch=state["epoch"], best_epoch=state["best_epoch"],
                reason="patience" if state["stale"] >= args.patience else "maximum_epochs",
                best_checkpoint_sha256=file_hash(output / "best.pt")))


def evaluate(args):
    metadata, tokenizer, parts = load_prepared(args.prepared, (args.partition,))
    checkpoint = load_checkpoint(args.checkpoint)
    if file_hash(args.prepared / "prepared.json") != checkpoint["binding"]["prepared_sha256"]:
        raise ValueError("Evaluation must use the checkpoint's own stage dataset snapshot")
    if tokenizer.as_dict() != checkpoint["tokenizer"]:
        raise ValueError("Evaluation tokenizer differs")
    check_records(parts[args.partition], tokenizer, checkpoint["config"], labels=False)
    device = select_device(args.device)
    seed_all(args.seed, args.cpu_threads)
    model = make_model(checkpoint["config"], tokenizer, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    steps = checkpoint["binding"]["options"]["steps"]
    scores, predictions = evaluate_rows(model, args.prepared, parts[args.partition], tokenizer, checkpoint["config"], device, steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Evaluation report exists; choose another --output")
    write_json(args.output, dict(stage=metadata["stage"], partition=args.partition, checkpoint_sha256=file_hash(args.checkpoint),
               normalization="no_harakat_v1 only; literal whitespace and other code points preserved",
               metrics=scores, predictions=predictions))
    print(json.dumps(scores, indent=2))
