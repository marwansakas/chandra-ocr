"""
Train a custom CTC line OCR model on Surya line crops.

The dataset contract is the same line-level dataset used elsewhere:

  thesis/labels/surya_line_ground_truth/<page>/lines.jsonl
  thesis/labels/surya_line_ground_truth/<page>/line_images/line_###.png

Each usable record contributes one full line image and one full transcription.
The model is a small CRNN:

  grayscale line image -> CNN -> horizontal feature sequence -> BiLSTM -> CTC

Run:
  python thesis/scripts/train_ctc_line_ocr_model.py --prepare-only
  python thesis/scripts/train_ctc_line_ocr_model.py --epochs 20 --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import common


PROJECT_ROOT = common.PROJECT_ROOT
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "thesis" / "labels" / "surya_line_ground_truth"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "thesis" / "labels" / "surya_line_ground_truth_ctc"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "thesis" / "models" / "ctc_line_ocr"

BLANK_TOKEN = "<blank>"


@dataclass
class LineRecord:
    sample_id: str
    page_id: str
    rank: int
    line_index: int
    image_path: str
    text: str
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--include-empty-labels", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-lstm-layers", type=int, default=2)
    parser.add_argument("--sequence-direction", choices=("rtl", "ltr"), default="rtl")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--eval-limit", type=int)
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def iter_page_dirs(dataset_root: Path) -> Iterable[Path]:
    return sorted(
        (p for p in dataset_root.iterdir() if p.is_dir() and p.name[:4].isdigit()),
        key=lambda p: (int(p.name[:4]), p.name),
    )


def read_line_records(dataset_root: Path) -> list[LineRecord]:
    records: list[LineRecord] = []
    for page_dir in iter_page_dirs(dataset_root):
        lines_path = page_dir / "lines.jsonl"
        if not lines_path.exists():
            continue
        rank = int(page_dir.name[:4])
        for raw in lines_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            line_index = int(row["line_index"])
            image_path = project_path(row["line_image"])
            records.append(
                LineRecord(
                    sample_id=f"{page_dir.name}__line_{line_index:03d}",
                    page_id=page_dir.name,
                    rank=rank,
                    line_index=line_index,
                    image_path=str(image_path),
                    text=row.get("text") or "",
                    status=row.get("status") or "",
                )
            )
    return records


def keep_for_training(record: LineRecord, include_empty_labels: bool) -> bool:
    if record.status != "ocr_complete":
        return False
    if not Path(record.image_path).exists():
        return False
    return include_empty_labels or bool(record.text.strip())


def student_group(page_id: str) -> str:
    without_rank = re.sub(r"^\d{4}_\d{4}_", "", page_id)
    return re.sub(r"_Page_\d+$", "", without_rank, flags=re.IGNORECASE)


def split_records(
    records: list[LineRecord],
    train_ratio: float,
    seed: int,
) -> tuple[list[LineRecord], list[LineRecord]]:
    by_student: dict[str, list[LineRecord]] = {}
    for record in records:
        by_student.setdefault(student_group(record.page_id), []).append(record)
    students = sorted(by_student)
    random.Random(seed).shuffle(students)
    train_count = (
        max(1, min(len(students) - 1, round(len(students) * train_ratio)))
        if len(students) > 1
        else len(students)
    )
    train_students = set(students[:train_count])
    train = [r for r in records if student_group(r.page_id) in train_students]
    test = [r for r in records if student_group(r.page_id) not in train_students]
    return train, test


def build_vocab(records: list[LineRecord]) -> list[str]:
    chars = sorted({char for record in records for char in normalize_label(record.text)})
    return [BLANK_TOKEN, *chars]


def char_maps(vocab: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    return char_to_idx, idx_to_char


def encode_text(text: str, char_to_idx: dict[str, int]) -> list[int]:
    return [char_to_idx[char] for char in text if char in char_to_idx and char != BLANK_TOKEN]


def normalize_label(text: str) -> str:
    """Normalize equivalent Unicode forms while preserving spaces and punctuation."""
    return unicodedata.normalize("NFC", text)


def ctc_greedy_decode(indices: list[int], idx_to_char: dict[int, str], blank_id: int = 0) -> str:
    output: list[str] = []
    previous = blank_id
    for index in indices:
        if index != blank_id and index != previous:
            output.append(idx_to_char[index])
        previous = index
    return "".join(output)


def write_manifest(path: Path, records: list[LineRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "page_id", "rank", "line_index", "image_path", "text", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "page_id": record.page_id,
                    "rank": record.rank,
                    "line_index": record.line_index,
                    "image_path": str(Path(record.image_path).relative_to(PROJECT_ROOT)),
                    "text": record.text,
                    "status": record.status,
                }
            )


def write_jsonl(path: Path, records: list[LineRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "sample_id": record.sample_id,
                        "page_id": record.page_id,
                        "rank": record.rank,
                        "line_index": record.line_index,
                        "image_path": str(Path(record.image_path).relative_to(PROJECT_ROOT)),
                        "text": record.text,
                        "status": record.status,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_summary(
    path: Path,
    all_records: list[LineRecord],
    usable: list[LineRecord],
    train: list[LineRecord],
    test: list[LineRecord],
    vocab: list[str],
    too_short_for_ctc: int = 0,
) -> None:
    blank = sum(1 for r in all_records if not r.text.strip())
    missing = sum(1 for r in all_records if r.status != "ocr_complete")
    lines = [
        "# CTC Line OCR Split",
        "",
        f"- Source pages: {len({r.page_id for r in all_records})}",
        f"- Source lines: {len(all_records)}",
        f"- Usable labeled lines: {len(usable)}",
        f"- Blank OCR labels excluded by default: {blank}",
        f"- Non-complete OCR records: {missing}",
        f"- Labels too long for CTC time steps: {too_short_for_ctc}",
        f"- Train lines: {len(train)}",
        f"- Test lines: {len(test)}",
        f"- Vocabulary symbols including blank: {len(vocab)}",
        "",
        "The split is notebook/student-level, so pages from the same child do not appear in both train and test.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class CTCLineDataset:
    def __init__(
        self,
        records: list[LineRecord],
        char_to_idx: dict[str, int],
        image_height: int,
        max_width: int,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.char_to_idx = char_to_idx
        self.image_height = image_height
        self.max_width = max_width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image, ImageEnhance, ImageOps

        record = self.records[index]
        image = Image.open(record.image_path).convert("L")
        if self.augment:
            rng = random.Random(record.sample_id)
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
            image = ImageOps.expand(image, border=rng.randint(0, 3), fill=255)

        width, height = image.size
        scaled_width = max(1, round(width * (self.image_height / height)))
        scaled_width = min(scaled_width, self.max_width)
        image = image.resize((scaled_width, self.image_height))
        pixels = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        pixels = pixels.float().view(self.image_height, scaled_width) / 255.0
        pixels = (1.0 - pixels).unsqueeze(0)
        targets = torch.tensor(encode_text(normalize_label(record.text), self.char_to_idx), dtype=torch.long)
        return {
            "image": pixels,
            "width": scaled_width,
            "target": targets,
            "target_length": len(targets),
            "record": record,
        }


def collate_batch(batch):
    import torch

    max_width = max(item["width"] for item in batch)
    height = batch[0]["image"].shape[-2]
    images = torch.zeros(len(batch), 1, height, max_width, dtype=torch.float32)
    widths = []
    targets = []
    target_lengths = []
    for row, item in enumerate(batch):
        width = item["width"]
        images[row, :, :, :width] = item["image"]
        widths.append(width)
        targets.append(item["target"])
        target_lengths.append(item["target_length"])
    flat_targets = torch.cat(targets) if targets else torch.tensor([], dtype=torch.long)
    return {
        "images": images,
        "widths": torch.tensor(widths, dtype=torch.long),
        "targets": flat_targets,
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "records": [item["record"] for item in batch],
    }


class CTCLineOCR:
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        sequence_direction: str = "rtl",
    ) -> None:
        import torch.nn as nn

        self.nn = nn
        self.sequence_direction = sequence_direction
        self.model = nn.Module()
        self.model.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.model.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_lstm_layers > 1 else 0.0,
        )
        self.model.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def to_torch_module(self):
        import torch
        import torch.nn as nn

        outer = self

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cnn = outer.model.cnn
                self.rnn = outer.model.rnn
                self.classifier = outer.model.classifier
                self.sequence_direction = outer.sequence_direction

            def forward(self, images):
                features = self.cnn(images)
                features = features.mean(dim=2).permute(0, 2, 1)
                if self.sequence_direction == "rtl":
                    features = features.flip(dims=[1])
                sequence, _ = self.rnn(features)
                logits = self.classifier(sequence)
                return torch.log_softmax(logits, dim=-1)

        return Module()


def ctc_input_lengths(widths, time_steps: int):
    import torch

    lengths = torch.div(widths, 4, rounding_mode="floor")
    return torch.clamp(lengths, min=1, max=time_steps)


def build_model(
    vocab_size: int,
    hidden_size: int,
    num_lstm_layers: int,
    sequence_direction: str = "rtl",
):
    return CTCLineOCR(vocab_size, hidden_size, num_lstm_layers, sequence_direction).to_torch_module()


def ctc_time_steps_for_width(width: int) -> int:
    return max(1, width // 4)


def ctc_min_time_steps(target: list[int]) -> int:
    if not target:
        return 0
    repeated_neighbors = sum(1 for left, right in zip(target, target[1:]) if left == right)
    return len(target) + repeated_neighbors


def scaled_width_for_image(path: Path, image_height: int, max_width: int) -> int:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return min(max(1, round(width * (image_height / height))), max_width)


def target_fits_ctc(record: LineRecord, char_to_idx: dict[str, int], image_height: int, max_width: int) -> bool:
    target = encode_text(normalize_label(record.text), char_to_idx)
    width = scaled_width_for_image(Path(record.image_path), image_height, max_width)
    return ctc_min_time_steps(target) <= ctc_time_steps_for_width(width)


def evaluate(model, loader, idx_to_char: dict[int, str], device, eval_limit: int | None) -> dict:
    import torch

    model.eval()
    total_cer = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            log_probs = model(images)
            pred_ids = log_probs.argmax(dim=-1).cpu().tolist()
            for record, ids in zip(batch["records"], pred_ids):
                prediction = ctc_greedy_decode(ids, idx_to_char)
                total_cer += common.ratio_error(
                    common.normalize_text(record.text),
                    common.normalize_text(prediction),
                )
                count += 1
                if eval_limit is not None and count >= eval_limit:
                    return {"eval_samples": count, "eval_cer": total_cer / count}
    return {"eval_samples": count, "eval_cer": total_cer / count if count else math.nan}


def train_model(
    train: list[LineRecord],
    test: list[LineRecord],
    vocab: list[str],
    args: argparse.Namespace,
) -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(f"CTC training dependencies are missing: {exc}") from exc

    char_to_idx, idx_to_char = char_maps(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not available; training on CPU will be slow.")

    train_ds = CTCLineDataset(
        train,
        char_to_idx,
        args.image_height,
        args.max_width,
        augment=args.augment,
    )
    test_ds = CTCLineDataset(test, char_to_idx, args.image_height, args.max_width)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    model = build_model(
        len(vocab),
        args.hidden_size,
        args.num_lstm_layers,
        args.sequence_direction,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                images = batch["images"].to(device)
                targets = batch["targets"].to(device)
                target_lengths = batch["target_lengths"].to(device)
                log_probs = model(images)
                input_lengths = ctc_input_lengths(batch["widths"], log_probs.shape[1]).to(device)
                loss = criterion(
                    log_probs.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running_loss += loss.item()
                if step % 25 == 0 or step == len(train_loader):
                    print(
                        f"epoch={epoch} step={step}/{len(train_loader)} "
                        f"loss={running_loss / step:.4f}"
                    )

            eval_result = evaluate(model, test_loader, idx_to_char, device, args.eval_limit)
            row = {
                "epoch": epoch,
                "train_loss": running_loss / max(1, len(train_loader)),
                **eval_result,
            }
            metrics_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_handle.flush()
            print(json.dumps(row, ensure_ascii=False, indent=2))
            save_checkpoint(args.output_dir / f"checkpoint-epoch-{epoch}", model, vocab, args)

    save_checkpoint(args.output_dir, model, vocab, args)
    print(f"Saved CTC model to {args.output_dir}")


def save_checkpoint(path: Path, model, vocab: list[str], args: argparse.Namespace) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {
        "architecture": "crnn_ctc",
        "image_height": args.image_height,
        "max_width": args.max_width,
        "hidden_size": args.hidden_size,
        "num_lstm_layers": args.num_lstm_layers,
        "sequence_direction": args.sequence_direction,
        "blank_token": BLANK_TOKEN,
        "blank_id": 0,
    }
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.dataset_root = project_path(args.dataset_root)
    args.split_root = project_path(args.split_root)
    args.output_dir = project_path(args.output_dir)
    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be between 0 and 1")

    records = read_line_records(args.dataset_root)
    usable = [r for r in records if keep_for_training(r, args.include_empty_labels)]
    usable.sort(key=lambda r: (r.rank, r.line_index, r.sample_id))
    if args.max_samples is not None:
        usable = usable[: args.max_samples]
    if len(usable) < 2:
        raise SystemExit("Not enough labeled line records to split into train/test.")

    train, test = split_records(usable, args.train_ratio, args.seed)
    vocab = build_vocab(usable)
    char_to_idx, _ = char_maps(vocab)
    before_ctc_filter = len(train) + len(test)
    train = [
        record
        for record in train
        if target_fits_ctc(record, char_to_idx, args.image_height, args.max_width)
    ]
    test = [
        record
        for record in test
        if target_fits_ctc(record, char_to_idx, args.image_height, args.max_width)
    ]
    too_short_for_ctc = before_ctc_filter - len(train) - len(test)
    if len(train) < 1 or len(test) < 1:
        raise SystemExit("Not enough CTC-compatible labeled lines after filtering.")
    args.split_root.mkdir(parents=True, exist_ok=True)
    write_manifest(args.split_root / "train.csv", train)
    write_manifest(args.split_root / "test.csv", test)
    write_jsonl(args.split_root / "train.jsonl", train)
    write_jsonl(args.split_root / "test.jsonl", test)
    (args.split_root / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(args.split_root / "summary.md", records, usable, train, test, vocab, too_short_for_ctc)

    print(f"Total line records: {len(records)}")
    print(f"Usable labeled line records: {len(usable)}")
    print(f"Train: {len(train)} lines")
    print(f"Test: {len(test)} lines")
    print(f"Labels too long for CTC time steps: {too_short_for_ctc}")
    print(f"Vocabulary symbols including blank: {len(vocab)}")
    print(f"Wrote CTC split files to: {args.split_root}")

    if args.prepare_only:
        return

    train_model(train, test, vocab, args)


if __name__ == "__main__":
    main()
