"""Generate Arabic line predictions with the custom CTC line OCR model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "thesis" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_ctc_line_ocr_model import (  # noqa: E402
    CTCLineDataset,
    build_model,
    char_maps,
    collate_batch,
    ctc_greedy_decode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "thesis" / "models" / "ctc_line_ocr")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "thesis" / "labels" / "surya_line_ground_truth",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "thesis" / "predictions_ctc")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_records(dataset_root: Path):
    from train_ctc_line_ocr_model import read_line_records

    records = [
        record
        for record in read_line_records(dataset_root)
        if Path(record.image_path).is_file()
    ]
    return records


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(f"CTC inference dependencies are missing: {exc}") from exc

    config_path = args.model / "config.json"
    vocab_path = args.model / "vocab.json"
    weights_path = args.model / "model.pt"
    if not config_path.exists() or not vocab_path.exists() or not weights_path.exists():
        raise SystemExit(
            f"Missing CTC model files under {args.model}. "
            "Expected config.json, vocab.json, and model.pt."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    char_to_idx, idx_to_char = char_maps(vocab)
    records = load_records(args.dataset_root)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"No line images found under {args.dataset_root}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = build_model(
        len(vocab),
        int(config.get("hidden_size", 256)),
        int(config.get("num_lstm_layers", 2)),
        config.get("sequence_direction", "rtl"),
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = CTCLineDataset(
        records,
        char_to_idx,
        int(config.get("image_height", 48)),
        int(config.get("max_width", 1024)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    rows: list[dict[str, str | int]] = []
    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch["images"].to(device))
            pred_ids = log_probs.argmax(dim=-1).cpu().tolist()
            for record, ids in zip(batch["records"], pred_ids):
                rows.append(
                    {
                        "page_id": record.page_id,
                        "line_index": record.line_index,
                        "image_path": str(Path(record.image_path).relative_to(ROOT)),
                        "reference": record.text,
                        "prediction": ctc_greedy_decode(ids, idx_to_char),
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"samples": len(rows), "device": str(device), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
