"""Generate Arabic line predictions with the trained thesis TrOCR model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "thesis" / "models" / "surya_chandra_line_ocr")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "thesis" / "labels" / "surya_line_ground_truth",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "thesis" / "predictions")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_samples(dataset_root: Path) -> list[dict]:
    samples: list[dict] = []
    for lines_path in sorted(dataset_root.glob("*/lines.jsonl")):
        for raw in lines_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            image = ROOT / row["line_image"]
            if image.is_file():
                samples.append(
                    {
                        "page_id": lines_path.parent.name,
                        "line_index": int(row["line_index"]),
                        "image": image,
                        "reference": str(row.get("text") or ""),
                    }
                )
    return samples


def main() -> None:
    args = parse_args()
    try:
        import torch
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise SystemExit(f"Thesis inference dependencies are missing: {exc}") from exc

    samples = load_samples(args.dataset_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No line images found under {args.dataset_root}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()
    rows: list[dict] = []
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        images = [Image.open(sample["image"]).convert("RGB") for sample in batch]
        pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated = model.generate(pixels, max_length=args.max_length)
        predictions = processor.batch_decode(generated, skip_special_tokens=True)
        for sample, prediction in zip(batch, predictions):
            rows.append(
                {
                    "page_id": sample["page_id"],
                    "line_index": sample["line_index"],
                    "image_path": str(sample["image"].relative_to(ROOT)),
                    "reference": sample["reference"],
                    "prediction": prediction.strip(),
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

