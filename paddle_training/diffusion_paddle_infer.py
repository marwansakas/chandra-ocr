"""A distinct, conservative diffusion + PaddleOCR recognition pipeline.

It keeps the original crop as candidate zero.  A diffusion img2img model creates
additional restoration candidates, each is recognized by PaddleOCR, and a
diffusion result is accepted only when it produces exactly the same text as the
original with a configurable confidence gain.  This protects against invented
Arabic dots/letters; it is an experiment, not a general-purpose OCR fix.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROMPT = "clean high contrast scan of the same handwritten Arabic text, black ink on white paper, preserve every character, dot and stroke"
NEGATIVE = "changed text, extra text, missing text, printed font, watermark, signature, decoration"


def predict(recognizer, image):
    result = list(recognizer.predict(input=image, batch_size=1))
    payload = result[0].json if hasattr(result[0], "json") else result[0]
    if isinstance(payload, str):
        payload = json.loads(payload)
    payload = payload.get("res", payload)
    return str(payload["rec_text"]), float(payload["rec_score"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A directory of cropped line images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paddle-model", default="arabic_PP-OCRv5_mobile_rec", help="Paddle model name or exported model directory")
    parser.add_argument("--diffusion-model", required=True, help="Hugging Face image-to-image diffusion model ID or local path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--strength", type=float, default=.15)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--minimum-confidence-gain", type=float, default=.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 1 <= args.variants <= 8 or not 0 < args.strength <= .35:
        parser.error("variants must be 1..8 and strength must be in (0,.35]")
    if args.output.exists():
        parser.error("output must not already exist")
    from PIL import Image
    import torch
    from diffusers import AutoPipelineForImage2Image
    from paddleocr import TextRecognition
    output = args.output.resolve(); output.mkdir(parents=True)
    rec_device = "gpu:0" if args.device.startswith("cuda") else "cpu"
    model_path = Path(args.paddle_model)
    recognizer = (TextRecognition(model_dir=str(model_path), device=rec_device)
                  if model_path.is_dir() else TextRecognition(model_name=args.paddle_model, device=rec_device))
    pipe = AutoPipelineForImage2Image.from_pretrained(args.diffusion_model, torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32)
    pipe = pipe.to(args.device)
    rows = []
    images = sorted([*args.input.rglob("*.png"), *args.input.rglob("*.jpg"), *args.input.rglob("*.jpeg")])
    for index, path in enumerate(images):
        original = Image.open(path).convert("RGB")
        base_text, base_score = predict(recognizer, original)
        chosen_text, chosen_score, chosen = base_text, base_score, "original"
        candidates = []
        for variant in range(args.variants):
            generator = torch.Generator(device=args.device).manual_seed(args.seed + index * 100 + variant)
            restored = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, image=original, strength=args.strength,
                            guidance_scale=args.guidance_scale, generator=generator).images[0]
            text, confidence = predict(recognizer, restored)
            candidates.append({"variant": variant, "text": text, "confidence": confidence})
            # Do not use a visually altered image to change a transcription. Agreement is mandatory.
            if text == base_text and confidence >= base_score + args.minimum_confidence_gain and confidence > chosen_score:
                chosen_text, chosen_score, chosen = text, confidence, f"diffusion_{variant}"
                restored.save(output / f"{path.stem}_accepted_variant_{variant}.png")
        rows.append({"image": str(path), "prediction": chosen_text, "confidence": chosen_score,
                     "selected": chosen, "original_prediction": base_text, "original_confidence": base_score,
                     "candidates": json.dumps(candidates, ensure_ascii=False)})
        print(f"[{index + 1}/{len(images)}] {path.name}: {chosen}", flush=True)
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["image"])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
