"""A distinct, conservative diffusion + PaddleOCR recognition pipeline.

It keeps the original crop as candidate zero. A diffusion img2img model creates
additional restoration candidates, each recognized by PaddleOCR. ``strict``
mode never alters text; ``consensus`` mode may use an alternative only when at
least two seeded variants agree and beat the original confidence. This is an
experiment, not a general-purpose OCR fix.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROMPT = "clean high contrast scan of the same handwritten Arabic text, black ink on white paper, preserve every character, dot and stroke"
NEGATIVE = "changed text, extra text, missing text, printed font, watermark, signature, decoration"


def predict(recognizer, image):
    # PaddleOCR v3 accepts paths or BGR NumPy arrays, but not Pillow images.
    # Diffusers produces Pillow RGB images, so convert without changing pixels.
    if not isinstance(image, (str, bytes)):
        import numpy as np
        image = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    result = list(recognizer.predict(input=image, batch_size=1))
    if not result:
        raise RuntimeError("PaddleOCR returned no result for an image candidate")
    payload = result[0].json if hasattr(result[0], "json") else result[0]
    if isinstance(payload, str):
        payload = json.loads(payload)
    payload = payload.get("res", payload)
    return str(payload["rec_text"]), float(payload["rec_score"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A directory of cropped line images")
    parser.add_argument("--manifest", type=Path,
                        help="Optional PaddleOCR list (for example test_list.txt); limits input to its first-column images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paddle-model", default="arabic_PP-OCRv5_mobile_rec", help="Paddle model name or exported model directory")
    parser.add_argument("--diffusion-model", required=True, help="Hugging Face image-to-image diffusion model ID or local path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--strength", type=float, default=.15)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--minimum-confidence-gain", type=float, default=.03)
    parser.add_argument("--selection", choices=("strict", "consensus"), default="strict",
                        help="strict keeps the original text; consensus may replace it when at least two diffusion variants agree on a higher-confidence alternative")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 1 <= args.variants <= 8 or not 0 < args.strength <= .35:
        parser.error("variants must be 1..8 and strength must be in (0,.35]")
    if args.selection == "consensus" and args.variants < 2:
        parser.error("--selection consensus requires at least --variants 2")
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
    if args.manifest:
        images = []
        for line in args.manifest.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            relative = line.split("\t", 1)[0]
            image = (args.input / relative).resolve()
            if not image.is_file() or args.input.resolve() not in image.parents:
                raise ValueError(f"Invalid manifest image path: {relative}")
            images.append(image)
    else:
        images = sorted([*args.input.rglob("*.png"), *args.input.rglob("*.jpg"), *args.input.rglob("*.jpeg")])
    for index, path in enumerate(images):
        original = Image.open(path).convert("RGB")
        base_text, base_score = predict(recognizer, original)
        chosen_text, chosen_score, chosen = base_text, base_score, "original"
        candidates = []
        candidate_images = {}
        for variant in range(args.variants):
            generator = torch.Generator(device=args.device).manual_seed(args.seed + index * 100 + variant)
            restored = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, image=original, strength=args.strength,
                            guidance_scale=args.guidance_scale, generator=generator).images[0]
            text, confidence = predict(recognizer, restored)
            candidates.append({"variant": variant, "text": text, "confidence": confidence})
            candidate_images[variant] = restored
            # Strict mode never changes the text and is useful as a visual-restoration
            # audit. Consensus mode below is the actual OCR-improvement experiment.
            if args.selection == "strict" and text == base_text and confidence >= base_score + args.minimum_confidence_gain and confidence > chosen_score:
                chosen_text, chosen_score, chosen = text, confidence, f"diffusion_{variant}"
        if args.selection == "consensus":
            groups = {}
            for candidate in candidates:
                groups.setdefault(candidate["text"], []).append(candidate)
            alternatives = []
            for text, group in groups.items():
                mean_confidence = sum(item["confidence"] for item in group) / len(group)
                if (text != base_text and len(group) >= 2 and
                        mean_confidence >= base_score + args.minimum_confidence_gain):
                    alternatives.append((mean_confidence, text, group))
            if alternatives:
                chosen_score, chosen_text, group = max(alternatives, key=lambda item: item[0])
                chosen = "diffusion_consensus_" + "_".join(str(item["variant"]) for item in group)
                candidate_images[group[0]["variant"]].save(output / f"{path.stem}_{chosen}.png")
        rows.append({"image": str(path), "prediction": chosen_text, "confidence": chosen_score,
                     "selected": chosen, "original_prediction": base_text, "original_confidence": base_score,
                     "candidates": json.dumps(candidates, ensure_ascii=False)})
        print(f"[{index + 1}/{len(images)}] {path.name}: {chosen}", flush=True)
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["image"])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
