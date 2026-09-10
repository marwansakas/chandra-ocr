"""Generate labeled Arabic OCR training lines without changing real datasets.

render: font-rendered Arabic (not a learned simulation of children's writing).
augment: mild variations of handwriting selected from a training manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FONTS = PROJECT_ROOT / "thesis/assets/synthetic_fonts"
VERSION = 1


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    # Labels belong to single-line crops, but transcription files may contain
    # tabs or wrapped text. Normalize whitespace without changing the letters.
    text = unicodedata.normalize("NFC", " ".join(text.split()))
    if not text or text.casefold() == "ignore":
        raise ValueError("Labels must contain nonempty, verified text.")
    if any(unicodedata.category(c) == "Cc" for c in text):
        codes = sorted({f"U+{ord(c):04X}" for c in text if unicodedata.category(c) == "Cc"})
        raise ValueError(f"Label contains unsupported control characters: {', '.join(codes)}")
    if any(c in text for c in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"):
        raise ValueError("Use logical-order Unicode text without explicit bidi controls.")
    return text


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_training_sources(manifest: Path) -> list[dict]:
    rows = []
    seen = set()
    for raw in manifest.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or sample_id in seen:
            raise ValueError("Training manifest sample IDs must be unique strings.")
        seen.add(sample_id)
        if row.get("split", "train") != "train" or row.get("synthetic", False):
            raise ValueError("Use a manifest containing only original training samples.")
        try:
            text = clean_text(row["text"])
        except ValueError as exc:
            raise ValueError(f"Training sample {sample_id!r}: {exc}") from exc
        rows.append({
            "source_sample_id": sample_id,
            "source_page_id": row.get("page_id", ""),
            "text": text,
            "image_path": resolve_project_path(row["image_path"]),
        })
    if not rows:
        raise ValueError("The training manifest is empty.")
    return rows


def demo_sentence(rng: random.Random) -> str:
    """Small demonstration corpus; a user's training corpus is preferable."""
    boy = rng.choice(["أحمد", "عمر", "سامي", "خالد", "يوسف", "كريم", "علي", "حسن"])
    girl = rng.choice(["مريم", "سارة", "نور", "ليلى", "هند", "ريم", "سلمى", "فاطمة"])
    place = rng.choice(["المدرسة", "الحديقة", "المكتبة", "البيت", "القرية"])
    templates = [
        lambda: f"ذهب {boy} إلى {place} مع أخيه",
        lambda: f"ذهبت {girl} إلى {place} مع أختها",
        lambda: f"قرأ {boy} قصة عن {rng.choice(['القمر', 'البحر', 'الأسد', 'العصفور', 'الشجرة'])}",
        lambda: f"رسمت {girl} {rng.choice(['زهرة جميلة', 'شجرة كبيرة', 'فراشة صغيرة', 'سمكة زرقاء'])}",
        lambda: f"هذا {rng.choice(['كتاب', 'قلم', 'دفتر', 'باب', 'بيت'])} {rng.choice(['جديد', 'جميل', 'صغير', 'كبير'])}",
        lambda: f"هذه {rng.choice(['حقيبة', 'مدرسة', 'حديقة', 'غرفة', 'قصة'])} {rng.choice(['جديدة', 'جميلة', 'صغيرة', 'كبيرة'])}",
        lambda: f"أنا أحب {rng.choice(['القراءة', 'الرسم', 'الكتابة', 'الرياضة', 'العلوم'])} وأتعلم كل يوم",
        lambda: f"كتب {boy} {rng.choice(['كلمة', 'جملة', 'قصة قصيرة'])} في الدفتر",
        lambda: f"تجلس {girl} قرب {rng.choice(['النافذة', 'الشجرة', 'الطاولة', 'السبورة'])}",
        lambda: f"يلعب {boy} مع {rng.choice(['أصدقائه', 'إخوته', 'جيرانه'])} في الحديقة",
    ]
    return rng.choice(templates)()


class ArabicRenderer:
    def __init__(self, font_dir: Path, engine: str = "auto"):
        from PIL import features
        from fontTools.ttLib import TTFont

        self.raqm = features.check_feature("raqm") if engine == "auto" else engine == "raqm"
        if self.raqm and not features.check_feature("raqm"):
            raise ValueError("This Pillow build lacks RAQM. Use --shaping fallback.")
        self.reshaper = None
        if not self.raqm:
            import arabic_reshaper
            self.reshaper = arabic_reshaper.ArabicReshaper(configuration={
                "delete_harakat": False, "shift_harakat_position": True,
            })
        self.fonts = []
        for path in sorted(font_dir.rglob("*")):
            if path.suffix.lower() not in {".ttf", ".otf"}:
                continue
            with TTFont(path) as font:
                coverage = set((font.getBestCmap() or {}).keys())
            self.fonts.append((path, coverage, hashlib.sha256(path.read_bytes()).hexdigest()))
        if not self.fonts:
            raise ValueError(f"No .ttf/.otf fonts found in {font_dir}")

    @property
    def engine(self) -> str:
        return "raqm" if self.raqm else "reshaper-bidi"

    def display_text(self, text: str) -> str:
        if self.raqm:
            return text
        from bidi.algorithm import get_display
        return get_display(self.reshaper.reshape(text), base_dir="R")

    def render(self, text: str, rng: random.Random, max_width: int = 1800):
        from PIL import Image, ImageDraw, ImageFont

        display = self.display_text(text)
        needed = {ord(c) for c in display if not c.isspace() and c not in "\u200c\u200d"}
        compatible = [(p, digest) for p, cmap, digest in self.fonts if needed <= cmap]
        if not compatible:
            raise ValueError(f"No font covers every rendered character in: {text!r}")
        path, digest = rng.choice(compatible)
        size = rng.randint(40, 64)
        layout = ImageFont.Layout.RAQM if self.raqm else ImageFont.Layout.BASIC
        kwargs = {"direction": "rtl", "language": "ar"} if self.raqm else {}
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        font = ImageFont.truetype(str(path), size, layout_engine=layout)
        box = probe.textbbox((0, 0), display, font=font, **kwargs)
        if box[2] - box[0] > max_width:
            raise ValueError("A text line is too wide. Split it into shorter complete lines.")
        margin = max(20, size // 2)
        background = rng.randint(244, 255)
        image = Image.new("RGB", (box[2] - box[0] + 2 * margin,
                                  box[3] - box[1] + 2 * margin), (background,) * 3)
        ink = rng.choice([(25, 25, 25), (34, 45, 78), (45, 45, 45)])
        ImageDraw.Draw(image).text((margin - box[0], margin - box[1]), display,
                                   font=font, fill=ink, **kwargs)
        return image, {"font": path.name, "font_sha256": digest,
                       "font_size": size, "shaping": self.engine}


def vary_image(image, rng: random.Random, seed: int):
    """Pad before rotating; no cropping, erosion, flipping, or strong blur."""
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    image = ImageOps.expand(image.convert("RGB"), border=12, fill="white")
    angle = rng.uniform(-1.5, 1.5)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    scale = rng.uniform(0.97, 1.03)
    image = image.resize((max(1, round(image.width * scale)), image.height), Image.Resampling.BICUBIC)
    contrast = rng.uniform(0.94, 1.06)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    blur = rng.uniform(0.0, 0.28)
    image = image.filter(ImageFilter.GaussianBlur(blur))
    sigma = rng.uniform(0.0, 1.2)
    pixels = np.asarray(image).astype(np.float32)
    pixels += np.random.default_rng(seed).normal(0, sigma, pixels.shape)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype("uint8"))
    return image, {"rotation_degrees": angle, "width_scale": scale,
                   "contrast": contrast, "blur_radius": blur, "noise_sigma": sigma}


def save_json(path: Path, value: dict):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_preview(root: Path, sample_ids: list[str]):
    from PIL import Image, ImageDraw, ImageOps

    tile_w, tile_h = 900, 170
    sheet = Image.new("RGB", (tile_w, tile_h * len(sample_ids)), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for index, sample_id in enumerate(sample_ids):
        with Image.open(root / sample_id / "image.png") as image:
            thumb = ImageOps.contain(image, (tile_w - 30, tile_h - 30))
            sheet.paste(thumb, ((tile_w - thumb.width) // 2, index * tile_h + 22))
        draw.text((12, index * tile_h + 5), sample_id, fill="black")
    sheet.save(root / "preview.jpg", quality=94)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("render", "augment"), default="render")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--train-manifest", type=Path, help="Real training rows only; never test.jsonl.")
    source.add_argument("--text-file", type=Path, help="UTF-8 text, one complete line per sample.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "thesis/datasets/synthetic_train_lines")
    parser.add_argument("--fonts-dir", type=Path, default=DEFAULT_FONTS)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-width", type=int, default=1800)
    parser.add_argument("--shaping", choices=("auto", "raqm", "fallback"), default="auto")
    return parser.parse_args(argv)


def generate(args) -> dict:
    from PIL import Image

    if args.count < 1 or args.max_width < 64:
        raise ValueError("--count must be positive and --max-width at least 64.")
    if args.mode == "augment" and not args.train_manifest:
        raise ValueError("--mode augment requires --train-manifest.")
    root = args.output_root.resolve()
    real_root = (PROJECT_ROOT / "thesis/datasets/ground_truth_lines").resolve()
    if root == real_root or root in real_root.parents or real_root in root.parents:
        raise ValueError("Synthetic output must be separate from ground_truth_lines.")
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Output is not empty: {root}. Choose a new output directory.")
    sources = []
    source_description = "built-in demonstration sentences"
    manifest_hash = None
    if args.train_manifest:
        manifest = args.train_manifest.resolve()
        if manifest.stem.lower() == "val" or any(part in manifest.stem.lower() for part in ("test", "valid", "val.")):
            raise ValueError("Use train.jsonl, not a test/validation manifest.")
        sources = read_training_sources(manifest)
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        source_description = str(manifest)
        for source in sources:
            parent = source["image_path"].parent
            if root == parent or root in parent.parents or parent in root.parents:
                raise ValueError("Synthetic output overlaps a source sample directory.")
            if args.mode == "augment" and not source["image_path"].is_file():
                raise ValueError(f"Missing source image: {source['image_path']}")
    elif args.text_file:
        sources = [{"text": clean_text(line)} for line in args.text_file.read_text(encoding="utf-8-sig").splitlines()
                   if line.strip()]
        if not sources:
            raise ValueError("The text file is empty.")
        source_description = str(args.text_file.resolve())
    renderer = ArabicRenderer(args.fonts_dir, args.shaping) if args.mode == "render" else None
    # Preflight all supplied text before writing any output.
    if renderer:
        for source in sources:
            renderer.render(source["text"], random.Random(args.seed), args.max_width)
    config = {"schema_version": VERSION, "complete": False, "synthetic": True, "split": "train",
              "mode": args.mode, "count": args.count, "seed": args.seed,
              "source": source_description, "source_manifest_sha256": manifest_hash,
              "shaping": renderer.engine if renderer else None,
              "note": "Font rendering is synthetic typography, not verified children's handwriting."}
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "run.json", config)
    order = list(range(len(sources)))
    random.Random(args.seed).shuffle(order)
    preview_ids = []
    texts = set()
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as output:
        for index in range(args.count):
            sample_seed = int(text_hash(f"{args.seed}:{index}")[:16], 16)
            rng = random.Random(sample_seed)
            source = sources[order[index % len(order)]] if sources else {"text": demo_sentence(rng)}
            text = source["text"]
            if args.mode == "render":
                image, detail = renderer.render(text, rng, args.max_width)
            else:
                with Image.open(source["image_path"]) as original:
                    image = original.convert("RGB")
                detail = {"source_image_sha256": hashlib.sha256(source["image_path"].read_bytes()).hexdigest()}
            image, transforms = vary_image(image, rng, sample_seed)
            sample_id = f"synthetic_{args.mode}_{index + 1:06d}"
            directory = root / sample_id
            directory.mkdir()
            image.save(directory / "image.png")
            (directory / "ground_truth.txt").write_text(text + "\n", encoding="utf-8")
            metadata = {"sample_id": sample_id, "synthetic": True, "split": "train",
                        "generation_method": args.mode, "seed": sample_seed,
                        "source_sample_id": source.get("source_sample_id"),
                        "source_page_id": source.get("source_page_id"),
                        "source_text_sha256": text_hash(text) if source.get("source_sample_id") else None,
                        "text_sha256": text_hash(text), "transforms": transforms, **detail}
            save_json(directory / "metadata.json", metadata)
            output.write(json.dumps({**metadata, "text": text,
                                     "image_path": f"{sample_id}/image.png"}, ensure_ascii=False) + "\n")
            if len(preview_ids) < 12:
                preview_ids.append(sample_id)
            texts.add(text)
            if (index + 1) % 100 == 0 or index + 1 == args.count:
                print(f"Generated {index + 1}/{args.count}", flush=True)
    write_preview(root, preview_ids)
    config.update(complete=True, unique_transcriptions=len(texts))
    save_json(root / "run.json", config)
    print(f"Saved {args.count} training samples to {root}\nPreview: {root / 'preview.jpg'}")
    return config


def main():
    try:
        generate(parse_args())
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
