"""CPU integration smoke test with generated patterns; NOT a handwriting benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New artifact directory")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from PIL import Image, ImageDraw
    from thesis.diffusion.data import audit_dataset, freeze_splits, json_bytes
    root = args.output_dir / "data"
    writers = {}
    for index in range(12):
        sample_id = f"sample_{index:03d}"
        folder = root / sample_id
        folder.mkdir(parents=True)
        image = Image.new("RGB", (32 + index * 4, 16), "white")
        draw = ImageDraw.Draw(image)
        draw.line((2, 3 + index % 8, 20 + index, 12), fill=(index * 10, 0, 0), width=2)
        image.save(folder / "image.png")
        (folder / "ground_truth.txt").write_bytes((["مدرصة ١", "أ ب 12"][index % 2] + "\n").encode("utf-8"))
        writers[sample_id] = f"fixture_writer_{index // 2}"
    (args.output_dir / "writer_map.json").write_bytes(json_bytes(writers))
    freeze_splits(audit_dataset(root, writers), args.output_dir / "splits")
    commands = [
        ["thesis/scripts/train_diffusion_line_ocr_model.py", "--dataset-root", str(root), "--split-root", str(args.output_dir / "splits"),
         "--output-dir", str(args.output_dir / "run"), "--epochs", "1", "--batch-size", "2", "--height", "16", "--max-width", "128",
         "--patch-size", "4", "--d-model", "32", "--heads", "4", "--encoder-layers", "1", "--decoder-layers", "1", "--sampling-steps", "4", "--device", "cpu"],
        ["pipelines/diffusion_line_run.py", "--model", str(args.output_dir / "run"), "--input-root", str(root), "--split-root", str(args.output_dir / "splits"),
         "--output-root", str(args.output_dir / "predictions"), "--steps", "4", "--device", "cpu", "--trace"],
        ["thesis/scripts/evaluate_diffusion_line_ocr.py", "--dataset-root", str(root), "--split-root", str(args.output_dir / "splits"),
         "--predictions", "smoke=" + str(args.output_dir / "predictions"), "--output-dir", str(args.output_dir / "evaluation")],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    report = {"passed": True, "purpose": "CPU code integration only; generated patterns are not labeled handwriting",
              "commands": commands, "artifacts": str(args.output_dir)}
    (args.output_dir / "smoke_result.json").write_bytes(json_bytes(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
