"""Run a separate PaddleOCR fine-tuning experiment from official source checkout."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paddleocr-root", type=Path, required=True, help="Clone of github.com/PaddlePaddle/PaddleOCR")
    parser.add_argument("--splits", type=Path, required=True, help="Output from prepare_dataset.py")
    parser.add_argument("--dataset", type=Path, required=True, help="Original image.png/ground_truth.txt root")
    parser.add_argument("--pretrained-model", type=Path, required=True, help="Official Arabic PP-OCRv5 training checkpoint prefix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args(argv)
    root, splits, dataset, output = (args.paddleocr_root.resolve(), args.splits.resolve(),
                                      args.dataset.resolve(), args.output.resolve())
    config = root / "configs/rec/PP-OCRv5/multi_language/arabic_PP-OCRv5_mobile_rec.yaml"
    train_tool = root / "tools/train.py"
    for required in (config, train_tool, splits / "train_list.txt", splits / "val_list.txt"):
        if not required.is_file():
            parser.error(f"Missing required file: {required}")
    if output.exists():
        parser.error(f"output must not already exist: {output}")
    # Overrides keep the official architecture, dictionary, decoder and augmentations intact.
    overrides = [
        f"Global.use_gpu={str(args.device == 'gpu')}", f"Global.epoch_num={args.epochs}",
        f"Global.save_model_dir={output}", f"Global.pretrained_model={args.pretrained_model.resolve()}",
        f"Train.dataset.data_dir={dataset}", f"Train.dataset.label_file_list=[{splits / 'train_list.txt'}]",
        f"Eval.dataset.data_dir={dataset}", f"Eval.dataset.label_file_list=[{splits / 'val_list.txt'}]",
        f"Train.loader.batch_size_per_card={args.batch_size}", f"Eval.loader.batch_size_per_card={args.batch_size}",
        f"Optimizer.lr.learning_rate={args.learning_rate}", f"max_text_length={args.max_text_length}",
        f"Architecture.Head.head_list.1.NRTRHead.max_text_length={args.max_text_length}",
    ]
    command = [sys.executable, str(train_tool), "-c", str(config), "-o", *overrides]
    print("Running:\n", " ".join(map(str, command)))
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
