"""Run a separate PaddleOCR fine-tuning experiment from official source checkout."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def make_config(source: Path, target: Path, args, dataset: Path, splits: Path, output: Path) -> None:
    """Write a concrete config rather than addressing YAML list items via ``-o``.

    PaddleOCR's ``-o`` merger supports dictionary paths but not ``head_list.1``.
    The official Arabic config keeps the NRTR head in such a list, and its text
    length must match the global text length.  Resolving the config here avoids
    that parser limitation.
    """
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required; install PaddleOCR requirements first.") from error
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    global_config = config["Global"]
    global_config.update(
        use_gpu=args.device == "gpu",
        epoch_num=args.epochs,
        save_model_dir=str(output),
        pretrained_model=str(args.pretrained_model.resolve()),
    )
    config["max_text_length"] = args.max_text_length
    for head in config["Architecture"]["Head"]["head_list"]:
        if "NRTRHead" in head:
            head["NRTRHead"]["max_text_length"] = args.max_text_length
    for partition, manifest in (("Train", "train_list.txt"), ("Eval", "val_list.txt")):
        config[partition]["dataset"]["data_dir"] = str(dataset)
        config[partition]["dataset"]["label_file_list"] = [str(splits / manifest)]
        config[partition]["loader"]["batch_size_per_card"] = args.batch_size
    config["Optimizer"]["lr"]["learning_rate"] = args.learning_rate
    # Small handwriting datasets can have fewer than 1,000 train batches.
    config["Global"]["save_epoch_step"] = 1
    config["Global"]["eval_batch_step"] = [0, 100]
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


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
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_config = output.parent / f"{output.name}_training.yaml"
    make_config(config, generated_config, args, dataset, splits, output)
    command = [sys.executable, str(train_tool), "-c", str(generated_config)]
    print("Running:\n", " ".join(map(str, command)))
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
