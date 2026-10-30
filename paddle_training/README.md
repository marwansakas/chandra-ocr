# Separate PaddleOCR experiments

This directory intentionally creates two separate models/pipelines:

1. `paddle-finetuned-real`: fine-tunes Arabic PP-OCRv5 on real labelled handwriting.
2. `paddle-diffusion-conservative`: frozen or fine-tuned PaddleOCR plus a separate
   image-to-image diffusion restoration model at inference. It cannot alter a
   transcription unless the original and restored images produce the same text.

The second pipeline is deliberately conservative. Generic diffusion models do not
reliably preserve Arabic letter dots, so it may make no changes at all. It must be
evaluated against the untouched test images and is not used to manufacture labels.

## 1. Create safe manifests

Use the root containing sample folders with `image.png`, `ground_truth.txt`, and,
when available, `metadata.json`:

```bash
python paddle_training/prepare_dataset.py \
  --dataset /path/to/ground_truth_lines \
  --output thesis/datasets/paddle_split \
  --group-key student_id
```

`--group-key` is a dotted metadata path such as `writer.id`. It makes all samples
from the same child land in the same train/validation/test partition. Omit it only
when writer IDs truly are unavailable; the fallback is a line-level split.

By default, labels are read in logical Unicode order and converted once to the
visual order used for Arabic PP-OCRv5 training. Do not reverse the source labels.
Use `--label-order visual` only when your files were already explicitly stored
in visual/display order. Inspect `split.json` before training. Never use its test manifest in training or
checkpoint selection.

## 2. Fine-tune PaddleOCR (model one)

Clone the official PaddleOCR repository and install its GPU requirements in a fresh
environment. Download the **Arabic PP-OCRv5 training checkpoint** (not the
inference-only package) and give its checkpoint prefix to this command:

```bash
python paddle_training/train_paddle.py \
  --paddleocr-root /path/to/PaddleOCR \
  --dataset /path/to/ground_truth_lines \
  --splits thesis/datasets/paddle_split \
  --pretrained-model /path/to/arabic_PP-OCRv5_mobile_rec_pretrained \
  --output thesis/models/paddle_finetuned_real \
  --epochs 40 --batch-size 32 --learning-rate 0.0001 --max-text-length 64
```

This invokes PaddleOCR's official Arabic PP-OCRv5 configuration and only overrides
paths/training budget. Select the checkpoint with the best validation score, then
evaluate once on `test_list.txt` using the official `tools/eval.py` command.

## 3. Diffusion + PaddleOCR (model two)

Create a separate environment, since Diffusers/PyTorch need not share Paddle's
binary environment:

```bash
pip install -r paddle_training/requirements-diffusion.txt
python paddle_training/diffusion_paddle_infer.py \
  --input /path/to/ground_truth_lines \
  --manifest thesis/datasets/paddle_split/test_list.txt \
  --output thesis/predictions/paddle_diffusion_conservative \
  --paddle-model /path/to/exported_finetuned_paddle_model \
  --diffusion-model YOUR_IMAGE_TO_IMAGE_MODEL_ID
```

Start with `--strength 0.10` to `0.15`. Compare its predictions directly with the
normal Paddle model on exactly the same untouched test set. Retain it only if CER
improves; visual appeal is not an OCR metric.
