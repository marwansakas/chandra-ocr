# OCR pipelines

The project has four primary stages plus dictionary-aware and CNN-fronted TrOCR
variants. Each owns one output directory and can run without changing another
engine's output.

Five additional pretrained baselines use the same cropped line dataset through
the `pretrained_ocr` package. Their separate environment setup and complete
run/scoring commands are in the [pretrained evaluation guide](../pretrained_ocr/README.md).

| Stage | Entry point | Default output |
| --- | --- | --- |
| Surya line detection | `python pipelines/surya.py` | `thesis/datasets/surya_lines` |
| Chandra line OCR | `python pipelines/chandra_run.py` | `thesis/predictions/chandra` |
| Invizo line OCR | `python pipelines/invizo_run.py` | `thesis/predictions/invizo` |
| Thesis TrOCR | `python pipelines/my_TrOCR_run.py` | `thesis/predictions/my_TrOCR` |
| General-Arabic TrOCR | `python pipelines/general_arabic_trocr_run.py` | `thesis/predictions/general_arabic_TrOCR` |
| CNN-TrOCR | `python pipelines/cnn_TrOCR_run.py` | `thesis/predictions/cnn_TrOCR` |
| PaddleOCR Arabic PP-OCRv5 | `python -m pretrained_ocr run --model paddle-arabic` | `thesis/predictions/pretrained/paddle-arabic` |
| GOT-OCR2.0 | `python -m pretrained_ocr run --model got-ocr2` | `thesis/predictions/pretrained/got-ocr2` |
| GLM-OCR | `python -m pretrained_ocr run --model glm-ocr` | `thesis/predictions/pretrained/glm-ocr` |
| HunyuanOCR | `python -m pretrained_ocr run --model hunyuan-ocr` | `thesis/predictions/pretrained/hunyuan-ocr` |
| LightOnOCR-2 | `python -m pretrained_ocr run --model lighton-ocr2` | `thesis/predictions/pretrained/lighton-ocr2` |

All paths may be overridden on the command line. Large source files, downloaded
repositories, model weights, and generated outputs stay outside Git.

Surya processes 100 PDFs by default. Use `--start-pdf 1` for PDFs 1-100,
`--start-pdf 101` for PDFs 101-200, and so on.

The OCR runners never use `ground_truth.txt` as input to recognition or modify
it. Each writes `prediction.txt`, `metadata.json`, a combined
`predictions.jsonl`, and `run_summary.json` under its own output directory. The
general-Arabic decoder additionally preserves `raw_prediction.txt` and all beam
scores in `candidates.json`. CNN-TrOCR uses a separate checkpoint trained by
`python thesis/scripts/train_cnn_trocr_model.py`.

To train multiple baseline TrOCR checkpoints with different image augmentation
profiles, run:

```bash
python thesis/scripts/train_trocr_augmentation_models.py \
  --profiles none,light,scan,geometry,stroke,mixed \
  --epochs 5 --batch-size 4 --views-per-line 3 --device cuda
```

