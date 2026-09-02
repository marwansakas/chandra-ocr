# OCR pipelines

The project has four primary stages plus dictionary-aware and CNN-fronted TrOCR
variants. Each owns one output directory and can run without changing another
engine's output.

| Stage | Entry point | Default output |
| --- | --- | --- |
| Surya line detection | `python pipelines/surya.py` | `thesis/datasets/surya_lines` |
| Chandra line OCR | `python pipelines/chandra_run.py` | `thesis/predictions/chandra` |
| Invizo line OCR | `python pipelines/invizo_run.py` | `thesis/predictions/invizo` |
| Thesis TrOCR | `python pipelines/my_TrOCR_run.py` | `thesis/predictions/my_TrOCR` |
| General-Arabic TrOCR | `python pipelines/general_arabic_trocr_run.py` | `thesis/predictions/general_arabic_TrOCR` |
| CNN-TrOCR | `python pipelines/cnn_TrOCR_run.py` | `thesis/predictions/cnn_TrOCR` |

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

