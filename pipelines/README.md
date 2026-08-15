# OCR pipelines

The project has four separate stages. Each stage owns one output directory and
can be run without changing the output of another engine.

| Stage | Entry point | Default output |
| --- | --- | --- |
| Surya line detection | `python pipelines/surya.py` | `thesis/datasets/surya_lines` |
| Chandra line OCR | `python pipelines/chandra_run.py` | `thesis/predictions/chandra` |
| Invizo line OCR | `python pipelines/invizo_run.py` | `thesis/predictions/invizo` |
| Thesis TrOCR | `python pipelines/my_TrOCR_run.py` | `thesis/predictions/my_TrOCR` |

All paths may be overridden on the command line. Large source files, downloaded
repositories, model weights, and generated outputs stay outside Git.

Surya processes 100 PDFs by default. Use `--start-pdf 1` for PDFs 1-100,
`--start-pdf 101` for PDFs 101-200, and so on.

The three OCR runners read only `image.png`; they never read or modify
`ground_truth.txt`. Each writes `prediction.txt`, `metadata.json`, a combined
`predictions.jsonl`, and `run_summary.json` under its own output directory.

