# OCR pipelines

The project has four separate stages. Each stage owns one output directory and
can be run without changing the output of another engine.

| Stage | Entry point | Default output |
| --- | --- | --- |
| Surya line detection | `python pipelines/surya.py` | `thesis/labels/surya_line_ground_truth` |
| Chandra line OCR | `python pipelines/chandra.py` | updates each page's `lines.jsonl` and `chandra_line_ocr/` |
| Invizo paragraph OCR | `python pipelines/invizo.py` | `thesis/invizo/outputs` |
| Custom CTC line OCR inference | `python pipelines/ctc.py` | `thesis/predictions_ctc` |
| Legacy TrOCR inference | `python pipelines/thesis.py` | `thesis/predictions` |

All paths may be overridden on the command line. Large source files, downloaded
repositories, model weights, and generated outputs stay outside Git.

