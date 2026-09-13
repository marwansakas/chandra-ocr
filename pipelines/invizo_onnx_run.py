"""Run Invizo's published ONNX recognizer on a held-out line manifest.

Uses upstream OCRModule/OCRConfig from Hedrax/Invizo-OCR commit
2b6f416b6218adc87de06c941382b2f708749090. No document detector is needed
because the input images are already line crops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from line_runner_common import ROOT, write_prediction, write_run_files


class CheckedSession:
    """Surface inference failures that upstream greedy_decode otherwise catches."""
    def __init__(self, session):
        self.session = session
        self.error = None

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, *args, **kwargs):
        try:
            return self.session.run(*args, **kwargs)
        except Exception as exc:
            self.error = exc
            raise


def checked_predict(recognizer, session, image):
    session.error = None
    prediction = recognizer.predict(image)
    if session.error is not None:
        raise RuntimeError("Invizo inference failed; partial prediction was discarded") from session.error
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    import cv2
    import onnxruntime as ort
    sys.path.insert(0, str(args.upstream_dir.resolve()))
    from OCRConfig import OCRConfig
    from OCRModule import OCRModule
    records = [json.loads(line) for line in args.test_manifest.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    if not records:
        raise ValueError("Test manifest is empty")
    seen = set()
    for row in records:
        sid = row["sample_id"]
        if not isinstance(sid, str) or sid in seen or sid in {".", ".."} or "/" in sid or "\\" in sid:
            raise ValueError(f"Invalid or duplicate sample ID: {sid!r}")
        seen.add(sid)
    session = CheckedSession(ort.InferenceSession(str(args.model.resolve()), providers=["CPUExecutionProvider"]))
    recognizer = OCRModule(OCRConfig(), session)
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    rows = []
    for index, record in enumerate(records, 1):
        destination = args.output_root / record["sample_id"]
        if (destination / "prediction.txt").exists() and not args.force:
            raise ValueError(f"Existing prediction: {destination}. Use a fresh output or --force.")
        path = ROOT / record["image_path"]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        prediction = checked_predict(recognizer, session, image)
        metadata = {"model": "invizo", "model_sha256": model_hash,
                    "sample_id": record["sample_id"], "image_path": str(path),
                    "scope": "line recognition only, upstream resize and greedy decode"}
        write_prediction(destination, prediction, metadata)
        rows.append(metadata)
        print(f"Invizo {index}/{len(records)}: {record['sample_id']}", flush=True)
    write_run_files(args.output_root, rows, {"model": "invizo", "completed": len(rows),
                    "test_manifest_sha256": hashlib.sha256(args.test_manifest.read_bytes()).hexdigest(),
                    "model_sha256": model_hash})


if __name__ == "__main__":
    main()
