"""
Extract simple handwriting/image-quality features.

This script scores pages when the image can be loaded by OpenCV. For PDFs, it
tries to use PyMuPDF if installed. In the current environment PyMuPDF may not be
available, so PDF pages can be reported as unprocessed until a renderer is added.

Useful dependency for PDF rendering:
  pip install pymupdf

Writes:
  thesis/reports/handwriting_features_pages.csv
  thesis/reports/handwriting_features_students.csv

Run:
  python thesis/scripts/score_handwriting.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import tempfile

import common


def load_image(path: Path):
    import cv2

    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except Exception:
            return None, "pdf_renderer_missing"
        doc = fitz.open(str(path))
        if not doc:
            return None, "pdf_empty"
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            pix.save(tmp.name)
            img = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE)
        return img, "ok" if img is not None else "load_failed"

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img, "ok" if img is not None else "load_failed"


def image_features(img):
    import cv2
    import numpy as np

    height, width = img.shape[:2]
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ink_pixels = int(np.count_nonzero(binary))
    total_pixels = int(binary.size)
    ink_density = ink_pixels / total_pixels if total_pixels else 0.0

    horizontal_projection = binary.sum(axis=1) / 255
    active_rows = np.where(horizontal_projection > max(5, width * 0.01))[0]
    line_count_est = 0
    if len(active_rows):
        groups = np.split(active_rows, np.where(np.diff(active_rows) > 3)[0] + 1)
        line_count_est = sum(1 for g in groups if len(g) >= 3)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [(x, y, w, h) for x, y, w, h in boxes if 3 <= w <= width * 0.4 and 3 <= h <= height * 0.25]

    component_count = len(boxes)
    heights = [h for _, _, _, h in boxes]
    widths = [w for _, _, w, _ in boxes]
    median_component_height = float(np.median(heights)) if heights else 0.0
    median_component_width = float(np.median(widths)) if widths else 0.0
    height_cv = float(np.std(heights) / np.mean(heights)) if heights and np.mean(heights) else 0.0

    # More ink is not always better. This is a rough process-quality proxy.
    density_score = max(0.0, min(1.0, 1.0 - abs(ink_density - 0.08) / 0.08))
    consistency_score = max(0.0, min(1.0, 1.0 - height_cv))
    legibility_proxy = (density_score + consistency_score) / 2

    return {
        "image_width": width,
        "image_height": height,
        "ink_density": f"{ink_density:.5f}",
        "estimated_line_count": line_count_est,
        "component_count": component_count,
        "median_component_height": f"{median_component_height:.2f}",
        "median_component_width": f"{median_component_width:.2f}",
        "component_height_cv": f"{height_cv:.4f}",
        "handwriting_legibility_proxy": f"{legibility_proxy:.4f}",
    }


def average_float(rows, field):
    values = [float(r[field]) for r in rows if r.get(field) not in ("", None)]
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    cfg = common.load_config()
    metadata_dir = common.resolve(cfg["metadata_dir"])
    reports_dir = common.resolve(cfg["reports_dir"])
    records = list(common.read_jsonl(metadata_dir / "pages.jsonl"))

    page_rows = []
    for record in records:
        image_path = common.resolve(record["image_path"])
        img, status = load_image(image_path)
        row = {
            "page_id": record["page_id"],
            "student_id": record["student_id"],
            "place": record["place"],
            "grade": record.get("grade") or "unknown",
            "gender": record.get("gender") or "unknown",
            "image_path": record["image_path"],
            "status": status,
        }
        if img is not None:
            row.update(image_features(img))
        page_rows.append(row)

    page_fields = [
        "page_id",
        "student_id",
        "place",
        "grade",
        "gender",
        "image_path",
        "status",
        "image_width",
        "image_height",
        "ink_density",
        "estimated_line_count",
        "component_count",
        "median_component_height",
        "median_component_width",
        "component_height_cv",
        "handwriting_legibility_proxy",
    ]
    common.write_csv(reports_dir / "handwriting_features_pages.csv", page_rows, fields=page_fields)

    by_student = defaultdict(list)
    for row in page_rows:
        if row["status"] == "ok":
            by_student[row["student_id"]].append(row)

    student_rows = []
    for student_id, rows in sorted(by_student.items()):
        first = rows[0]
        student_rows.append(
            {
                "student_id": student_id,
                "place": first["place"],
                "grade": first["grade"],
                "gender": first["gender"],
                "pages_scored": len(rows),
                "avg_ink_density": f"{average_float(rows, 'ink_density'):.5f}",
                "avg_estimated_line_count": f"{average_float(rows, 'estimated_line_count'):.2f}",
                "avg_component_height_cv": f"{average_float(rows, 'component_height_cv'):.4f}",
                "avg_handwriting_legibility_proxy": f"{average_float(rows, 'handwriting_legibility_proxy'):.4f}",
            }
        )
    common.write_csv(reports_dir / "handwriting_features_students.csv", student_rows)

    ok = sum(1 for r in page_rows if r["status"] == "ok")
    print(f"Processed image features for {ok}/{len(page_rows)} pages.")
    print(f"Wrote {reports_dir / 'handwriting_features_pages.csv'}")
    print(f"Wrote {reports_dir / 'handwriting_features_students.csv'}")
    if ok == 0:
        print("No PDF renderer was available. Install PyMuPDF to score PDF pages.")


if __name__ == "__main__":
    main()
