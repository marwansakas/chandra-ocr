import os
import csv
import cv2
import fitz  # PyMuPDF
import numpy as np
from shutil import copy2


INPUT_FOLDER = "/home/sakas/chandra-ocr/split_papers"
OUTPUT_FOLDER = "ranked_pdfs"
CSV_FILE = "pdf_quality_scores.csv"

PDF_EXTENSIONS = (".pdf",)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def find_pdfs(folder):
    pdf_paths = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(PDF_EXTENSIONS):
                pdf_paths.append(os.path.join(root, file))

    return pdf_paths


def score_page(gray):
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()

    brightness = gray.mean()
    brightness_score = max(0, 100 - abs(brightness - 180))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = np.mean(np.abs(gray.astype(float) - blur.astype(float)))
    noise_score = max(0, 100 - noise * 5)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    text_ratio = np.count_nonzero(thresh) / thresh.size

    if 0.01 <= text_ratio <= 0.35:
        text_score = text_ratio * 300
    else:
        text_score = 0

    final_score = (
        sharpness * 0.45 +
        contrast * 2.0 +
        brightness_score * 1.2 +
        noise_score * 1.5 +
        text_score * 2.0
    )

    return {
        "score": final_score,
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "noise": noise,
        "text_ratio": text_ratio
    }


def score_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Could not open PDF: {pdf_path}")
        print(e)
        return None

    page_results = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )

            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)

            if pix.n == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif pix.n == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            else:
                gray = img

            page_score = score_page(gray)
            page_results.append(page_score)

    except Exception as e:
        print(f"Error processing PDF: {pdf_path}")
        print(e)
        doc.close()
        return None

    doc.close()

    if not page_results:
        return None

    avg_score = sum(p["score"] for p in page_results) / len(page_results)
    avg_sharpness = sum(p["sharpness"] for p in page_results) / len(page_results)
    avg_contrast = sum(p["contrast"] for p in page_results) / len(page_results)
    avg_brightness = sum(p["brightness"] for p in page_results) / len(page_results)
    avg_noise = sum(p["noise"] for p in page_results) / len(page_results)
    avg_text_ratio = sum(p["text_ratio"] for p in page_results) / len(page_results)

    return {
        "original_path": pdf_path,
        "original_name": os.path.basename(pdf_path),
        "page_count": len(page_results),
        "score": round(avg_score, 2),
        "sharpness": round(avg_sharpness, 2),
        "contrast": round(avg_contrast, 2),
        "brightness": round(avg_brightness, 2),
        "noise": round(avg_noise, 2),
        "text_ratio": round(avg_text_ratio, 4)
    }


def make_ranked_name(rank, original_path, rank_digits):
    relative_path = os.path.relpath(original_path, INPUT_FOLDER)
    safe_relative_name = relative_path.replace(os.sep, "__")

    return f"{rank:0{rank_digits}d}_{safe_relative_name}"


def main():
    pdf_paths = find_pdfs(INPUT_FOLDER)

    print(f"Found {len(pdf_paths)} PDF files.")

    results = []

    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"Processing {i}/{len(pdf_paths)}: {pdf_path}")

        result = score_pdf(pdf_path)

        if result:
            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)

    rank_digits = len(str(len(results)))

    for rank, result in enumerate(results, start=1):
        ranked_name = make_ranked_name(rank, result["original_path"], rank_digits)
        output_path = os.path.join(OUTPUT_FOLDER, ranked_name)

        copy2(result["original_path"], output_path)

        result["rank"] = rank
        result["ranked_name"] = ranked_name

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "ranked_name",
                "original_name",
                "original_path",
                "page_count",
                "score",
                "sharpness",
                "contrast",
                "brightness",
                "noise",
                "text_ratio"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("Done.")
    print(f"Ranked {len(results)} PDFs.")
    print(f"Copied ranked PDFs to: {OUTPUT_FOLDER}")
    print(f"Saved scores to: {CSV_FILE}")


if __name__ == "__main__":
    main()