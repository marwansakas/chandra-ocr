"""Create accent- and digit-normalized copies of OCR reports and score them.

This command never edits ``thesis/predictions``.  It copies every model report
folder to ``thesis/cleaned_predictions`` (unless paths are supplied), cleans
the stored transcriptions, regenerates each model's reports, and writes a
one-row-per-model ``summary.csv`` in the destination root.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import unicodedata

from .cli import PROJECT, dump_json, write_reports
from .metrics import aggregate, score


DEFAULT_CLEANED_OUTPUT = PROJECT / "thesis/cleaned_predictions"
DEFAULT_PREDICTIONS = PROJECT / "thesis/predictions"
DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def clean_text(text):
    """Remove combining marks/tatweel, translate Arabic digits, and normalize spaces."""
    text = unicodedata.normalize("NFKD", str(text)).translate(DIGIT_TRANSLATION)
    text = "".join(character for character in text
                   if not unicodedata.category(character).startswith("M") and character != "\u0640")
    return " ".join(unicodedata.normalize("NFC", text).split())


def smith_waterman_similarity(reference, prediction):
    """Return local character-alignment similarity on the 0..1 scale."""
    if not reference or not prediction:
        return 0.0
    previous = [0] * (len(prediction) + 1)
    best = 0
    for reference_char in reference:
        current = [0]
        for index, predicted_char in enumerate(prediction, 1):
            value = max(0, previous[index - 1] + (2 if reference_char == predicted_char else -1),
                        previous[index] - 1, current[index - 1] - 1)
            current.append(value)
            best = max(best, value)
        previous = current
    return min(1.0, best / (2 * max(len(reference), len(prediction))))


def text_alignment_similarity(reference, prediction):
    """Return global character-alignment similarity on the 0..1 scale."""
    longest = max(len(reference), len(prediction))
    if not longest:
        return 1.0
    previous = [-index for index in range(len(prediction) + 1)]
    for reference_index, reference_char in enumerate(reference, 1):
        current = [-reference_index]
        for prediction_index, predicted_char in enumerate(prediction, 1):
            current.append(max(previous[prediction_index - 1] + (2 if reference_char == predicted_char else -1),
                               previous[prediction_index] - 1, current[prediction_index - 1] - 1))
        previous = current
    return max(0.0, min(1.0, previous[-1] / (2 * longest)))


def clean_row(row):
    result = dict(row)
    original_reference = str(result.get("reference", ""))
    original_prediction = str(result.get("raw_prediction", result.get("prediction", "")))
    reference, prediction = clean_text(original_reference), clean_text(original_prediction)
    result.update(
        original_reference=original_reference,
        original_prediction=original_prediction,
        reference=reference,
        raw_prediction=prediction,
        prediction=prediction,
        # Older thesis reports predate the explicit status field. A non-empty
        # prediction in those reports is a completed recognition result.
        status=result.get("status") or ("ok" if prediction else "error"),
        hit_token_limit=bool(result.get("hit_token_limit", False)),
        seconds=result.get("seconds"),
    )
    result.update(score(reference, prediction, "standard"))
    result["smith_waterman_similarity"] = smith_waterman_similarity(reference, prediction)
    result["text_alignment_similarity"] = text_alignment_similarity(reference, prediction)
    return result


def percent(value):
    return f"{100 * value:.2f}" if value is not None else ""


def summary_row(model, rows):
    metrics = aggregate(rows)
    reference_chars = metrics["reference_chars"]
    return {
        "model": model,
        "evaluated_lines": len(rows),
        "missing_predictions": sum(not row["raw_prediction"].strip() or row.get("status") != "ok" for row in rows),
        "letter_accuracy_pct": percent(metrics["char_recall"]),
        "word_accuracy_pct": percent(metrics["word_recall"]),
        "line_accuracy_pct": percent(metrics["exact_match_rate"]),
        "smith_waterman_similarity_pct": percent(
            sum(row["smith_waterman_similarity"] for row in rows) / len(rows) if rows else None),
        "text_alignment_similarity_pct": percent(
            sum(row["text_alignment_similarity"] for row in rows) / len(rows) if rows else None),
        "total_reference_chars": reference_chars,
        "total_char_distance": metrics["char_edits"],
        "total_reference_words": metrics["reference_words"],
        "total_word_distance": metrics["word_edits"],
    }


def read_rows(path):
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSONL report: {path}: {error}") from error


def update_sample_files(output, rows):
    for row in rows:
        sample_id = row.get("sample_id")
        if not sample_id:
            continue
        sample_output = output / "samples" / sample_id
        sample_output.mkdir(parents=True, exist_ok=True)
        (sample_output / "prediction.txt").write_text(row["prediction"], encoding="utf-8")
        dump_json(sample_output / "metadata.json", row)


def clean_model(source, destination):
    source_rows = read_rows(source / "predictions.jsonl")
    rows = [clean_row(row) for row in source_rows]
    old_summary_path = source / "run_summary.json"
    old_summary = json.loads(old_summary_path.read_text(encoding="utf-8")) if old_summary_path.is_file() else {}
    context = {key: value for key, value in old_summary.items()
               if key not in {"metrics", "successful_only_metrics", "performance", "completed_samples",
                              "successful_samples", "failed_samples", "token_limit_samples"}}
    context.update(
        model=context.get("model", source.name),
        model_name=context.get("model_name", source.name),
        normalization="cleaned-arabic-digits",
        cleaned_from=str(source.resolve()),
        cleaning={"remove_unicode_marks": True, "remove_tatweel": True,
                  "arabic_and_persian_digits": "0123456789"},
    )
    write_reports(destination, rows, context)
    update_sample_files(destination, rows)
    return summary_row(context["model"], rows)


def write_summary(destination, summaries, source, complete):
    if summaries:
        columns = list(summaries[0])
        with (destination / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summaries)
    dump_json(destination / "cleaning_manifest.json", {
        "source": str(source), "destination": str(destination),
        "models": [item["model"] for item in summaries], "status": "complete" if complete else "running",
        "normalization": "NFKD, remove all Unicode marks and tatweel, Arabic/Persian digits to ASCII",
    })


def run(args):
    source, destination = args.input.resolve(), args.output.resolve()
    if not source.is_dir():
        raise ValueError(f"Prediction folder does not exist: {source}")
    if destination.exists() and not args.resume:
        raise ValueError(f"Destination already exists: {destination}. Choose a new --output path.")
    model_folders = sorted(path.parent for path in source.rglob("predictions.jsonl"))
    if not model_folders:
        raise ValueError(f"No model folders containing predictions.jsonl found in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    summaries = []
    for folder in model_folders:
        cleaned_model = destination / folder.relative_to(source)
        old_summary = cleaned_model / "run_summary.json"
        if args.resume and old_summary.is_file():
            saved = json.loads(old_summary.read_text(encoding="utf-8"))
            if saved.get("cleaned_from") == str(folder.resolve()) and (cleaned_model / "predictions.jsonl").is_file():
                summaries.append(summary_row(saved.get("model", folder.name), read_rows(cleaned_model / "predictions.jsonl")))
                print(f"Reused {folder.name}")
                write_summary(destination, summaries, source, complete=False)
                continue
        if not cleaned_model.exists():
            cleaned_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(folder, cleaned_model)
        summaries.append(clean_model(folder, cleaned_model))
        print(f"Cleaned {folder.name}")
        # Persist progress so a scheduler timeout still leaves a useful partial summary.
        write_summary(destination, summaries, source, complete=False)
    write_summary(destination, summaries, source, complete=True)
    print(f"Wrote cleaned reports and summary: {destination}")
    return 0


def parser():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--input", type=Path, default=DEFAULT_PREDICTIONS,
                                 help="source thesis/predictions folder")
    argument_parser.add_argument("--output", type=Path, default=DEFAULT_CLEANED_OUTPUT,
                                 help="new cleaned-report folder; must not already exist")
    argument_parser.add_argument("--resume", action="store_true",
                                 help="reuse already-cleaned model folders in an interrupted output")
    argument_parser.set_defaults(func=run)
    return argument_parser


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as error:
        argument = parser()
        argument.error(str(error))


if __name__ == "__main__":
    main()
