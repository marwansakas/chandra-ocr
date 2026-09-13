"""Literal and historical metrics with deterministic edit alignments."""

from __future__ import annotations

from collections import Counter
from thesis.scripts.common import normalize_text_for_evaluation, tokenize_words


def align(reference, prediction):
    """Unit-cost Levenshtein alignment; ties prefer diagonal, deletion, insertion."""
    costs = [list(range(len(prediction) + 1))]
    for i, expected in enumerate(reference, 1):
        row = [i]
        for j, actual in enumerate(prediction, 1):
            row.append(min(costs[-1][j - 1] + (expected != actual), costs[-1][j] + 1, row[-1] + 1))
        costs.append(row)
    i, j, operations = len(reference), len(prediction), []
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]):
            operations.append(("equal" if reference[i - 1] == prediction[j - 1] else "substitution", i - 1, reference[i - 1], prediction[j - 1]))
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            operations.append(("deletion", i - 1, reference[i - 1], None))
            i -= 1
        else:
            operations.append(("insertion", None, None, prediction[j - 1]))
            j -= 1
    return operations[::-1]


def score_pair(reference: str, prediction: str, *, historical=False):
    if historical:
        reference = normalize_text_for_evaluation(reference)
        prediction = normalize_text_for_evaluation(prediction)
        ref_words, pred_words = tokenize_words(reference), tokenize_words(prediction)
    else:
        ref_words, pred_words = reference.split(), prediction.split()
    operations = align(reference, prediction)
    counts = Counter(op[0] for op in operations)
    return {"reference_chars": len(reference), "reference_words": len(ref_words),
            "substitutions": counts["substitution"], "insertions": counts["insertion"], "deletions": counts["deletion"],
            "char_distance": sum(counts[k] for k in ("substitution", "insertion", "deletion")),
            "word_distance": sum(op[0] != "equal" for op in align(ref_words, pred_words)),
            "exact_lines": int(reference == prediction),
            "space_insertions": sum(op == "insertion" and actual == " " for op, _, expected, actual in operations),
            "space_deletions": sum(op == "deletion" and expected == " " for op, _, expected, actual in operations),
            "space_substitutions": sum(op == "substitution" and (expected == " " or actual == " ") for op, _, expected, actual in operations)}


def aggregate(pairs, *, historical=False):
    totals = Counter()
    count = 0
    for reference, prediction in pairs:
        totals.update(score_pair(reference, prediction, historical=historical))
        count += 1
    return {**dict(totals), "lines": count,
            "cer": totals["char_distance"] / totals["reference_chars"] if totals["reference_chars"] else None,
            "wer": totals["word_distance"] / totals["reference_words"] if totals["reference_words"] else None,
            "exact_line_accuracy": totals["exact_lines"] / count if count else None}


def correction_changes(reference: str, before: str, after: str):
    before_correct = {index for op, index, _, _ in align(reference, before) if op == "equal"}
    after_correct = {index for op, index, _, _ in align(reference, after) if op == "equal"}
    return {"previously_correct_reference_chars": len(before_correct),
            "damaged_reference_chars": len(before_correct - after_correct),
            "fixed_reference_chars": len(after_correct - before_correct),
            "char_error_delta": score_pair(reference, after)["char_distance"] - score_pair(reference, before)["char_distance"]}
