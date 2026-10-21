"""Corpus edit rates, ignoring Unicode accents and diacritics by default."""

import unicodedata


DEFAULT_NORMALIZATION = "arabic-light"


def normalize(text, mode=DEFAULT_NORMALIZATION):
    if mode == "raw":
        return text
    text = unicodedata.normalize("NFC", text)
    if mode == "arabic-light":
        text = unicodedata.normalize("NFKD", text).replace("\u0640", "")
        text = "".join(
            c for c in text
            if not unicodedata.category(c).startswith("M")
        )
        text = unicodedata.normalize("NFC", text)
    elif mode != "standard":
        raise ValueError(f"Unknown normalization: {mode}")
    return " ".join(text.split())


def edit_distance(reference, prediction):
    # Use the shorter sequence as the row to bound memory.
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def edit_operations(reference, prediction):
    """Return Levenshtein substitutions, deletions and insertions."""
    previous = [(index, 0, index, 0) for index in range(len(prediction) + 1)]
    for row, reference_item in enumerate(reference, 1):
        current = [(row, 0, row, 0)]
        for column, prediction_item in enumerate(prediction, 1):
            if reference_item == prediction_item:
                current.append(previous[column - 1])
                continue
            substitution, deletion, insertion = previous[column - 1], previous[column], current[column - 1]
            candidates = (
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            )
            current.append(min(candidates, key=lambda item: item[0]))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return substitutions, deletions, insertions


def precision_recall_f1(correct, reference_total, prediction_total):
    precision = correct / prediction_total if prediction_total else None
    recall = correct / reference_total if reference_total else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return precision, recall, f1


def smith_waterman_similarity(reference, prediction):
    """Best local character alignment, normalized to 0..1."""
    if not reference or not prediction:
        return 0.0
    previous = [0] * (len(prediction) + 1)
    best = 0
    for a in reference:
        current = [0]
        for j, b in enumerate(prediction, 1):
            current.append(max(0, previous[j - 1] + (2 if a == b else -1),
                               previous[j] - 1, current[j - 1] - 1))
            best = max(best, current[-1])
        previous = current
    return min(1.0, best / (2 * max(len(reference), len(prediction))))


def text_alignment_similarity(reference, prediction):
    """Whole-line global alignment similarity, normalized to 0..1."""
    longest = max(len(reference), len(prediction))
    if not longest:
        return 1.0
    previous = [-j for j in range(len(prediction) + 1)]
    for i, a in enumerate(reference, 1):
        current = [-i]
        for j, b in enumerate(prediction, 1):
            current.append(max(previous[j - 1] + (2 if a == b else -1),
                               previous[j] - 1, current[j - 1] - 1))
        previous = current
    return max(0.0, min(1.0, previous[-1] / (2 * longest)))


def score(reference, prediction, mode=DEFAULT_NORMALIZATION):
    reference, prediction = normalize(reference, mode), normalize(prediction, mode)
    chars = len(reference)
    words = len(reference.split())
    predicted_chars = len(prediction)
    predicted_words = len(prediction.split())
    char_substitutions, char_deletions, char_insertions = edit_operations(reference, prediction)
    word_substitutions, word_deletions, word_insertions = edit_operations(reference.split(), prediction.split())
    char_edits = char_substitutions + char_deletions + char_insertions
    word_edits = word_substitutions + word_deletions + word_insertions
    char_correct = chars - char_substitutions - char_deletions
    word_correct = words - word_substitutions - word_deletions
    char_precision, char_recall, char_f1 = precision_recall_f1(char_correct, chars, predicted_chars)
    word_precision, word_recall, word_f1 = precision_recall_f1(word_correct, words, predicted_words)
    return {
        "reference_chars": chars, "reference_words": words,
        "predicted_chars": predicted_chars, "predicted_words": predicted_words,
        "char_correct": char_correct, "word_correct": word_correct,
        "char_substitutions": char_substitutions, "char_deletions": char_deletions, "char_insertions": char_insertions,
        "word_substitutions": word_substitutions, "word_deletions": word_deletions, "word_insertions": word_insertions,
        "char_edits": char_edits, "word_edits": word_edits,
        "cer": char_edits / chars if chars else None,
        "wer": word_edits / words if words else None,
        "char_precision": char_precision, "char_recall": char_recall, "char_f1": char_f1,
        "word_precision": word_precision, "word_recall": word_recall, "word_f1": word_f1,
        "exact_match": reference == prediction,
        "smith_waterman_similarity": smith_waterman_similarity(reference, prediction),
        "text_alignment_similarity": text_alignment_similarity(reference, prediction),
    }


def aggregate(rows):
    result = {key: sum(row[key] for row in rows) for key in
              ("reference_chars", "reference_words", "predicted_chars", "predicted_words",
               "char_correct", "word_correct", "char_substitutions", "char_deletions", "char_insertions",
               "word_substitutions", "word_deletions", "word_insertions", "char_edits", "word_edits")}
    char_precision, char_recall, char_f1 = precision_recall_f1(
        result["char_correct"], result["reference_chars"], result["predicted_chars"])
    word_precision, word_recall, word_f1 = precision_recall_f1(
        result["word_correct"], result["reference_words"], result["predicted_words"])
    result.update(
        samples=len(rows),
        cer=result["char_edits"] / result["reference_chars"] if result["reference_chars"] else None,
        wer=result["word_edits"] / result["reference_words"] if result["reference_words"] else None,
        char_precision=char_precision, char_recall=char_recall, char_f1=char_f1,
        word_precision=word_precision, word_recall=word_recall, word_f1=word_f1,
        exact_match_rate=sum(row["exact_match"] for row in rows) / len(rows) if rows else None,
        smith_waterman_similarity=sum(row["smith_waterman_similarity"] for row in rows) / len(rows) if rows else None,
        text_alignment_similarity=sum(row["text_alignment_similarity"] for row in rows) / len(rows) if rows else None,
    )
    return result
