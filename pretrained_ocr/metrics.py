"""Corpus edit rates, ignoring Arabic diacritics by default."""

import unicodedata


DEFAULT_NORMALIZATION = "arabic-light"


def normalize(text, mode=DEFAULT_NORMALIZATION):
    if mode == "raw":
        return text
    text = unicodedata.normalize("NFC", text)
    if mode == "arabic-light":
        text = unicodedata.normalize("NFKC", text).replace("\u0640", "")
        text = "".join(
            c for c in text
            if not (unicodedata.category(c).startswith("M") and "ARABIC" in unicodedata.name(c, ""))
        )
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
    char_edits = edit_distance(reference, prediction)
    word_edits = edit_distance(reference.split(), prediction.split())
    return {
        "reference_chars": chars, "reference_words": words,
        "char_edits": char_edits, "word_edits": word_edits,
        "cer": char_edits / chars if chars else None,
        "wer": word_edits / words if words else None,
        "exact_match": reference == prediction,
        "smith_waterman_similarity": smith_waterman_similarity(reference, prediction),
        "text_alignment_similarity": text_alignment_similarity(reference, prediction),
    }


def aggregate(rows):
    result = {key: sum(row[key] for row in rows) for key in
              ("reference_chars", "reference_words", "char_edits", "word_edits")}
    result.update(
        samples=len(rows),
        cer=result["char_edits"] / result["reference_chars"] if result["reference_chars"] else None,
        wer=result["word_edits"] / result["reference_words"] if result["reference_words"] else None,
        exact_match_rate=sum(row["exact_match"] for row in rows) / len(rows) if rows else None,
        smith_waterman_similarity=sum(row["smith_waterman_similarity"] for row in rows) / len(rows) if rows else None,
        text_alignment_similarity=sum(row["text_alignment_similarity"] for row in rows) / len(rows) if rows else None,
    )
    return result
