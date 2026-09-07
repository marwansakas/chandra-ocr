"""Corpus edit rates, preserving spelling and Arabic letter distinctions."""

import unicodedata


def normalize(text, mode="standard"):
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


def score(reference, prediction, mode="standard"):
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
    }


def aggregate(rows):
    result = {key: sum(row[key] for row in rows) for key in
              ("reference_chars", "reference_words", "char_edits", "word_edits")}
    result.update(
        samples=len(rows),
        cer=result["char_edits"] / result["reference_chars"] if result["reference_chars"] else None,
        wer=result["word_edits"] / result["reference_words"] if result["reference_words"] else None,
        exact_match_rate=sum(row["exact_match"] for row in rows) / len(rows) if rows else None,
    )
    return result
