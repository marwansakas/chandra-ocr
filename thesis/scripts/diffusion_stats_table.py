"""Render saved diffusion scores using the project's OCR comparison columns."""


def format_evaluation_table(evaluation, policy="historical", model_prefix=None):
    key = {"historical": "historical_normalized_all_expected",
           "strict": "strict_all_expected"}[policy]
    rows = []
    for model, result in evaluation["models"].items():
        scores = result[key]
        # Match common.finalize_ocr_aggregate, including zero denominators.
        letter = (max(0.0, 1 - scores["char_distance"] / scores["reference_chars"])
                  if scores["reference_chars"] else 0.0)
        word = (max(0.0, 1 - scores["word_distance"] / scores["reference_words"])
                if scores["reference_words"] else 0.0)
        exact = scores["exact_lines"] / scores["lines"] if scores["lines"] else 0.0
        name = f"{model}/{model_prefix}" if model_prefix else model
        rows.append((name, scores["lines"], 100 * letter, 100 * word, 100 * exact))
    width = max(32, *(len(row[0]) for row in rows)) if rows else 32
    lines = [f"{'Model':<{width}} {'Lines':>7} {'Letter %':>11} {'Word %':>11} {'Exact %':>11}"]
    lines.extend(f"{name:<{width}} {count:>7} {letter:>11.2f} {word:>11.2f} {exact:>11.2f}"
                 for name, count, letter, word, exact in rows)
    lines.extend(["", f"Partition: {evaluation['partition']} | Policy: {policy} | Coverage: all expected lines",
                  "Letter/Word % = 100 * max(0, 1 - CER/WER); Exact % = fully matching lines."])
    if policy == "historical":
        lines.append("Uses the other OCR models' project normalization and word tokenization.")
    lines.append("Use JSON for unclamped error rates, failures and detailed statistics.")
    return "\n".join(lines)
