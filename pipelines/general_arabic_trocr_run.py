"""Run TrOCR with conservative general-Arabic dictionary-aware decoding."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Protocol, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = ROOT / "thesis" / "datasets" / "ground_truth_lines_ml"
DEFAULT_MODEL = ROOT / "thesis" / "models" / "ground_truth_line_trocr"
DEFAULT_OUTPUT = ROOT / "thesis" / "predictions" / "general_arabic_TrOCR"
DEFAULT_DICTIONARY = ROOT / "thesis" / "dictionaries" / "general_arabic" / "ar"
WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
ARABIC_MARK_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
CONFUSION_GROUPS = (
    frozenset("بتثني"),
    frozenset("پب"),
    frozenset("جحخ"),
    frozenset("دذ"),
    frozenset("رز"),
    frozenset("سش"),
    frozenset("صض"),
    frozenset("طظ"),
    frozenset("عغ"),
    frozenset("فق"),
    frozenset("هةت"),
    frozenset("يى"),
    frozenset("اأإآٱ"),
    frozenset("ءأؤئ"),
)


class Lexicon(Protocol):
    def lookup(self, word: str) -> bool: ...

    def suggest(self, word: str, limit: int) -> Sequence[str]: ...


class HunspellLexicon:
    """Small cached adapter around the pure-Python Spylls Hunspell reader."""

    def __init__(self, dictionary_stem: Path):
        try:
            from spylls.hunspell import Dictionary
        except ImportError as exc:
            raise SystemExit("Missing dependency 'spylls'. Run: python -m pip install -r requirements.txt") from exc
        self.dictionary = Dictionary.from_files(str(dictionary_stem))

    @lru_cache(maxsize=100_000)
    def lookup(self, word: str) -> bool:
        return bool(self.dictionary.lookup(word))

    @lru_cache(maxsize=50_000)
    def _suggest_cached(self, word: str, limit: int) -> tuple[str, ...]:
        return tuple(islice(self.dictionary.suggest(word), limit))

    def suggest(self, word: str, limit: int) -> Sequence[str]:
        return self._suggest_cached(word, limit)


@dataclass(frozen=True)
class WordCorrection:
    original: str
    replacement: str
    edit_distance: int
    normalized_distance: float


@dataclass
class CandidateScore:
    candidate_index: int
    raw_text: str
    corrected_text: str
    visual_score: float
    final_score: float
    dictionary_coverage: float
    unresolved_words: int
    corrections: list[WordCorrection]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def normalize_lookup_word(word: str) -> str:
    word = unicodedata.normalize("NFKC", word)
    word = word.replace("ـ", "").replace("ک", "ك").replace("ی", "ي")
    return ARABIC_MARK_RE.sub("", word)


def edit_distance(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for index, char_a in enumerate(first, start=1):
        current = [index]
        for column, char_b in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def substitution_cost(first: str, second: str) -> float:
    if first == second:
        return 0.0
    if any(first in group and second in group for group in CONFUSION_GROUPS):
        return 0.35
    return 1.0


def weighted_edit_distance(first: str, second: str) -> float:
    """Arabic-aware distance used only to rank already-valid suggestions."""
    if len(first) < len(second):
        first, second = second, first
    previous = [float(index) for index in range(len(second) + 1)]
    for index, char_a in enumerate(first, start=1):
        current = [float(index)]
        for column, char_b in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1.0,
                    previous[column] + 1.0,
                    previous[column - 1] + substitution_cost(char_a, char_b),
                )
            )
        previous = current
    return previous[-1]


def correct_word(
    word: str,
    lexicon: Lexicon,
    *,
    max_edit_distance: int,
    max_distance_ratio: float,
    suggestion_limit: int,
) -> WordCorrection | None:
    lookup_word = normalize_lookup_word(word)
    if not lookup_word or not ARABIC_RE.search(lookup_word) or lexicon.lookup(lookup_word):
        return None
    ranked: list[tuple[float, int, int, int, str]] = []
    seen: set[str] = set()
    for rank, suggestion in enumerate(lexicon.suggest(lookup_word, suggestion_limit)):
        suggestion = normalize_lookup_word(suggestion)
        if not suggestion or suggestion in seen:
            continue
        seen.add(suggestion)
        distance = edit_distance(lookup_word, suggestion)
        ranked.append(
            (
                weighted_edit_distance(lookup_word, suggestion),
                distance,
                abs(len(lookup_word) - len(suggestion)),
                rank,
                suggestion,
            )
        )
    if not ranked:
        return None
    _, distance, _, _, replacement = min(ranked)
    ratio = distance / max(len(lookup_word), len(replacement), 1)
    if distance == 0 or distance > max_edit_distance or ratio > max_distance_ratio:
        return None
    return WordCorrection(word, replacement, distance, ratio)


def correct_text(
    text: str,
    lexicon: Lexicon,
    *,
    max_edit_distance: int,
    max_distance_ratio: float,
    suggestion_limit: int,
) -> tuple[str, list[WordCorrection]]:
    corrections: list[WordCorrection] = []
    output: list[str] = []
    cursor = 0
    for match in WORD_RE.finditer(text):
        output.append(text[cursor : match.start()])
        word = match.group(0)
        correction = correct_word(
            word,
            lexicon,
            max_edit_distance=max_edit_distance,
            max_distance_ratio=max_distance_ratio,
            suggestion_limit=suggestion_limit,
        )
        if correction is None:
            output.append(word)
        else:
            output.append(correction.replacement)
            corrections.append(correction)
        cursor = match.end()
    output.append(text[cursor:])
    return "".join(output), corrections


def dictionary_stats(text: str, lexicon: Lexicon) -> tuple[int, int]:
    words = [normalize_lookup_word(match.group(0)) for match in WORD_RE.finditer(text)]
    arabic_words = [word for word in words if word and ARABIC_RE.search(word)]
    valid = sum(1 for word in arabic_words if lexicon.lookup(word))
    return valid, len(arabic_words) - valid


def score_candidate(
    candidate_index: int,
    text: str,
    visual_score: float,
    lexicon: Lexicon,
    *,
    dictionary_weight: float,
    edit_penalty: float,
    oov_penalty: float,
    max_edit_distance: int,
    max_distance_ratio: float,
    suggestion_limit: int,
) -> CandidateScore:
    corrected, corrections = correct_text(
        text,
        lexicon,
        max_edit_distance=max_edit_distance,
        max_distance_ratio=max_distance_ratio,
        suggestion_limit=suggestion_limit,
    )
    valid, unresolved = dictionary_stats(corrected, lexicon)
    total = valid + unresolved
    coverage = valid / total if total else 0.0
    edit_cost = sum(item.normalized_distance for item in corrections)
    unresolved_ratio = unresolved / total if total else 0.0
    final_score = visual_score + dictionary_weight * coverage - edit_penalty * edit_cost - oov_penalty * unresolved_ratio
    return CandidateScore(
        candidate_index=candidate_index,
        raw_text=text,
        corrected_text=corrected,
        visual_score=visual_score,
        final_score=final_score,
        dictionary_coverage=coverage,
        unresolved_words=unresolved,
        corrections=corrections,
    )


def choose_prediction(
    texts: Sequence[str],
    visual_scores: Sequence[float],
    lexicon: Lexicon,
    *,
    dictionary_weight: float,
    edit_penalty: float,
    oov_penalty: float,
    min_improvement: float,
    max_edit_distance: int,
    max_distance_ratio: float,
    suggestion_limit: int,
) -> tuple[str, CandidateScore, float, bool, list[CandidateScore]]:
    if not texts or len(texts) != len(visual_scores):
        raise ValueError("Candidate texts and visual scores must have the same non-zero length")
    scored = [
        score_candidate(
            index,
            text,
            float(visual_scores[index]),
            lexicon,
            dictionary_weight=dictionary_weight,
            edit_penalty=edit_penalty,
            oov_penalty=oov_penalty,
            max_edit_distance=max_edit_distance,
            max_distance_ratio=max_distance_ratio,
            suggestion_limit=suggestion_limit,
        )
        for index, text in enumerate(texts)
    ]
    best = max(scored, key=lambda item: (item.final_score, -item.candidate_index))
    raw_valid, raw_unresolved = dictionary_stats(texts[0], lexicon)
    raw_total = raw_valid + raw_unresolved
    raw_coverage = raw_valid / raw_total if raw_total else 0.0
    raw_score = float(visual_scores[0]) + dictionary_weight * raw_coverage
    if raw_total:
        raw_score -= oov_penalty * (raw_unresolved / raw_total)
    use_dictionary = best.corrected_text != texts[0] and best.final_score >= raw_score + min_improvement
    return (best.corrected_text if use_dictionary else texts[0]), best, raw_score, use_dictionary, scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrOCR with general-Arabic dictionary-aware beam decoding.")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_SPLIT_ROOT / "test.jsonl")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY, help="Hunspell path without .aff/.dic")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--dictionary-weight", type=float, default=0.8)
    parser.add_argument("--edit-penalty", type=float, default=0.35)
    parser.add_argument("--oov-penalty", type=float, default=0.4)
    parser.add_argument("--min-improvement", type=float, default=0.1)
    parser.add_argument("--max-edit-distance", type=int, default=2)
    parser.add_argument("--max-distance-ratio", type=float, default=0.34)
    parser.add_argument("--suggestion-limit", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Missing test manifest: {path}")
    return [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip()]


def serializable_candidate(candidate: CandidateScore) -> dict:
    row = asdict(candidate)
    row["visual_score"] = round(candidate.visual_score, 6)
    row["final_score"] = round(candidate.final_score, 6)
    row["dictionary_coverage"] = round(candidate.dictionary_coverage, 6)
    return row


def main() -> None:
    args = parse_args()
    args.test_manifest = project_path(args.test_manifest)
    args.model = project_path(args.model)
    args.output_root = project_path(args.output_root)
    args.dictionary = project_path(args.dictionary)
    if args.batch_size < 1 or args.max_length < 1:
        raise SystemExit("--batch-size and --max-length must be at least 1")
    if args.num_beams < 1 or not 1 <= args.num_candidates <= args.num_beams:
        raise SystemExit("--num-candidates must be between 1 and --num-beams")
    if args.max_edit_distance < 0 or not 0 <= args.max_distance_ratio <= 1:
        raise SystemExit("Invalid dictionary edit-distance settings")

    records = read_manifest(args.test_manifest)
    if args.dry_run:
        for record in records:
            print(f"{record['sample_id']}: {record['image_path']}")
        return
    if not args.model.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model}")
    if not args.dictionary.with_suffix(".aff").is_file() or not args.dictionary.with_suffix(".dic").is_file():
        raise SystemExit(
            f"Dictionary files are missing for {args.dictionary}. "
            "Run: python thesis/scripts/download_general_arabic_dictionary.py"
        )

    try:
        import torch
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise SystemExit(f"Model dependencies are missing: {exc}") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    lexicon = HunspellLexicon(args.dictionary)
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()

    rows: list[dict] = []
    skipped = 0
    corrected_lines = 0
    started = time.monotonic()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(records), args.batch_size):
        pending = []
        for record in records[offset : offset + args.batch_size]:
            output_dir = args.output_root / record["sample_id"]
            prediction_path = output_dir / "prediction.txt"
            if prediction_path.is_file() and not args.force:
                skipped += 1
                continue
            image_path = project_path(Path(record["image_path"]))
            pending.append((record, image_path, output_dir, prediction_path))
        if not pending:
            continue

        images = [Image.open(image_path).convert("RGB") for _, image_path, _, _ in pending]
        pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated = model.generate(
                pixels,
                max_length=args.max_length,
                num_beams=args.num_beams,
                num_return_sequences=args.num_candidates,
                return_dict_in_generate=True,
                output_scores=True,
            )
        texts = processor.batch_decode(generated.sequences, skip_special_tokens=True)
        scores_tensor = getattr(generated, "sequences_scores", None)
        scores = scores_tensor.detach().cpu().tolist() if scores_tensor is not None else [0.0] * len(texts)

        for pending_index, (record, image_path, output_dir, prediction_path) in enumerate(pending):
            start = pending_index * args.num_candidates
            end = start + args.num_candidates
            prediction, best, raw_score, used_dictionary, candidates = choose_prediction(
                texts[start:end],
                scores[start:end],
                lexicon,
                dictionary_weight=args.dictionary_weight,
                edit_penalty=args.edit_penalty,
                oov_penalty=args.oov_penalty,
                min_improvement=args.min_improvement,
                max_edit_distance=args.max_edit_distance,
                max_distance_ratio=args.max_distance_ratio,
                suggestion_limit=args.suggestion_limit,
            )
            raw_prediction = texts[start].strip()
            output_dir.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text(prediction.strip() + "\n", encoding="utf-8")
            (output_dir / "raw_prediction.txt").write_text(raw_prediction + "\n", encoding="utf-8")
            (output_dir / "candidates.json").write_text(
                json.dumps([serializable_candidate(item) for item in candidates], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            row = {
                "sample_id": record["sample_id"],
                "page_id": record["page_id"],
                "line_index": record["line_index"],
                "image_path": str(image_path.relative_to(ROOT)),
                "prediction_path": str(prediction_path.relative_to(ROOT)),
                "model": "general_arabic_TrOCR",
                "dictionary_used": used_dictionary,
                "raw_score": round(raw_score, 6) if math.isfinite(raw_score) else raw_score,
                "selected_score": round(best.final_score, 6) if math.isfinite(best.final_score) else best.final_score,
                "corrections": [asdict(item) for item in best.corrections] if used_dictionary else [],
            }
            (output_dir / "metadata.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            rows.append(row)
            corrected_lines += int(used_dictionary)
            marker = "dictionary" if used_dictionary else "raw fallback"
            print(f"{record['sample_id']} [{marker}]: {prediction.strip()}", flush=True)

    with (args.output_root / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_root / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["sample_id", "page_id", "line_index", "image_path", "prediction_path", "model", "dictionary_used"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": "general_arabic_TrOCR",
        "test_samples": len(records),
        "completed": len(rows),
        "skipped": skipped,
        "dictionary_corrected_lines": corrected_lines,
        "raw_fallback_lines": len(rows) - corrected_lines,
        "device": str(device),
        "dictionary": str(args.dictionary),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
