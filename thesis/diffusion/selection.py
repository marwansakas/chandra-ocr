"""Validation-only checkpoint selection for diffusion OCR training."""

from __future__ import annotations


EXACT_LINE_CER_BAND = 0.005  # 0.5 absolute CER percentage points.


def checkpoint_candidate(checkpoint: str, epoch: int, strict_cer: float, exact_lines: int) -> dict:
    return {
        "checkpoint": checkpoint,
        "epoch": int(epoch),
        "strict_cer": float(strict_cer),
        "exact_lines": int(exact_lines),
    }


def _dominates(left: dict, right: dict) -> bool:
    """Return whether left can always replace right in any future lower CER band."""
    no_worse = left["strict_cer"] <= right["strict_cer"] and left["exact_lines"] >= right["exact_lines"]
    strictly_better = left["strict_cer"] < right["strict_cer"] or left["exact_lines"] > right["exact_lines"]
    same_metrics_earlier = (
        left["strict_cer"] == right["strict_cer"]
        and left["exact_lines"] == right["exact_lines"]
        and left["epoch"] < right["epoch"]
    )
    return no_worse and (strictly_better or same_metrics_earlier)


def update_exact_line_frontier(
    candidates: list[dict],
    current: dict,
    absolute_best_cer: float,
    cer_band: float = EXACT_LINE_CER_BAND,
) -> tuple[list[dict], dict, bool]:
    """Keep the checkpoints that can still win the exact-line near-tie rule.

    An epoch outside the current band cannot enter a later band because the
    absolute-best CER can only decrease. Dominated epochs cannot beat their
    dominator under any later band, so their weights do not need preserving.
    """
    if cer_band < 0:
        raise ValueError("CER band must be nonnegative")
    limit = absolute_best_cer + cer_band
    frontier = [dict(item) for item in candidates if item["strict_cer"] <= limit]
    keep_current = current["strict_cer"] <= limit and not any(_dominates(item, current) for item in frontier)
    if keep_current:
        frontier = [item for item in frontier if not _dominates(current, item)]
        frontier.append(dict(current))
    frontier.sort(key=lambda item: item["epoch"])
    if not frontier:
        raise ValueError("Checkpoint frontier cannot be empty")
    selected = min(frontier, key=lambda item: (-item["exact_lines"], item["strict_cer"], item["epoch"]))
    return frontier, selected, keep_current
