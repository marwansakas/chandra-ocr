"""Checkpoint selection keeps exact-line quality without sacrificing CER."""

import pytest

from thesis.diffusion.selection import checkpoint_candidate, update_exact_line_frontier


def candidate(epoch, cer, exact):
    return checkpoint_candidate(f"checkpoint-{epoch:04d}", epoch, cer, exact)


def test_exact_lines_select_within_half_point_cer_band():
    frontier, selected, saved = update_exact_line_frontier([], candidate(25, 0.8246251703771014, 0), 0.8246251703771014)
    assert saved and selected["epoch"] == 25

    frontier, selected, saved = update_exact_line_frontier(
        frontier, candidate(45, 0.8293957292139936, 3), 0.8246251703771014
    )
    assert saved and selected["epoch"] == 45

    frontier, selected, saved = update_exact_line_frontier(frontier, candidate(66, 0.8567, 4), 0.8246251703771014)
    assert not saved and selected["epoch"] == 45


def test_lower_absolute_best_moves_band_and_keeps_needed_frontier_models():
    frontier, _, _ = update_exact_line_frontier([], candidate(10, 0.824, 0), 0.824)
    frontier, _, _ = update_exact_line_frontier(frontier, candidate(11, 0.827, 3), 0.824)
    frontier, _, _ = update_exact_line_frontier(frontier, candidate(12, 0.825, 1), 0.824)

    frontier, selected, saved = update_exact_line_frontier(frontier, candidate(20, 0.820, 0), 0.820)
    assert saved
    assert {item["epoch"] for item in frontier} == {12, 20}
    assert selected["epoch"] == 12


def test_dominated_epoch_is_not_preserved_and_ties_are_deterministic():
    frontier, _, _ = update_exact_line_frontier([], candidate(3, 0.80, 2), 0.80)
    frontier, selected, saved = update_exact_line_frontier(frontier, candidate(4, 0.801, 1), 0.80)
    assert not saved and selected["epoch"] == 3

    frontier, selected, saved = update_exact_line_frontier(frontier, candidate(5, 0.80, 2), 0.80)
    assert not saved and selected["epoch"] == 3


def test_cer_band_is_inclusive_and_validated():
    frontier, selected, saved = update_exact_line_frontier([], candidate(1, 0.805, 1), 0.80)
    assert saved and selected["epoch"] == 1
    with pytest.raises(ValueError, match="nonnegative"):
        update_exact_line_frontier(frontier, candidate(2, 0.79, 0), 0.79, cer_band=-0.1)
