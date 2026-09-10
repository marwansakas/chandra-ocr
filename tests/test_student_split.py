"""Writer-disjoint splitting should target line counts, not writer counts."""
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thesis/scripts"))
from train_line_ocr_model import split_records_by_student, student_group


def records_for(sizes):
    return [SimpleNamespace(sample_id=f"{writer}-{line}",
                            page_id=f"writer{writer}_Page_{line % 3:03d}")
            for writer, size in enumerate(sizes) for line in range(size)]


def test_unequal_writers_reach_exact_line_ratio_without_leakage():
    records = records_for([50, 20, 15, 10, 5])
    train, test = split_records_by_student(records, 0.8, 42)
    assert (len(train), len(test)) == (80, 20)
    assert {student_group(r.page_id) for r in train}.isdisjoint(
        student_group(r.page_id) for r in test)
    assert sorted(r.sample_id for r in train + test) == sorted(r.sample_id for r in records)
    assert (train, test) == split_records_by_student(records, 0.8, 42)


def test_unattainable_ratio_is_globally_closest():
    sizes = [39, 28, 17, 11]
    train, test = split_records_by_student(records_for(sizes), 0.8, 42)
    target = sum(sizes) * 0.8
    best_error = min(abs(sum(subset) - target)
                     for n in range(1, len(sizes))
                     for subset in combinations(sizes, n))
    assert abs(len(train) - target) == best_error
    assert train and test


def test_empty_and_single_writer():
    assert split_records_by_student([], 0.8, 42) == ([], [])
    records = records_for([10])
    assert split_records_by_student(records, 0.8, 42) == (records, [])
