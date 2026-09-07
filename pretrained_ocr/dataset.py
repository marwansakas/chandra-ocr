"""Read image.png + ground_truth.txt folders without modifying source files."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: Path
    reference: str

    def fingerprint(self):
        digest = sha256(self.image.read_bytes())
        digest.update(b"\x00")
        digest.update(self.reference.encode("utf-8"))
        return digest.hexdigest()


def read_dataset(root: Path):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset folder does not exist: {root}")
    samples, skipped = [], []
    folders = sorted(
        {p.parent for name in ("image.png", "ground_truth.txt") for p in root.rglob(name)},
        key=lambda p: p.relative_to(root).as_posix(),
    )
    for folder in folders:
        sample_id = folder.relative_to(root).as_posix()
        image, label = folder / "image.png", folder / "ground_truth.txt"
        reason = None
        if not image.is_file():
            reason = "missing_image"
        elif not label.is_file():
            reason = "missing_ground_truth"
        else:
            reference = label.read_bytes().decode("utf-8-sig")
            if not reference.strip():
                reason = "empty_ground_truth"
            elif reference.strip().casefold() == "ignore":
                reason = "ignore_marker"
        if reason:
            skipped.append({"sample_id": sample_id, "reason": reason})
        else:
            samples.append(Sample(sample_id, image.resolve(), reference))
    if not samples:
        raise ValueError(f"No labeled image.png samples found in {root}. Skipped: {len(skipped)}")
    return samples, skipped


def select_samples(samples, start_line=1, limit=None):
    if start_line < 1 or (limit is not None and limit < 1):
        raise ValueError("--start-line and --limit must be positive")
    selected = samples[start_line - 1:]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("The requested sample range is empty")
    return selected


def check_output_path(dataset: Path, output: Path):
    dataset, output = dataset.resolve(), output.resolve()
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Output and dataset directories must not overlap")
