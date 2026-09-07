"""Read image.png + ground_truth.txt folders without modifying source files."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath, PureWindowsPath


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


def read_test_manifest(path: Path):
    """Use project test.jsonl sample IDs; never trust stored image paths as inputs."""
    raw = path.read_bytes()
    sample_ids, seen = [], set()
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}, line {line_number}: {error.msg}") from error
        sample_id = record.get("sample_id") if isinstance(record, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Missing string sample_id in {path}, line {line_number}")
        relative = PurePosixPath(sample_id)
        if (relative.is_absolute() or PureWindowsPath(sample_id).drive or "\\" in sample_id
                or ".." in relative.parts or sample_id == "." or relative.as_posix() != sample_id):
            raise ValueError(f"sample_id must be a relative sample folder: {sample_id!r}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in test manifest: {sample_id}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
    if not sample_ids:
        raise ValueError(f"Test manifest contains no samples: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(raw).hexdigest(), "sample_ids": sample_ids}


def read_dataset(root: Path, sample_ids=None):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset folder does not exist: {root}")
    samples, skipped = [], []
    if sample_ids is None:
        folders = sorted(
            {p.parent for name in ("image.png", "ground_truth.txt") for p in root.rglob(name)},
            key=lambda p: p.relative_to(root).as_posix(),
        )
    else:
        folders = [root / sample_id for sample_id in sorted(sample_ids)]
    for folder in folders:
        sample_id = folder.relative_to(root).as_posix()
        image, label = folder / "image.png", folder / "ground_truth.txt"
        if sample_ids is not None and any(not path.resolve().is_relative_to(root) for path in (image, label)):
            raise ValueError(f"Held-out sample points outside the dataset: {sample_id}")
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
    if sample_ids is not None and skipped:
        details = "; ".join(f"{item['sample_id']}: {item['reason']}" for item in skipped[:5])
        raise ValueError(f"{len(skipped)} held-out samples are missing or unusable: {details}")
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
