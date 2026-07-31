"""Detect text lines with Surya without requiring Chandra."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    command = [
        sys.executable,
        str(ROOT / "build_surya_line_ground_truth.py"),
        "--detect-only",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()

