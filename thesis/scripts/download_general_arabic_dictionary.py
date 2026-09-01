"""Download the pinned LibreOffice Arabic Hunspell dictionary."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import common


UPSTREAM_COMMIT = "32b006a2c22a4ac7e8ed3f03346f7b3d85a970a4"
UPSTREAM_ROOT = f"https://raw.githubusercontent.com/LibreOffice/dictionaries/{UPSTREAM_COMMIT}/ar"
FILES = ("ar.aff", "ar.dic", "AUTHORS.txt", "COPYING.txt", "README_ar.txt")
DEFAULT_OUTPUT = common.PROJECT_ROOT / "thesis" / "dictionaries" / "general_arabic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the pinned general Arabic Hunspell dictionary.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "chandra-ocr-thesis/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty file from {url}")
        temporary.replace(destination)
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {url}: {exc}") from exc


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else common.PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = output_dir / filename
        if destination.is_file() and not args.force:
            print(f"SKIP {destination.relative_to(common.PROJECT_ROOT)}")
            continue
        print(f"DOWNLOAD {filename}", flush=True)
        download_file(f"{UPSTREAM_ROOT}/{filename}", destination)

    metadata = {
        "source": "https://github.com/LibreOffice/dictionaries/tree/master/ar",
        "commit": UPSTREAM_COMMIT,
        "files": list(FILES),
    }
    (output_dir / "source.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Dictionary ready: {output_dir}")


if __name__ == "__main__":
    main()
