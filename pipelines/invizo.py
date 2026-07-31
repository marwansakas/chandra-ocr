"""Run a downloaded Invizo inference command and normalize paragraph outputs.

Invizo is an external application, so its exact launcher depends on the local
checkout/release. The command template must contain ``{input}`` and ``{output}``.
Example:

  python pipelines/invizo.py --command \
    "python /opt/invizo/infer.py --input {input} --output {output}"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "split_papers")
    parser.add_argument("--output", type=Path, default=ROOT / "thesis" / "invizo" / "outputs")
    parser.add_argument(
        "--command",
        required=True,
        help="Invizo command template containing {input} and {output}.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def normalize_text(output_dir: Path) -> str:
    text_files = sorted((*output_dir.rglob("*.txt"), *output_dir.rglob("*.md")))
    if len(text_files) == 1:
        return text_files[0].read_text(encoding="utf-8", errors="replace").strip()
    json_files = sorted(output_dir.rglob("*.json"))
    if len(json_files) == 1:
        payload = json.loads(json_files[0].read_text(encoding="utf-8"))
        for key in ("text", "paragraph", "transcription", "prediction"):
            if isinstance(payload, dict) and isinstance(payload.get(key), str):
                return payload[key].strip()
    raise RuntimeError("Invizo output must contain one text/Markdown file or one JSON text field.")


def main() -> None:
    args = parse_args()
    if "{input}" not in args.command or "{output}" not in args.command:
        raise SystemExit("--command must contain both {input} and {output}")
    sources = sorted(
        path for path in args.input.rglob("*")
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    manifest: list[dict[str, str]] = []
    failures = 0
    for index, source in enumerate(sources, 1):
        relative = source.relative_to(args.input)
        destination = args.output / relative.parent / source.stem
        prediction = destination / "prediction.txt"
        if prediction.exists() and not args.force:
            text = prediction.read_text(encoding="utf-8")
        else:
            destination.mkdir(parents=True, exist_ok=True)
            command = args.command.format(input=str(source), output=str(destination))
            try:
                subprocess.run(shlex.split(command, posix=os.name != "nt"), check=True)
                text = normalize_text(destination)
                prediction.write_text(text + "\n", encoding="utf-8")
            except Exception as exc:
                failures += 1
                print(f"[{index}/{len(sources)}] ERROR {source}: {exc}")
                if not args.continue_on_error:
                    raise
                continue
        manifest.append({"input": str(relative), "prediction": str(prediction.relative_to(ROOT))})

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("input", "prediction"))
        writer.writeheader()
        writer.writerows(manifest)
    print(json.dumps({"inputs": len(sources), "outputs": len(manifest), "failed": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
