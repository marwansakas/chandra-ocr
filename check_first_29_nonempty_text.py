#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path('/home/sakas/chandra-ocr/thesis/labels/surya_line_ground_truth')
FIRST_N = 29


def main() -> None:
    subdirs = sorted(
        d for d in ROOT.iterdir() if d.is_dir() and not d.name.startswith('_')
    )

    selected = subdirs[:FIRST_N]
    total_lines = 0
    non_empty_text_lines = 0

    for folder in selected:
        lines_path = folder / 'lines.jsonl'
        if not lines_path.exists():
            continue

        with lines_path.open('r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get('text', '')
                if text != '':
                    non_empty_text_lines += 1

    print(f'Checked {len(selected)} directories from {ROOT}')
    print(f'Total lines scanned: {total_lines}')
    print(f'Lines with non-empty text: {non_empty_text_lines}')


if __name__ == '__main__':
    main()
