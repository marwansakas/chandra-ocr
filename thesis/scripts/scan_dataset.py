"""
Scan split_papers/ and print a human-readable inventory:
counts by place, by grade, by notebook, plus parsing anomalies.

Read-only. Writes nothing. Safe to run anytime.

    python thesis/scripts/scan_dataset.py
"""

from __future__ import annotations

from collections import Counter, defaultdict

import common


def main() -> None:
    cfg = common.load_config()
    raw_root = common.resolve(cfg["raw_data_dir"])

    if not raw_root.exists():
        raise SystemExit(f"Raw data dir not found: {raw_root}")

    records = [common.parse_pdf(p, raw_root) for p in common.iter_pdfs(raw_root)]
    total = len(records)

    by_place = Counter(r.place for r in records)
    by_grade = Counter(
        (r.grade if r.grade is not None else "unknown") for r in records
    )
    by_place_grade = Counter((r.place, r.grade_folder) for r in records)
    notebooks = defaultdict(set)
    for r in records:
        notebooks[(r.place, r.grade_folder)].add(r.notebook_id)

    unparsed = [r for r in records if not r.filename_parsed]
    no_gender = [r for r in records if r.gender is None]
    grade_from_folder = [r for r in records if r.grade_source == "folder"]
    grade_unknown = [r for r in records if r.grade is None]

    print("=" * 64)
    print(f"DATASET INVENTORY  ({raw_root})")
    print("=" * 64)
    print(f"Total PDF pages: {total}")
    print(f"Distinct notebooks (== students, by assumption): "
          f"{len({r.notebook_id for r in records})}")

    print("\n-- Pages by place --")
    for place, n in by_place.most_common():
        print(f"  {place:<14} {n}")

    print("\n-- Pages by grade --")
    for grade, n in sorted(by_grade.items(), key=lambda x: str(x[0])):
        print(f"  grade {grade!s:<8} {n}")

    print("\n-- Pages by place x grade-folder --")
    for (place, gf), n in sorted(by_place_grade.items()):
        nb = len(notebooks[(place, gf)])
        print(f"  {place:<12} | {gf:<14} | pages={n:<5} notebooks={nb}")

    print("\n-- Parsing / metadata anomalies --")
    print(f"  filenames not matching pattern : {len(unparsed)}")
    print(f"  pages with no gender token     : {len(no_gender)}")
    print(f"  grade inferred from folder     : {len(grade_from_folder)}")
    print(f"  pages with unknown grade       : {len(grade_unknown)}")
    if unparsed:
        print("  examples of unparsed names:")
        for r in unparsed[:5]:
            print(f"    {r.image_path}")

    print("\nNext: python thesis/scripts/build_metadata.py")


if __name__ == "__main__":
    main()
