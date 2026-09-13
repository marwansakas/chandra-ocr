"""Audit the expanded Drive staging folder and freeze a new 70/10/20 crop split."""
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis.diffusion.data import audit_dataset, freeze_line_split, json_bytes, load_frozen, normalize_exclusion_ids, file_hash


def inventory_sample_ids(path: Path) -> set[str]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError("Verified inventory must be a JSON list from rclone lsjson")
    result = set()
    for row in rows:
        value = row.get("Path") if isinstance(row, dict) else None
        if not isinstance(value, str) or value.count("/") != 1:
            raise ValueError(f"Malformed verified inventory path: {value!r}")
        sample_id, name = value.split("/", 1)
        if name in ("image.png", "ground_truth.txt"):
            result.add(sample_id)
    if not result:
        raise ValueError("Verified inventory has no paired-crop paths")
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--split-root', type=Path, required=True)
    p.add_argument('--stats-dir', type=Path, required=True)
    p.add_argument('--exclusion-list', type=Path, help='Recorded standalone-≠ IDs; exclusions are applied without deleting local folders')
    p.add_argument('--verified-inventory', type=Path, help='Current rclone lsjson inventory; required for a new freeze')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    exclusions = normalize_exclusion_ids(args.exclusion_list.read_text(encoding="utf-8").splitlines()) if args.exclusion_list else []
    allowed_ids = inventory_sample_ids(args.verified_inventory) if args.verified_inventory else None
    if not args.split_root.exists() and allowed_ids is None:
        raise ValueError("A new immutable split requires --verified-inventory to reject stale local folders")
    audit = audit_dataset(args.dataset_root.resolve(), excluded_sample_ids=exclusions, allowed_sample_ids=allowed_ids)
    if audit['issues']:
        raise ValueError(json.dumps(audit['issues'][:20], ensure_ascii=False))
    lock = freeze_line_split(audit, args.split_root, train_ratio=.7, validation_ratio=.1, seed=args.seed)
    parts, _, tok = load_frozen(args.split_root, args.dataset_root)
    widths = [math.ceil(max(1, r['width'] * 64 / r['height']) / 8) * 8 for r in parts['train']]
    report = {'dataset_fingerprint': lock['dataset_fingerprint'], 'counts': lock['counts'],
              'fractions': {k: len(v)/len(audit['records']) for k,v in parts.items()},
              'seed': args.seed, 'capacity_including_eos': lock['capacity'],
              'vocabulary_size_with_specials': len(tok), 'usable_labeled_lines': len(audit['records']),
              'training_max_width_at_height_64_rounded_to_patch_8': max(widths),
              'multiline_crops': sum('\n' in r['text'] for r in audit['records']),
              'tatweel_crops': sum('ـ' in r['text'] for r in audit['records']),
              'included_records': len(audit['records']), 'excluded_records': audit['excluded_records'],
              'missing_exclusion_ids': audit['missing_exclusion_ids'],
              'verified_inventory_sha256': file_hash(args.verified_inventory) if args.verified_inventory else None,
              'split_audit': {key: lock[key] for key in ('cross_split_duplicate_pixel_hashes', 'cross_split_duplicate_pixel_count', 'pages_per_partition', 'pages_in_multiple_partitions', 'length_codepoints_by_partition', 'page_note')},
              'length_codepoints': audit['lengths_codepoints'],
              'note': '70% train, 10% validation, 20% final test: validation plus test are the requested 30% holdout. Validation is used only for loss/overfitting diagnostics; test remains untouched until final evaluation.'}
    args.stats_dir.mkdir(parents=True, exist_ok=False)
    (args.stats_dir/'data_stats.json').write_bytes(json_bytes(report))
    (args.stats_dir/'audit.json').write_bytes(json_bytes(audit))
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__': main()
