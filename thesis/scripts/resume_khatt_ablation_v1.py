"""Resume the original KHATT suite with a reviewed unused-package addition."""
from pathlib import Path
import json
import argparse
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from thesis.scripts.run_khatt_ablation_v1 import parse_args as original_args


def parse_args(argv=None):
    extra=argparse.ArgumentParser(add_help=False)
    extra.add_argument('--skip-10-percent',action='store_true')
    flags,remaining=extra.parse_known_args(argv)
    args=original_args(remaining)
    args.skip_10_percent=flags.skip_10_percent
    if args.skip_10_percent and args.action not in ('Check','TestAll','Stats'):
        extra.error('--skip-10-percent is only valid for Check, TestAll or Stats')
    return args


def main():
    args=parse_args()
    if args.dry_run:
        print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},sort_keys=True))
        return
    from thesis.khatt_resume_compat_v1 import run
    run(args)


if __name__=='__main__':main()
