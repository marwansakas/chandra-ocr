"""Separate AR-default line-selector workflow; base model bindings stay untouched."""
from pathlib import Path
import argparse
import json
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['Check','Fit','Validation','ValidationStats','Test'],required=True)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--run-name',required=True)
    p.add_argument('--ar-run-name',default='ar1b_s54101')
    p.add_argument('--mdiff-run-name',default='ma1b_s54101')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--test-cache-name',default='mdar_20261031_x01')
    p.add_argument('--allow-exploratory',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    for name in [a.run_name,a.ar_run_name,a.mdiff_run_name,a.test_cache_name]:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,39}',name):p.error('Use short safe run/cache names')
    if a.allow_exploratory and a.action!='Test':p.error('--allow-exploratory is only valid for Test')
    a.project=a.project.resolve();a.repo=a.project/'.codex-pretrained-selection';a.drive=a.project/'Diffusion/drive_all'
    a.runner=a.project/'Diffusion/run_ar_mdiff_hybrid_v1.ps1'
    return a


def main():
    args=parse_args()
    if args.dry_run:
        print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},sort_keys=True));return
    from thesis.ar_mdiff_hybrid_v1 import workflow as w
    print('Verifying immutable primary checkpoints, inputs and hybrid bindings...',flush=True)
    ctx=w.context(args)
    dict(Check=w.check,Fit=w.fit,Validation=w.validation,ValidationStats=w.stats,Test=w.test)[args.action](ctx)


if __name__=='__main__':main()
