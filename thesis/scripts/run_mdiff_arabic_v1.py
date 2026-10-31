"""Native Windows entry point; see thesis/MDIFF_ARABIC_V1.md."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['Setup','Check','Smoke','Overfit','Train','Validation','ValidationStats','Test'],required=True)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--run-name',required=True)
    p.add_argument('--peer-run-name',default='')
    p.add_argument('--arm',choices=['mdiff','ar'],default='mdiff')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--readiness-protocol',choices=['legacy_v1','geometry_matched_v1'],default='legacy_v1')
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--patience',type=int,default=30)
    p.add_argument('--overfit-updates',type=int,default=1200)
    p.add_argument('--overfit-seconds',type=int,default=900)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='Print parsed routing; no imports, files, model allocation or execution')
    a=p.parse_args(argv)
    import re
    for name in [a.run_name]+([a.peer_run_name] if a.peer_run_name else []):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,39}',name): p.error('Run names must be 2..40 safe ASCII characters')
    if a.peer_run_name==a.run_name: p.error('Primary arms need distinct run names')
    if not 1<=a.overfit_updates<=1200 or not 1<=a.overfit_seconds<=900: p.error('Overfit strict budget: updates1..1200, seconds1..900')
    if a.resume and a.action!='Train': p.error('--resume requires --action Train')
    a.project=a.project.resolve();a.repo=a.project/'.codex-pretrained-selection';a.drive=a.project/'Diffusion/drive_all'
    a.runner=a.project/('Diffusion/run_mdiff_arabic_v1_1.ps1' if a.readiness_protocol=='geometry_matched_v1' else 'Diffusion/run_mdiff_arabic_v1.ps1')
    return a


def main(argv=None):
    args=parse_args(argv)
    if args.dry_run:
        print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},sort_keys=True))
        return
    from thesis.mdiff_arabic_v1 import experiment as e
    if args.action=='Setup': e.setup(args);return
    ctx=e.context(args)
    actions=dict(Check=e.check,Smoke=e.smoke,Overfit=e.overfit,Train=e.train,Validation=e.validation,
                 ValidationStats=e.stats,Test=e.test)
    try:
        actions[args.action](ctx)
    except e.ReadinessFailure as error:
        print(str(error),file=sys.stderr)
        raise SystemExit(2)


if __name__=='__main__':
    main()
