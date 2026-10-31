"""Prepare/train/calibrate three fusion variants; historical Test is exactly 752 lines."""
from pathlib import Path
import argparse
import json
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
ACTIONS=('Check','Prepare','Calibrate','TrainShared','PrepareShared','CalibrateShared','Test','Stats','All')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=ACTIONS,required=True)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--run-name',default='ar_md_fusion2_s54101')
    p.add_argument('--ar-run-name',default='ar1b_s54101')
    p.add_argument('--mdiff-run-name',default='ma1b_s54101')
    p.add_argument('--test-cache-name',default='mdar_20261031_x01')
    p.add_argument('--previous-hybrid-name',default='ar_md_h1_s54101')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    for name in (a.run_name,a.ar_run_name,a.mdiff_run_name,a.test_cache_name,a.previous_hybrid_name):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,39}',name):p.error('Use short safe run/cache names')
    if not 1<=a.patience<=a.epochs<=200:p.error('Require 1 <= patience <= epochs <= 200')
    if a.resume and a.action not in ('All','TrainShared'):p.error('--resume is only valid for All or TrainShared')
    a.project=a.project.resolve();a.repo=a.project/'.codex-pretrained-selection';a.drive=a.project/'Diffusion/drive_all'
    a.runner=a.project/'Diffusion/run_ar_mdiff_fusion_v2.ps1'
    return a


def main():
    args=parse_args()
    if args.dry_run:
        print(json.dumps({**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'test_lines':752,'methods':['dual_rescore','span_merge','shared_encoder']},sort_keys=True));return
    from thesis.ar_mdiff_fusion_v2 import workflow as w
    print('Verifying selected primary checkpoints and immutable inputs...',flush=True)
    ctx=w.context(args)
    actions=dict(Check=w.check,Prepare=w.prepare_frozen,Calibrate=w.calibrate_frozen,
        TrainShared=w.train_shared,PrepareShared=w.prepare_shared,CalibrateShared=w.calibrate_shared,
        Test=w.historical_test,Stats=w.stats)
    with w.run_lock(ctx['run']):
        if args.action=='All':
            for name in ('Check','Prepare','Calibrate','TrainShared','PrepareShared','CalibrateShared','Test'):
                print('\n=== '+name+' ===',flush=True)
                # Test references remain inaccessible to earlier steps even in All.
                ctx['args'].action=name
                if name=='TrainShared' and args.resume and not (ctx['run']/'shared/run.json').exists():
                    ctx['args'].resume=False
                actions[name](ctx)
        else:actions[args.action](ctx)


if __name__=='__main__':main()
