"""Prepare or train five AR ablations. Evaluation requires an explicit TestAll."""
from pathlib import Path
import argparse
import json
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['Prepare','Check','Train','TrainAll','TestAll','Stats'],required=True)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--suite-name',default='khatt_ar_v1fix1_s54101')
    p.add_argument('--trial',choices=['khatt100_only','khatt100_ours','khatt50_ours','khatt30_ours','khatt10_ours'])
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,39}',a.suite_name):p.error('Use a short safe SuiteName')
    if a.action=='Train' and not a.trial:p.error('Train requires --trial')
    if a.resume and a.action not in ('Train','TrainAll'):p.error('Resume is only valid for training')
    if a.epochs!=100 or a.patience!=10:p.error('This experiment is fixed to 100 maximum epochs and patience 10')
    a.project=a.project.resolve();a.repo=a.project/'.codex-pretrained-selection'
    a.drive=a.project/'Diffusion/drive_all';a.suite=a.drive/'khatt_ablations'/a.suite_name
    a.runner=a.project/'Diffusion/run_khatt_ablation_v1.ps1'
    return a


def main():
    args=parse_args()
    if args.dry_run:
        print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},sort_keys=True));return
    from thesis.khatt_ablation_v1 import data,workflow as w
    with w.run_lock(args.suite):
        if args.action=='Prepare':data.prepare(args);return
        if args.action=='Stats':w.stats(args);return
        ctx=w.context(args);w.check(ctx)
        if args.action=='Train':w.train_one(ctx,args.trial)
        elif args.action=='TrainAll':
            for name in data.TRIALS:w.train_one(ctx,name)
        elif args.action=='TestAll':w.test_all(ctx)


if __name__=='__main__':main()
