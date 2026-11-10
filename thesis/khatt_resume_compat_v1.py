"""Audited compatibility for one added, unused OpenCV distribution.

This module stays outside khatt_ablation_v1: existing executable snapshots,
binding hashes, optimizer states and model selections remain immutable.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys

from thesis.khatt_ablation_v1 import workflow as w
from thesis.mdiff_arabic_v1.artifacts import digest, file_hash, read, write
from thesis.mdiff_arabic_v1.experiment import stamp

ALLOWED_ADDITIONS = {'opencv-python': '5.0.0.93'}


def differences(expected, actual, prefix='binding'):
    if isinstance(expected, dict) and isinstance(actual, dict):
        paths=[]
        for key in sorted(expected.keys() | actual.keys()):
            path=prefix+'.'+key
            if key not in expected or key not in actual:paths.append(path)
            else:paths.extend(differences(expected[key],actual[key],path))
        return paths
    return [] if expected==actual else [prefix]


def compatible_additions(expected, actual):
    """Accept only the known additive package; everything else remains exact."""
    old_packages=expected['environment']['packages']
    actual_packages=actual['environment']['packages']
    added={k:v for k,v in actual_packages.items() if k not in old_packages}
    for name,version in added.items():
        if ALLOWED_ADDITIONS.get(name)!=version:
            raise ValueError(f'Unreviewed added package: {name}=={version}')
    projected=copy.deepcopy(actual)
    for name in added:del projected['environment']['packages'][name]
    changed=differences(expected,projected)
    if changed:
        raise ValueError('Resume compatibility rejected changes: '+', '.join(changed))
    return added


def verify_opencv_unused(source_paths):
    if any(name=='cv2' or name.startswith('cv2.') for name in sys.modules):
        raise ValueError('OpenCV is loaded; this unused-package compatibility exception does not apply')
    for filename in source_paths:
        path=Path(filename)
        if path.suffix!='.py':continue
        tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):names=[node.module or '']
            elif isinstance(node,ast.Constant) and isinstance(node.value,str):names=[node.value]
            if any(name=='cv2' or name.startswith('cv2.') for name in names):
                raise ValueError('Bound source references OpenCV: '+str(path))


def compatibility_context(args):
    # Original context still verifies CUDA/BF16, data/manifests, pretrained
    # assets, every original source, and the frozen comparison artifacts.
    ctx=w.context(args)
    saved=read(args.suite/'check.json')
    if digest(saved['binding'])!=saved['binding_sha256']:
        raise ValueError('Original check binding digest is invalid')
    added=compatible_additions(saved['binding'],ctx['binding'])
    if added:verify_opencv_unused(saved['binding']['sources'])
    helper_sources=[Path(__file__),args.repo/'thesis/scripts/resume_khatt_ablation_v1.py',
        args.repo/'tests/test_khatt_resume_compat_v1.py',args.project/'Diffusion/resume_khatt_ablation_v1.ps1']
    evidence=dict(version='khatt_resume_compat_v1',original_check_sha256=file_hash(args.suite/'check.json'),
        original_binding_sha256=saved['binding_sha256'],observed_binding=ctx['binding'],
        observed_binding_sha256=ctx['binding_sha256'],reviewed_added_packages=added,
        helper_sources={str(p.resolve()):file_hash(p) for p in helper_sources},
        policy='Only added opencv-python==5.0.0.93; all original sources, data, recipe, environment values and package versions unchanged; cv2 not imported or referenced by bound sources')
    # Retain the observed environment as a separate, hash-bound audit record.
    # Preserve the original identity for existing checkpoint/run comparisons.
    audit=args.suite/'environment_compatibility'/digest(evidence)
    audit.mkdir(parents=True,exist_ok=True)
    if (audit/'compatibility.json').exists():
        if read(audit/'compatibility.json')!=evidence:raise ValueError('Compatibility audit mismatch')
    else:
        snapshot=audit/'source';snapshot.mkdir(exist_ok=True)
        for i,(name,sha) in enumerate(evidence['helper_sources'].items()):
            dest=snapshot/f'{i:02d}_{Path(name).name}'
            dest.write_bytes(Path(name).read_bytes())
            if file_hash(dest)!=sha:raise ValueError('Compatibility helper snapshot mismatch')
        write(audit/'compatibility.json',evidence)
    ctx['observed_binding']=ctx['binding']
    ctx['binding']=saved['binding'];ctx['binding_sha256']=saved['binding_sha256']
    w.check(ctx)
    print('Reviewed unused package additions:',added or 'none',flush=True)
    print('Environment compatibility audit:',audit/'compatibility.json',flush=True)
    return ctx,audit


def verify_progress(ctx):
    """Hash-check saved progress without loading a model or running inference."""
    progress={}
    for name in w.TRIALS:
        folder=ctx['args'].suite/'runs'/name
        record=folder/'run.json'
        if not record.exists():
            progress[name]=dict(status='not_started');continue
        if read(record)['binding_sha256']!=ctx['binding_sha256']:
            raise ValueError('Saved run binding mismatch: '+name)
        if (folder/'complete.json').exists():
            _,best=w.selected(ctx,name)
            progress[name]=dict(status='completed',epoch=read(folder/'complete.json')['epoch'],selected_epoch=best['epoch'])
        elif (folder/'last.json').exists():
            last=read(folder/'last.json')
            if file_hash(folder/last['path'])!=last['sha256']:
                raise ValueError('Resume checkpoint hash mismatch: '+name)
            progress[name]=dict(status='interrupted',completed_epoch=last['epoch'],resume_epoch=last['epoch']+1)
        else:progress[name]=dict(status='no_completed_epoch',resume_epoch=1)
        print(name,progress[name],flush=True)
    return progress


def run(args):
    if args.action=='Prepare':
        raise ValueError('This compatibility launcher requires an already prepared, checked suite')
    with w.run_lock(args.suite):
        ctx,audit=compatibility_context(args)
        progress=verify_progress(ctx)
        from thesis import khatt_subset_eval_v1 as subset
        if getattr(args,'skip_10_percent',False):subset.prepare(ctx)
        trial_decision=subset.decision(ctx)
        write(audit/(args.action+'-'+stamp()+'.json'),dict(action=args.action,progress_before=progress,
            compatibility_sha256=file_hash(audit/'compatibility.json'),trial_decision=trial_decision))
        if args.action=='Train':
            if trial_decision and args.trial in trial_decision['abandoned_trials']:
                raise ValueError('This trial was abandoned by the user and will not be resumed')
            w.train_one(ctx,args.trial)
        elif args.action=='TrainAll':
            for name in w.TRIALS:
                if trial_decision and name in trial_decision['abandoned_trials']:
                    print('Skipping abandoned trial:',name,flush=True);continue
                w.train_one(ctx,name)
        elif args.action in ('TestAll','Stats'):
            if trial_decision:subset.evaluate(ctx,args.action)
            elif args.action=='TestAll':w.test_all(ctx)
            else:w.stats(args)
