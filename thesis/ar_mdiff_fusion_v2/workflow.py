"""Sealed development calibration and user-invoked historical 752-line tests."""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
import os
import shutil
import time

from thesis.scripts.run_mdiff_arabic_v1 import parse_args as primary_args
from thesis.mdiff_arabic_v1 import experiment as primary
from thesis.mdiff_arabic_v1.artifacts import (
    canonical,digest,file_hash,indexed,jsonl,load_data,metrics,read,require_equal,write,write_jsonl,
)
from thesis.ar_mdiff_hybrid_v1.logic import development_roles
from .logic import RECIPE,VERSION,calibrate,evaluate,prediction_only,select
from . import inference,shared


def source_files(args):
    paths=list((args.repo/'thesis/ar_mdiff_fusion_v2').glob('*.py'))
    paths.extend([args.repo/'thesis/scripts/run_ar_mdiff_fusion_v2.py',
        args.repo/'tests/test_ar_mdiff_fusion_v2.py',args.repo/'thesis/ar_mdiff_hybrid_v1/logic.py',args.runner])
    return {str(p.resolve()):file_hash(p) for p in sorted(paths)}


def context(args):
    pa=primary_args(['--action','ValidationStats','--project',str(args.project),
        '--run-name',args.mdiff_run_name,'--peer-run-name',args.ar_run_name,'--arm','mdiff',
        '--device',args.device,'--readiness-protocol','geometry_matched_v1'])
    p=primary.context(pa);mb=primary.selected(p);peer=primary.peer_context(p);ab=primary.selected(peer)
    if p['evidence']['coverage_issues']:raise ValueError('Canonical TRAIN/validation coverage issues')
    rows=p['parts']['validation'];roles=development_roles(rows)
    ar=indexed(jsonl(peer['run']/ab['predictions']),rows);md=indexed(jsonl(p['run']/mb['predictions']),rows)
    cache=args.drive/'exploratory'/args.test_cache_name
    inputs={str(cache/name):file_hash(cache/name) for name in ('report.json','request.json','predictions.jsonl')}
    old=args.drive/'hybrids'/args.previous_hybrid_name/'historical_test'
    if (old/'report.json').exists():
        inputs.update({str(old/name):file_hash(old/name) for name in ('report.json','predictions.jsonl')})
    sr=dict(RECIPE['shared'],maximum_epochs=args.epochs,patience=args.patience)
    binding=dict(version=VERSION,recipe=RECIPE,shared_recipe=sr,run_name=args.run_name,
        parent_names=dict(ar=args.ar_run_name,mdiff=args.mdiff_run_name),
        parent_bindings=dict(ar=peer['binding_sha256'],mdiff=p['binding_sha256']),
        selected=dict(ar=ab,mdiff=mb),source_sha256=source_files(args),
        split_hashes=p['evidence']['split_hashes'],split_lock_sha256=p['evidence']['split_lock_sha256'],
        environment=p['binding']['environment'],roles=roles,historical_input_sha256=inputs,
        authorization='User requested all three methods tested on exactly the historical 752 lines; commands only, no automatic execution')
    return dict(args=args,parent=p,peer=peer,parts=p['parts'],tokenizer=p['tokenizer'],image_config=p['image_config'],
        ar_best=ab,md_best=mb,ar_path=peer['run']/ab['checkpoint'],md_path=p['run']/mb['checkpoint'],
        ar=ar,mdiff=md,roles=roles,cache=cache,old_test=old,binding=binding,binding_sha256=digest(binding),
        run=args.drive/'hybrids'/args.run_name)


def once(path,value):
    if path.exists():require_equal(read(path),value,'existing '+str(path))
    else:write(path,value)


def once_jsonl(path,rows):
    rows=list(rows)
    if path.exists():require_equal(jsonl(path),rows,'existing '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        pending=path.with_suffix(path.suffix+'.pending')
        with pending.open('wb') as out:
            for row in rows:out.write(canonical(row))
        pending.replace(path)


@contextmanager
def run_lock(folder):
    """OS releases this lock on Ctrl+C/process exit; no stale PID cleanup."""
    folder.mkdir(parents=True,exist_ok=True)
    handle=(folder/'.active.lock').open('a+b');locked=False
    try:
        if handle.seek(0,2)==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked=True
        except OSError as error:raise RuntimeError('Another process is using this fusion RunName. Wait for it or stop it before resuming.') from error
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        handle.close()


def check(ctx):
    path=ctx['run']/'check.json'
    if path.exists():verify_check(ctx);print('Existing fusion Check matches:',path);return
    ctx['run'].mkdir(parents=True,exist_ok=True)
    snapshot=ctx['run']/'source';snapshot.mkdir(exist_ok=True)
    for i,(name,sha) in enumerate(ctx['binding']['source_sha256'].items()):
        target=snapshot/f'{i:02d}_{Path(name).name}'
        if not target.exists() or file_hash(target)!=sha:
            pending=target.with_suffix(target.suffix+'.pending');shutil.copyfile(name,pending);pending.replace(target)
        require_equal(file_hash(target),sha,'source snapshot')
    once(ctx['run']/'roles.json',ctx['roles'])
    once(path,dict(binding=ctx['binding'],binding_sha256=ctx['binding_sha256'],
        roles_sha256=file_hash(ctx['run']/'roles.json'),test_used=False))
    print('Check passed. Primary checkpoints and sources preserved.')
    print('Frozen development roles:',ctx['roles']['line_counts'],'| Test action is fixed to 752 historical lines.')
    print('Shared recipe:',ctx['binding']['shared_recipe'])
    print('Output:',ctx['run'])


def verify_check(ctx):
    checked=read(ctx['run']/'check.json')
    require_equal(checked['binding'],ctx['binding'],'fusion experiment binding; use a new RunName for code/recipe changes')
    require_equal(checked['binding_sha256'],ctx['binding_sha256'],'fusion binding digest')
    require_equal(checked['roles_sha256'],file_hash(ctx['run']/'roles.json'),'development role manifest')
    for i,(name,sha) in enumerate(ctx['binding']['source_sha256'].items()):
        require_equal(file_hash(ctx['run']/'source'/f'{i:02d}_{Path(name).name}'),sha,'sealed source copy')
    return checked


def test_inputs(ctx):
    if ctx['args'].action!='Test':raise ValueError('752-line references may be opened only by the explicit Test action')
    cache=ctx['cache'];report=read(cache/'report.json')
    require_equal(file_hash(cache/'request.json'),report['request_sha256'],'historical cache request')
    require_equal(file_hash(cache/'predictions.jsonl'),report['predictions_sha256'],'historical cache predictions')
    require_equal(report['split_hashes'],ctx['binding']['split_hashes'],'historical cache splits')
    for name,best in [('transformer_arabic_control',ctx['ar_best']),('mdiff_arabic_raw',ctx['md_best'])]:
        require_equal(report['selected_models'][name]['sha256'],best['sha256'],'cached selected checkpoint')
    parts,_,_,evidence=load_data(ctx['args'].drive,test_authorized=True,readiness_protocol='geometry_matched_v1')
    if evidence['coverage_issues']:raise ValueError('Canonical historical coverage issues; no examples may be dropped')
    rows=parts['test']
    if len(rows)!=RECIPE['test_lines']:raise ValueError('Test must contain exactly 752 lines')
    by_id={r['sample_id']:r for r in rows};models={}
    for p in jsonl(cache/'predictions.jsonl'):
        sid=p['sample_id']
        if sid not in by_id:raise ValueError('Unexpected cached sample')
        require_equal(p['reference_literal'],by_id[sid]['text'],'cached reference binding')
        require_equal(p['image_sha256'],by_id[sid]['image_sha256'],'cached image binding')
        values=models.setdefault(p['model'],{})
        if sid in values:raise ValueError('Duplicate cached model/sample')
        values[sid]=prediction_only(p)
    for name in ('paddle','frozen_v2_2','span_hybrid_verifier_v4','transformer_arabic_control','mdiff_arabic_raw'):
        indexed(list(models[name].values()),rows)
    if str(ctx['old_test']/'report.json') in ctx['binding']['historical_input_sha256']:
        old=read(ctx['old_test']/'report.json')
        require_equal(old['test_cache_predictions_sha256'],file_hash(cache/'predictions.jsonl'),'previous hybrid input cache')
        require_equal(old['predictions_sha256'],file_hash(ctx['old_test']/'predictions.jsonl'),'previous hybrid predictions')
        models['whole_line_hybrid_v1']=indexed(jsonl(ctx['old_test']/'predictions.jsonl'),rows)
    return rows,models


def row_identity(row,ar,md):
    return dict(sample_id=row['sample_id'],image_sha256=row['image_sha256'],image_path=row['image_path'],
        ar=prediction_only(ar),mdiff=prediction_only(md) if md is not None else None)


def prepare(ctx,*,shared_model=False,historical=None):
    """Per-line atomic caches make scoring resumable without rerunning finished lines."""
    verify_check(ctx);best=None
    if shared_model:
        # Verify completion/selection before opening any historical references.
        complete=read(ctx['run']/'shared/complete.json');best=complete['best']
        require_equal(complete['binding_sha256'],ctx['binding_sha256'],'shared preparation binding')
        for name,h in [('checkpoint','sha256'),('predictions','predictions_sha256')]:
            require_equal(file_hash(ctx['run']/'shared'/best[name]),best[h],'shared preparation '+name)
    if historical is None:
        rows=ctx['parts']['validation'];ar=ctx['ar'];md=None if shared_model else ctx['mdiff'];partition='validation'
    else:
        rows,models=historical;ar=models['transformer_arabic_control'];md=None if shared_model else models['mdiff_arabic_raw'];partition='historical_752'
    tag=('shared_' if shared_model else 'frozen_')+partition
    folder=ctx['run']/'work'/tag;folder.mkdir(parents=True,exist_ok=True)
    meta=dict(binding_sha256=ctx['binding_sha256'],partition=partition,
        shared_checkpoint_sha256=best['sha256'] if best else None,
        sample_ids=[r['sample_id'] for r in rows],test_used=historical is not None)
    once(folder/'meta.json',meta);meta_sha=digest(meta)
    manifest_path=folder/'manifest.json'
    previous=read(manifest_path) if manifest_path.exists() else None
    if previous:require_equal(previous['meta_sha256'],meta_sha,'prepared cache metadata')
    records=[];files={};models_loaded=None;began=time.perf_counter()
    expected_shared=indexed(jsonl(ctx['run']/'shared'/best['predictions']),rows) if shared_model and historical is None else None
    for i,row in enumerate(rows):
        path=folder/f'{i:04d}.json';sid=row['sample_id']
        identity=digest(row_identity(row,ar[sid],md[sid] if md else None))
        if path.exists():
            if previous:require_equal(file_hash(path),previous['files'][path.name],'sealed line cache')
            envelope=read(path)
            require_equal(envelope['meta_sha256'],meta_sha,'line cache experiment')
            require_equal(envelope['input_sha256'],identity,'line cache input')
            require_equal(envelope['payload_sha256'],digest(envelope['payload']),'line cache payload')
            result=envelope['payload']
        else:
            if previous:raise ValueError('Completed cache is missing a line')
            if models_loaded is None:
                a=inference.load_primary(ctx['ar_path'],ctx['tokenizer'],'ar',ctx['args'].device)
                if shared_model:
                    d,_,_=shared.load_best(ctx)
                    # Same actual modules, not just an architectural match.
                    d.encoder=a.encoder;d.visual=a.visual
                else:d=inference.load_primary(ctx['md_path'],ctx['tokenizer'],'mdiff',ctx['args'].device)
                models_loaded=(a,d)
            # Strip reference/label fields before crossing the inference boundary.
            image_row={k:row[k] for k in ('sample_id','image_path','image_sha256','pixels_sha256','width','height') if k in row}
            result=inference.record(image_row,ar[sid],md[sid] if md else None,models_loaded,
                ctx['tokenizer'],ctx['args'].drive/'ground_truth_lines',ctx['image_config'],ctx['args'].device,shared=shared_model)
            if expected_shared is not None:
                require_equal(prediction_only(result['mdiff']),prediction_only(expected_shared[sid]),'shared generation reproduces selected validation output')
            write(path,dict(meta_sha256=meta_sha,input_sha256=identity,payload=result,payload_sha256=digest(result)))
        require_equal(result['sample_id'],sid,'prepared sample order')
        require_equal(result['encoder_calls'],1 if shared_model else 2,'image encoder call count')
        records.append(result);files[path.name]=file_hash(path)
        if (i+1)%10==0 or i+1==len(rows):
            print(f'{tag}: {i+1}/{len(rows)} lines ready | this invocation {(time.perf_counter()-began)/60:.1f} min',flush=True)
    manifest=dict(meta_sha256=meta_sha,files=files,lines=len(rows),sample_ids=meta['sample_ids'])
    once(manifest_path,manifest)
    if models_loaded is not None:del models_loaded
    return rows,records,manifest_path


def prepare_frozen(ctx):prepare(ctx)
def prepare_shared(ctx):prepare(ctx,shared_model=True)


def load_records(ctx,tag):
    folder=ctx['run']/'work'/tag;manifest=read(folder/'manifest.json');meta=read(folder/'meta.json')
    require_equal(meta['binding_sha256'],ctx['binding_sha256'],'cached record binding')
    require_equal(manifest['meta_sha256'],digest(meta),'cached metadata hash')
    records=[]
    for name,sha in sorted(manifest['files'].items()):
        require_equal(file_hash(folder/name),sha,'cached score file')
        item=read(folder/name)
        require_equal(item['meta_sha256'],digest(meta),'cached line metadata')
        require_equal(item['payload_sha256'],digest(item['payload']),'cached score payload')
        records.append(item['payload'])
    require_equal([r['sample_id'] for r in records],meta['sample_ids'],'cached order and membership')
    return records,folder/'manifest.json'


def publish(folder,ctx,report,predictions,dependencies):
    folder.mkdir(parents=True,exist_ok=True)
    once_jsonl(folder/'predictions.jsonl',predictions)
    report=dict(report,predictions_sha256=file_hash(folder/'predictions.jsonl'),binding_sha256=ctx['binding_sha256'])
    once(folder/'report.json',report)
    seal=dict(binding_sha256=ctx['binding_sha256'],dependencies=dependencies,
        files={name:file_hash(folder/name) for name in ('report.json','predictions.jsonl')})
    once(folder/'seal.json',seal)
    text=render(report,folder/'report.json');(folder/'report.txt').write_text(text,encoding='utf-8')
    print(text)
    return report


def verify_published(folder,ctx):
    seal=read(folder/'seal.json')
    require_equal(seal['binding_sha256'],ctx['binding_sha256'],'published binding')
    for name,sha in seal['files'].items():require_equal(file_hash(folder/name),sha,'published '+name)
    for name,sha in seal['dependencies'].items():require_equal(file_hash(Path(name)),sha,'published dependency')
    report=read(folder/'report.json')
    require_equal(report['binding_sha256'],ctx['binding_sha256'],'published report binding')
    return report


def calibrate_methods(ctx,*,shared_model=False):
    verify_check(ctx)
    tag='shared' if shared_model else 'frozen';folder=ctx['run']/'calibration'/tag
    if (folder/'seal.json').exists():
        report=verify_published(folder,ctx);print(render(report,folder/'report.json'));return
    records,manifest=load_records(ctx,tag+'_validation');rows=ctx['parts']['validation']
    indexed(records,rows)
    calibration_rows=[r for r in rows if ctx['roles']['sample_roles'][r['sample_id']]!='audit']
    audit_rows=[r for r in rows if ctx['roles']['sample_roles'][r['sample_id']]=='audit']
    methods=['shared_encoder'] if shared_model else ['dual_rescore','span_merge']
    selections={};results={};predictions=[];models=dict(transformer_arabic_control=metrics(rows,ctx['ar']))
    if shared_model:
        raw={r['sample_id']:r['mdiff'] for r in records};models['shared_mdiff_raw']=metrics(rows,raw)
    else:models['mdiff_arabic_raw']=metrics(rows,ctx['mdiff'])
    for method in methods:
        search_method='dual_rescore' if method=='shared_encoder' else method
        selected=calibrate(calibration_rows,records,search_method)
        values=select(records,search_method,selected['config']);selections[method]=selected
        results[method]=dict(full_development=evaluate(rows,ctx['ar'],values),
            audit=evaluate(audit_rows,ctx['ar'],{r['sample_id']:values[r['sample_id']] for r in audit_rows}))
        models[method]=results[method]['full_development']['metrics']
        predictions.extend(dict(p,model=method) for p in values.values())
    folder.mkdir(parents=True,exist_ok=True);once(folder/'selection.json',selections)
    report=dict(version=VERSION,partition='validation',test_used=False,models=models,results=results,
        configs={name:s['config'] for name,s in selections.items()},calibration_lines=len(calibration_rows),audit_lines=len(audit_rows),
        calibration_sample_ids=[r['sample_id'] for r in calibration_rows],audit_sample_ids=[r['sample_id'] for r in audit_rows],
        limitation='All 150 labels previously used for base checkpoint selection/development. Audit excludes current score-grid tuning only; not untouched data.',
        promotion_performed=False)
    deps={str(manifest):file_hash(manifest),str(folder/'selection.json'):file_hash(folder/'selection.json')}
    if shared_model:deps[str(ctx['run']/'shared/complete.json')]=file_hash(ctx['run']/'shared/complete.json')
    publish(folder,ctx,report,predictions,deps)


def calibrate_frozen(ctx):calibrate_methods(ctx)
def calibrate_shared(ctx):calibrate_methods(ctx,shared_model=True)


def train_shared(ctx):
    verify_check(ctx);shared.train(ctx)


def historical_test(ctx):
    verify_check(ctx)
    # Freeze and verify all settings before opening the historical references.
    frozen=verify_published(ctx['run']/'calibration/frozen',ctx)
    adapted=verify_published(ctx['run']/'calibration/shared',ctx)
    _,best,_=shared.load_best(ctx)
    folder=ctx['run']/'historical_752'
    if (folder/'seal.json').exists():
        report=verify_published(folder,ctx);print(render(report,folder/'report.json'));return
    print('Exploratory comparison on exactly 752 historical development-used lines; settings frozen; no promotion.',flush=True)
    rows,baselines=test_inputs(ctx)
    _,records,manifest=prepare(ctx,historical=(rows,baselines))
    _,shared_records,shared_manifest=prepare(ctx,shared_model=True,historical=(rows,baselines))
    ar=baselines['transformer_arabic_control'];values={};results={}
    for method in ('dual_rescore','span_merge'):
        values[method]=select(records,method,frozen['configs'][method]);results[method]=evaluate(rows,ar,values[method])
    values['shared_mdiff_raw']={r['sample_id']:r['mdiff'] for r in shared_records}
    values['shared_encoder']=select(shared_records,'dual_rescore',adapted['configs']['shared_encoder'])
    results['shared_encoder']=evaluate(rows,ar,values['shared_encoder'])
    models={name:metrics(rows,p) for name,p in baselines.items()}
    models.update({name:metrics(rows,p) for name,p in values.items()})
    if any(m['lines']!=752 or m['expected_lines']!=752 for m in models.values()):raise ValueError('All comparison rows must cover exactly 752 lines')
    report=dict(version=VERSION,partition='historical_752',test_used=True,
        benchmark_status='historical development-used exploratory comparison; no fresh generalization claim',
        models=models,results=results,configs={**frozen['configs'],**adapted['configs']},
        selected_shared_epoch=best['epoch'],shared_checkpoint_sha256=best['sha256'],
        development_preservation_checks={name:result['full_development']['passes_preservation_check']
            for validation in (frozen,adapted) for name,result in validation['results'].items()},
        timing=dict(frozen_pool_seconds=sum(r['seconds'] for r in records),shared_pool_seconds=sum(r['seconds'] for r in shared_records),
            peak_cuda_bytes=max((r['peak_cuda_bytes'] or 0 for r in records+shared_records),default=0),
            note='Frozen pool jointly builds/scores both method-1 and method-2 candidates; costs are amortized, not independent per-method latency.'),
        promotion_performed=False,original_gates_modified=False)
    deps={str(p):file_hash(p) for p in (manifest,shared_manifest,ctx['run']/'calibration/frozen/seal.json',ctx['run']/'calibration/shared/seal.json',ctx['run']/'shared/complete.json')}
    predictions=[dict(p,model=name) for name,preds in values.items() for p in preds.values()]
    publish(folder,ctx,report,predictions,deps)


def render(report,path):
    text=('HISTORICAL DEVELOPMENT-USED COMPARISON: exactly 752 lines; no promotion.\n' if report['test_used'] else
        'DEVELOPMENT VALIDATION: 120 calibration + 30 audit; base checkpoints already used these labels.\n')
    text+='Model                         Lines   CER %   WER %  Exact %   Sub   Ins   Del\n'
    for name,m in report['models'].items():
        text+=f'{name:<29} {m["lines"]:5d} {100*m["cer"]:7.2f} {100*m["wer"]:7.2f} {100*m["exact_line_accuracy"]:8.2f} {m["substitutions"]:5d} {m["insertions"]:5d} {m["deletions"]:5d}\n'
    for name,result in report['results'].items():
        result=result if report['test_used'] else result['audit'];changes=result['changes_vs_ar']
        text+=f'{name}: {"historical" if report["test_used"] else "audit"} AR-exact lost/gained={changes["exact_lines_lost"]}/{changes["exact_lines_gained"]}; preservation check={result["passes_preservation_check"]}\n'
    return text+f'Report: {path}\n'


def stats(ctx):
    verify_check(ctx)
    folder=ctx['run']/'historical_752'
    if not (folder/'seal.json').exists():raise ValueError('No completed 752-line comparison yet. Run Test yourself first.')
    report=verify_published(folder,ctx);print(render(report,folder/'report.json'))
