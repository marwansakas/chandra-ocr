"""Immutable workflow; Check reads saved validation only; Fit is user-invoked CPU work."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import time

from thesis.scripts.run_mdiff_arabic_v1 import parse_args as primary_args
from thesis.mdiff_arabic_v1 import experiment as primary
from thesis.mdiff_arabic_v1.artifacts import (
    POLICY, canonical, damage, digest, eligibility, file_hash, grouped_intervals,
    indexed, jsonl, load_data, metrics, read, require_equal, write, write_jsonl,
)
from .logic import RECIPE, VERSION, calibrate, complementarity, development_roles, evaluate, fit_gate, select


def sources(args):
    paths=list((args.repo/'thesis/ar_mdiff_hybrid_v1').glob('*.py'))
    paths += [args.repo/'thesis/scripts/run_ar_mdiff_hybrid_v1.py',args.repo/'tests/test_ar_mdiff_hybrid_v1.py',args.runner]
    return {str(p):file_hash(p) for p in sorted(paths)}


def context(args):
    p_args=primary_args(['--action','ValidationStats','--project',str(args.project),
        '--run-name',args.mdiff_run_name,'--peer-run-name',args.ar_run_name,'--arm','mdiff',
        '--device',args.device,'--readiness-protocol','geometry_matched_v1'])
    p=primary.context(p_args)
    md_best=primary.selected(p);peer=primary.peer_context(p);ar_best=primary.selected(peer)
    rows=p['parts']['validation']
    md=indexed(jsonl(p['run']/md_best['predictions']),rows)
    ar=indexed(jsonl(peer['run']/ar_best['predictions']),rows)
    validation=read(p['run']/'validation/report.json');seal=read(p['run']/'validation/gate.json')
    expected=dict(binding_sha256=p['binding_sha256'],validation_report_sha256=file_hash(p['run']/'validation/report.json'),
        selected_checkpoint_sha256=md_best['sha256'],peer_selected_checkpoint_sha256=ar_best['sha256'],
        predictions_sha256=file_hash(p['run']/'validation/predictions.jsonl'),policy_sha256=digest(POLICY))
    for key,value in expected.items(): require_equal(seal[key],value,'primary validation '+key)
    for name,preds in [('mdiff_arabic_raw',md),('transformer_arabic_control',ar)]:
        actual=eligibility(rows,p['baselines'],preds)
        require_equal(actual['conditions'],validation['arms'][name]['conditions'],'primary validation conditions')
    require_equal(validation['partition'],'validation','primary report partition')
    require_equal(validation['test_used'],False,'primary report test flag')
    roles=development_roles(rows)
    primary_artifacts={str(p['run']/'best.json'):file_hash(p['run']/'best.json'),
        str(peer['run']/'best.json'):file_hash(peer['run']/'best.json'),
        str(p['run']/md_best['predictions']):md_best['predictions_sha256'],
        str(peer['run']/ar_best['predictions']):ar_best['predictions_sha256'],
        str(p['run']/md_best['checkpoint']):md_best['sha256'],
        str(peer['run']/ar_best['checkpoint']):ar_best['sha256'],
        str(p['run']/'validation/report.json'):expected['validation_report_sha256'],
        str(p['run']/'validation/gate.json'):file_hash(p['run']/'validation/gate.json')}
    binding=dict(version=VERSION,recipe=RECIPE,run_name=args.run_name,
        parent_names=dict(ar=args.ar_run_name,mdiff=args.mdiff_run_name),
        parent_bindings=dict(ar=peer['binding_sha256'],mdiff=p['binding_sha256']),
        primary_artifacts=primary_artifacts,source_sha256=sources(args),
        environment=p['binding']['environment'],split_hashes=p['evidence']['split_hashes'],
        split_lock_sha256=p['evidence']['split_lock_sha256'],roles=roles,
        original_primary_validation_passes_release_gate=validation['validation_passes_release_gate'],
        primary_selected_epochs=dict(ar=ar_best['epoch'],mdiff=md_best['epoch']))
    return dict(args=args,parent=p,peer=peer,md_best=md_best,ar_best=ar_best,rows=rows,ar=ar,mdiff=md,
        roles=roles,binding=binding,binding_sha256=digest(binding),run=args.drive/'hybrids'/args.run_name)


def verify_check(ctx):
    checked=read(ctx['run']/'check.json')
    require_equal(ctx['binding'],checked['binding'],'hybrid check binding')
    require_equal(checked['binding_sha256'],ctx['binding_sha256'],'hybrid binding hash')
    require_equal(checked['roles_sha256'],file_hash(ctx['run']/'roles.json'),'role manifest hash')
    require_equal(checked['diagnostic_sha256'],file_hash(ctx['run']/'complementarity.json'),'diagnostic hash')
    return checked


def role_rows(ctx,name):
    return [r for r in ctx['rows'] if ctx['roles']['sample_roles'][r['sample_id']]==name]


def table(models):
    result='Model                        Lines   CER %   WER %  Exact %   Sub   Ins   Del\n'
    for name,m in models.items():
        result+=f'{name:<28} {m["lines"]:5d} {100*m["cer"]:7.2f} {100*m["wer"]:7.2f} {100*m["exact_line_accuracy"]:8.2f} {m["substitutions"]:5d} {m["insertions"]:5d} {m["deletions"]:5d}\n'
    return result


def check(ctx):
    if (ctx['run']/'check.json').exists():
        verify_check(ctx);print('Existing immutable hybrid Check matches:',ctx['run']);return
    ctx['run'].mkdir(parents=True,exist_ok=False)
    summary,details=complementarity(ctx['rows'],ctx['ar'],ctx['mdiff'])
    snapshot=ctx['run']/'source';snapshot.mkdir()
    for i,(path,sha) in enumerate(ctx['binding']['source_sha256'].items()):
        target=snapshot/f'{i:02d}_{Path(path).name}';shutil.copyfile(path,target)
        require_equal(file_hash(target),sha,'hybrid source copy')
    write(ctx['run']/'roles.json',ctx['roles'],exclusive=True)
    write(ctx['run']/'complementarity.json',dict(summary,test_used=False,partition='development-used validation'),exclusive=True)
    write_jsonl(ctx['run']/'complementarity_lines.jsonl',details)
    write(ctx['run']/'check.json',dict(binding=ctx['binding'],binding_sha256=ctx['binding_sha256'],
        diagnostic_sha256=file_hash(ctx['run']/'complementarity.json'),roles_sha256=file_hash(ctx['run']/'roles.json'),
        created_utc=primary.stamp(),passed=True,test_used=False),exclusive=True)
    print(table(dict(transformer_arabic_control=summary['ar'],mdiff_arabic_raw=summary['mdiff'])))
    print('MDiff better / AR better / tied:',summary['counts']['mdiff_better_lines'],summary['counts']['ar_better_lines'],summary['counts']['equal_character_distance_lines'])
    print('MDiff exact while AR wrong:',summary['counts']['mdiff_exact_ar_wrong'])
    print(f'Oracle CER={100*summary["oracle"]["cer"]:.2f}% | diagnostic upper bound only; not a model')
    print('Frozen development roles:',ctx['roles']['line_counts'],'| group counts:',ctx['roles']['group_counts'])
    print('Check only; no gate fit or inference performed. Report:',ctx['run']/'complementarity.json')


def fit(ctx):
    checked=verify_check(ctx)
    folder=ctx['run']/'fit';folder.mkdir(exist_ok=False)
    fitting=role_rows(ctx,'fit');calibration=role_rows(ctx,'calibration')
    gate=fit_gate(fitting,ctx['ar'],ctx['mdiff'])
    calibrated=calibrate(calibration,ctx['ar'],ctx['mdiff'],gate)
    write(folder/'model.json',gate,exclusive=True)
    write(folder/'calibration.json',calibrated,exclusive=True)
    write(folder/'evidence.json',dict(fit_sample_ids=[r['sample_id'] for r in fitting],
        calibration_sample_ids=[r['sample_id'] for r in calibration],
        audit_labels_used_for_gate_fit_or_threshold=False,
        limitation=RECIPE['limitations'],test_used=False),exclusive=True)
    write(folder/'seal.json',dict(binding_sha256=ctx['binding_sha256'],check_sha256=file_hash(ctx['run']/'check.json'),
        files={name:file_hash(folder/name) for name in ('model.json','calibration.json','evidence.json')}),exclusive=True)
    print('Fresh ridge gate fitted on CPU. Base recognizers unchanged.')
    print('Calibration threshold:',calibrated['threshold'],'| calibration passes AR gate:',calibrated['calibration_passes_ar_gate'])
    print(calibrated['selection_reason'])
    print('Fit report:',folder/'calibration.json')


def load_fit(ctx):
    verify_check(ctx);folder=ctx['run']/'fit';seal=read(folder/'seal.json')
    require_equal(seal['binding_sha256'],ctx['binding_sha256'],'fit binding')
    require_equal(seal['check_sha256'],file_hash(ctx['run']/'check.json'),'fit check')
    for name,sha in seal['files'].items():require_equal(file_hash(folder/name),sha,'fit '+name)
    return read(folder/'model.json'),read(folder/'calibration.json')


def model_scores(rows,baselines,ar,mdiff,hybrid):
    return {name:metrics(rows,preds) for name,preds in [
        *baselines.items(),('transformer_arabic_control',ar),('mdiff_arabic_raw',mdiff),('ar_mdiff_hybrid_raw',hybrid)]}


def validation(ctx):
    gate,calibration=load_fit(ctx)
    folder=ctx['run']/'validation';folder.mkdir(exist_ok=False)
    predictions=select(ctx['rows'],ctx['ar'],ctx['mdiff'],gate,calibration['threshold'])
    results={name:evaluate(role_rows(ctx,name),ctx['ar'],predictions) for name in ('fit','calibration','audit')}
    require_equal(results['calibration'],{k:v for k,v in calibration['chosen'].items() if k!='threshold'},'reproduced calibration')
    combined=evaluate(ctx['rows'],ctx['ar'],predictions)
    old=eligibility(ctx['rows'],ctx['parent']['baselines'],predictions)
    passed=all(x['passes_ar_gate'] for x in [results['calibration'],results['audit'],combined])
    report=dict(version=VERSION,partition='validation',test_used=False,recipe=RECIPE,
        binding_sha256=ctx['binding_sha256'],fit_seal_sha256=file_hash(ctx['run']/'fit/seal.json'),
        threshold=calibration['threshold'],roles=ctx['roles'],role_results=results,combined=combined,
        hybrid_validation_passes_gate=passed,passes_original_v4_release_gate=old['validation_passes_release_gate'],
        original_primary_validation_passes_release_gate=ctx['binding']['original_primary_validation_passes_release_gate'],
        damage_vs_v4=old['damage']['span_hybrid_verifier_v4'],damage_vs_v2_2=old['damage']['frozen_v2_2'],
        models=model_scores(ctx['rows'],ctx['parent']['baselines'],ctx['ar'],ctx['mdiff'],predictions),
        confidence_intervals_vs_ar=grouped_intervals(ctx['rows'],predictions,ctx['ar']),
        report_status='raw selector; no automatic AR fallback, model replacement or promotion',
        interpretation='Full 150-line table mixes fit/calibration/audit; inspect role results. '+RECIPE['limitations'])
    write_jsonl(folder/'predictions.jsonl',predictions.values());report['predictions_sha256']=file_hash(folder/'predictions.jsonl')
    write(folder/'report.json',report,exclusive=True)
    write(folder/'seal.json',dict(binding_sha256=ctx['binding_sha256'],report_sha256=file_hash(folder/'report.json'),
        predictions_sha256=report['predictions_sha256'],model_sha256=file_hash(ctx['run']/'fit/model.json'),
        calibration_sha256=file_hash(ctx['run']/'fit/calibration.json'),hybrid_validation_passes_gate=passed),exclusive=True)
    text=render(report,folder/'report.json');(folder/'report.txt').write_text(text,encoding='utf-8');print(text)


def render(report,path):
    text=('FULL DEVELOPMENT TABLE: fit/calibration/audit combined; not held-out generalization.\n'
          if report['partition']=='validation' else 'HISTORICAL DEVELOPMENT-USED COMPARISON: frozen selector, cached candidates.\n')
    text+=table(report['models'])
    if report['partition']=='validation':
        for name,r in report['role_results'].items():
            d=r['changes_vs_ar'];text+=f'{name}: lines={r["metrics"]["lines"]}, CER={100*r["metrics"]["cer"]:.2f}%, switches={r["switched_lines"]}, AR exact lost/gained={d["exact_lines_lost"]}/{d["exact_lines_gained"]}, passes={r["passes_ar_gate"]}\n'
        text+=f'Hybrid gate against AR: {report["hybrid_validation_passes_gate"]} | original v4 gate: {report["passes_original_v4_release_gate"]}\n'
    else:
        d=report['comparison_vs_ar']['changes_vs_ar'];text+=f'Exact lost/gained vs AR: {d["exact_lines_lost"]}/{d["exact_lines_gained"]}\n'
        text+=f'Frozen hybrid validation gate: {report["hybrid_validation_passes_gate"]} | exploratory override: {report["exploratory_override"]}\n'
    return text+f'Test used: {report["test_used"]} | Report: {path}\n'


def load_validation(ctx):
    gate,calibration=load_fit(ctx);folder=ctx['run']/'validation'
    report=read(folder/'report.json');seal=read(folder/'seal.json')
    require_equal(report['binding_sha256'],ctx['binding_sha256'],'validation report binding')
    require_equal(report['fit_seal_sha256'],file_hash(ctx['run']/'fit/seal.json'),'validation fit seal')
    require_equal(seal['binding_sha256'],ctx['binding_sha256'],'validation binding')
    for key,path in [('report_sha256',folder/'report.json'),('predictions_sha256',folder/'predictions.jsonl'),
                     ('model_sha256',ctx['run']/'fit/model.json'),('calibration_sha256',ctx['run']/'fit/calibration.json')]:
        require_equal(seal[key],file_hash(path),'validation '+key)
    predictions=select(ctx['rows'],ctx['ar'],ctx['mdiff'],gate,calibration['threshold'])
    require_equal(list(predictions.values()),jsonl(folder/'predictions.jsonl'),'frozen hybrid outputs')
    recomputed={name:evaluate(role_rows(ctx,name),ctx['ar'],predictions) for name in ('calibration','audit')}
    passed=all(v['passes_ar_gate'] for v in recomputed.values()) and evaluate(ctx['rows'],ctx['ar'],predictions)['passes_ar_gate']
    require_equal(passed,report['hybrid_validation_passes_gate'],'recomputed hybrid gate')
    require_equal(seal['hybrid_validation_passes_gate'],passed,'sealed hybrid gate')
    return gate,calibration,report


def stats(ctx):
    _,_,report=load_validation(ctx);print(render(report,ctx['run']/'validation/report.json'))


def test(ctx):
    gate,calibration,validation_report=load_validation(ctx)
    passed=validation_report['hybrid_validation_passes_gate']
    if not passed and not ctx['args'].allow_exploratory:
        raise ValueError('STOP: hybrid validation gate failed. No historical data loaded. An explicit -AllowExploratory is required for a descriptive comparison despite failure.')
    folder=ctx['run']/'historical_test'
    if folder.exists():raise FileExistsError('Historical hybrid comparison already exists; never overwrite')
    # No test references/candidate caches are opened by Check/Fit/Validation.
    cache=ctx['args'].drive/'exploratory'/ctx['args'].test_cache_name
    cache_report=read(cache/'report.json');request=read(cache/'request.json')
    require_equal(file_hash(cache/'request.json'),cache_report['request_sha256'],'test cache request')
    require_equal(file_hash(cache/'predictions.jsonl'),cache_report['predictions_sha256'],'test cache predictions')
    require_equal(cache_report['split_hashes'],ctx['binding']['split_hashes'],'test cache manifests')
    for name,best in [('mdiff_arabic_raw',ctx['md_best']),('transformer_arabic_control',ctx['ar_best'])]:
        require_equal(cache_report['selected_models'][name]['sha256'],best['sha256'],'test cache selected model')
    parts,_,_,evidence=load_data(ctx['args'].drive,test_authorized=True,readiness_protocol='geometry_matched_v1')
    rows=parts['test'];by_id={r['sample_id']:r for r in rows}
    models={}
    for prediction in jsonl(cache/'predictions.jsonl'):
        name=prediction['model'];sid=prediction['sample_id']
        if sid not in by_id:raise ValueError('Unexpected test cache sample')
        require_equal(prediction['reference_literal'],by_id[sid]['text'],'cached literal reference')
        require_equal(prediction['image_sha256'],by_id[sid]['image_sha256'],'cached image binding')
        values=models.setdefault(name,{})
        if sid in values:raise ValueError('Duplicate cached model/sample prediction')
        values[sid]=prediction
    for name,values in models.items():indexed(list(values.values()),rows)
    ar=models['transformer_arabic_control'];mdiff=models['mdiff_arabic_raw']
    began=time.perf_counter();hybrid=select(rows,ar,mdiff,gate,calibration['threshold']);elapsed=time.perf_counter()-began
    baseline={name:models[name] for name in ('paddle','frozen_v2_2','span_hybrid_verifier_v4')}
    comparison=evaluate(rows,ar,hybrid)
    report=dict(version=VERSION,partition='test',test_used=True,binding_sha256=ctx['binding_sha256'],
        hybrid_validation_passes_gate=passed,exploratory_override=not passed and ctx['args'].allow_exploratory,
        benchmark_status='historical development-used; not fresh generalization; no promotion',
        validation_report_sha256=file_hash(ctx['run']/'validation/report.json'),
        test_cache_report_sha256=file_hash(cache/'report.json'),test_cache_predictions_sha256=file_hash(cache/'predictions.jsonl'),
        threshold=calibration['threshold'],models=model_scores(rows,baseline,ar,mdiff,hybrid),
        comparison_vs_ar=comparison,damage_vs_v4=damage(rows,baseline['span_hybrid_verifier_v4'],hybrid),
        damage_vs_v2_2=damage(rows,baseline['frozen_v2_2'],hybrid),
        confidence_intervals_vs_ar=grouped_intervals(rows,hybrid,ar),selector_cpu_seconds=elapsed,
        inference='cached frozen candidates; no recognizer inference or retraining',promotion_performed=False,
        coverage_issues=[x for x in evidence['coverage_issues'] if x['partition']=='test'])
    folder.mkdir(exist_ok=False);write_jsonl(folder/'predictions.jsonl',hybrid.values())
    report['predictions_sha256']=file_hash(folder/'predictions.jsonl');write(folder/'report.json',report,exclusive=True)
    text=render(report,folder/'report.json');(folder/'report.txt').write_text(text,encoding='utf-8');print(text)
