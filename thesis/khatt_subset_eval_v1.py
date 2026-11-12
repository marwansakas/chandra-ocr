"""User-requested four-run evaluation; the unfinished 10% trial is abandoned."""
from contextlib import contextmanager
from pathlib import Path

from thesis.khatt_ablation_v1 import workflow as w
from thesis.mdiff_arabic_v1.artifacts import file_hash, read, write

INCLUDED=('khatt100_only','khatt100_ours','khatt50_ours','khatt30_ours')
EXCLUDED='khatt10_ours'


def decision(ctx):
    path=ctx['args'].suite/'trial_decision.json'
    if not path.exists():return None
    value=read(path)
    expected=dict(version='khatt_without10_v1',binding_sha256=ctx['binding_sha256'],
        included_trials=list(INCLUDED),abandoned_trials=[EXCLUDED],
        reason='User abandoned the 10% trial based on validation history; evaluate the other completed trials only',
        checkpoint_policy='Existing best children-validation checkpoint of each completed included run',
        historical_test_lines=752)
    if value!=expected:raise ValueError('Unexpected trial-decision contents or binding')
    return value


def prepare(ctx):
    # All included selections must be complete and hash-verified before any
    # historical test reference can be accessed. Never fake 10% completion.
    selections={name:w.selected(ctx,name)[1] for name in INCLUDED}
    folder=ctx['args'].suite/'test_plan_without10';folder.mkdir(exist_ok=True)
    value=dict(version='khatt_without10_v1',binding_sha256=ctx['binding_sha256'],
        included_trials=list(INCLUDED),abandoned_trials=[EXCLUDED],
        reason='User abandoned the 10% trial based on validation history; evaluate the other completed trials only',
        checkpoint_policy='Existing best children-validation checkpoint of each completed included run',
        historical_test_lines=752)
    sources=[Path(__file__),ctx['args'].repo/'tests/test_khatt_subset_eval_v1.py']
    request=dict(decision=value,selected=selections,source_sha256={str(p.resolve()):file_hash(p) for p in sources},
        test_executed_during_preparation=False)
    report=ctx['args'].suite/'historical_752/report.json'
    if report.exists() and EXCLUDED in read(report)['results']:
        raise ValueError('A prior five-run report exists; preserve it before making a separate four-run comparison')
    target=folder/'request.json'
    if target.exists():
        if read(target)!=request:raise ValueError('Four-run evaluation plan changed')
    else:
        snapshot=folder/'source';snapshot.mkdir(exist_ok=True)
        for i,p in enumerate(sources):
            dest=snapshot/f'{i:02d}_{p.name}';dest.write_bytes(p.read_bytes())
            if file_hash(dest)!=request['source_sha256'][str(p.resolve())]:raise ValueError('Evaluation source snapshot mismatch')
        write(target,request)
    path=ctx['args'].suite/'trial_decision.json'
    if path.exists():decision(ctx)
    else:write(path,value)
    print('10% trial abandoned; four-run evaluation plan verified. No training or testing in preparation.',flush=True)
    for name,best in selections.items():print(name,'selected epoch',best['epoch'],flush=True)
    return target


@contextmanager
def evaluation_trials():
    # Configure the existing tested evaluator in this locked process only.
    # Assignment does not mutate data.TRIALS or any frozen source/checkpoint.
    original=w.TRIALS
    w.TRIALS={name:original[name] for name in INCLUDED}
    try:yield
    finally:w.TRIALS=original


def evaluate(ctx,action):
    if action not in ('TestAll','Stats'):raise ValueError('Evaluation action required')
    request=prepare(ctx)
    result_binding=request.parent/'result_binding.json'
    output=ctx['args'].suite/'historical_752'
    if action=='Stats' and result_binding.exists():
        saved=read(result_binding)
        if (saved['request_sha256']!=file_hash(request) or
            saved['report_sha256']!=file_hash(output/'report.json') or
            saved['predictions_sha256']!=file_hash(output/'predictions.jsonl')):
            raise ValueError('Four-run report/selection binding changed')
    with evaluation_trials():
        if action=='TestAll':w.test_all(ctx)
        else:w.stats(ctx['args'])
    if action=='TestAll':
        write(result_binding,dict(request_sha256=file_hash(request),
            report_sha256=file_hash(output/'report.json'),predictions_sha256=file_hash(output/'predictions.jsonl'),
            included_trials=list(INCLUDED),abandoned_trials=[EXCLUDED],historical_test_lines=752))
