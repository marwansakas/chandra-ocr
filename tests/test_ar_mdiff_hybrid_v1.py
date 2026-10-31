"""Complementarity bounds, reference-free selection, role isolation and frozen audit."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thesis.ar_mdiff_hybrid_v1 import logic as h
from thesis.ar_mdiff_hybrid_v1 import workflow as w
from thesis.mdiff_arabic_v1.artifacts import digest, file_hash, read, write, write_jsonl
from thesis.scripts.run_ar_mdiff_hybrid_v1 import parse_args


def prediction(sid,text,**kw):
    return dict(sample_id=sid,text=text,status='completed',has_eos=True,length_failure=False,**kw)


def constant_gate(value):
    return dict(version=h.VERSION,feature_names=list(h.FEATURE_NAMES),mean=[0.]*len(h.FEATURE_NAMES),
        scale=[1.]*len(h.FEATURE_NAMES),coefficients=[0.]*len(h.FEATURE_NAMES),intercept=value)


def test_complementarity_is_a_reference_dependent_bound_not_model():
    texts=['abc','def','ghi','jk']
    rows=[dict(sample_id=str(i),text=t,width=100,height=32) for i,t in enumerate(texts)]
    ar={str(i):prediction(str(i),t) for i,t in enumerate(['abc','dxx','gxx','jk'])}
    md={str(i):prediction(str(i),t) for i,t in enumerate(['abx','def','gxx','jx'])}
    summary,details=h.complementarity(rows,ar,md)
    assert summary['counts']['mdiff_better_lines']==1
    assert summary['counts']['ar_better_lines']==2
    assert summary['counts']['equal_character_distance_lines']==1
    assert summary['counts']['mdiff_exact_ar_wrong']==1
    assert summary['oracle_character_edit_headroom']==2
    assert summary['oracle']['exact_lines']==3
    assert summary['oracle_is_usable_model'] is False
    assert all('text' not in d for d in details)


def test_features_and_selection_never_read_reference_or_normalize_literals():
    row=dict(sample_id='a',width=100,height=32)
    ar=prediction('a',' أ  ب ');md=prediction('a','أ ب')
    f=h.features(row,ar,md)
    assert f==h.features(dict(row,text='COMPLETELY DIFFERENT REFERENCE'),ar,md)
    assert f[0]==len(ar['text'])/144
    selected=h.select([row],{'a':ar},{'a':md},constant_gate(1),.1)
    assert selected['a']['text']=='أ ب'
    selected=h.select([row],{'a':ar},{'a':md},constant_gate(0),.1)
    assert selected['a']['text']==' أ  ب '


def test_strict_threshold_identical_output_and_invalid_source_handling():
    row=dict(sample_id='a',width=100,height=32)
    ar={'a':prediction('a','abc')};md={'a':prediction('a','abd')}
    assert h.select([row],ar,md,constant_gate(.1),.1)['a']['selected_model']=='ar'
    assert h.select([row],ar,md,constant_gate(.2),.1)['a']['selected_model']=='mdiff'
    invalid=copy.deepcopy(md);invalid['a'].update(status='failed',has_eos=False)
    assert h.select([row],ar,invalid,constant_gate(1),.1)['a']['selected_model']=='ar'
    broken=copy.deepcopy(ar);broken['a'].update(status='failed',has_eos=False)
    recovered=h.select([row],broken,md,constant_gate(-1),.1)['a']
    assert recovered['selected_model']=='mdiff' and recovered['status']=='completed'
    equal={'a':prediction('a','abc')}
    assert h.select([row],ar,equal,constant_gate(1),.1)['a']['reason']=='identical_outputs'


def grouped_rows():
    return [dict(sample_id=f'{i:06d}_0001_0001_School__Fourth__Grade__Notebook_{i%12}_Male_Fourth_Grade_Page_001_line_{i:03d}',
                 text=f'text{i}',width=100,height=32) for i in range(60)]


def test_roles_are_grouped_balanced_deterministic_and_ignore_labels():
    rows=grouped_rows();roles=h.development_roles(rows)
    assert sum(roles['line_counts'].values())==60
    assert min(roles['line_counts'].values())>=10
    assert roles==h.development_roles(list(reversed(rows)))
    changed=[dict(r,text='changed',error=999) for r in rows]
    assert roles==h.development_roles(changed)
    from thesis.mdiff_arabic_v1.artifacts import group
    for r in rows:assert roles['sample_roles'][r['sample_id']]==roles['group_roles'][group(r)]


def test_subsets_keep_denominators_and_equal_exact_total_does_not_hide_damage():
    rows=[dict(sample_id='a',text='ab'),dict(sample_id='b',text='cd')]
    ar={'a':prediction('a','ab'),'b':prediction('b','xx')}
    hybrid={'a':dict(prediction('a','ax'),selected_model='mdiff'),'b':dict(prediction('b','cd'),selected_model='mdiff')}
    result=h.evaluate(rows,ar,hybrid)
    assert result['changes_vs_ar']['exact_lines_lost']==1
    assert result['changes_vs_ar']['exact_lines_gained']==1
    assert result['conditions']['strictly_fewer_character_edits_than_ar'] is True
    assert result['conditions']['zero_ar_exact_lines_lost'] is False
    assert not result['passes_ar_gate']
    subset=h.evaluate(rows[:1],ar,hybrid)
    assert subset['metrics']['lines']==1 and subset['metrics']['reference_chars']==2


def test_failed_predictions_remain_in_metrics():
    rows=[dict(sample_id='a',text='abc')]
    ar={'a':prediction('a','abc')}
    hybrid={'a':dict(prediction('a',''),selected_model='ar')}
    hybrid['a'].update(status='failed',has_eos=False)
    result=h.evaluate(rows,ar,hybrid)
    assert result['metrics']['reference_chars']==3 and result['metrics']['deletions']==3
    assert result['metrics']['failed_lines']==1 and not result['passes_ar_gate']


def test_ridge_is_fresh_finite_and_score_is_not_probability():
    rows=[dict(sample_id=str(i),text='abcd',width=100+i,height=32) for i in range(12)]
    ar={r['sample_id']:prediction(r['sample_id'],'abxx' if i%2 else 'abcd') for i,r in enumerate(rows)}
    md={r['sample_id']:prediction(r['sample_id'],'abcd' if i%2 else 'abxd') for i,r in enumerate(rows)}
    gate=h.fit_gate(rows,ar,md)
    assert gate['training_lines']==12 and gate['score_is_probability'] is False
    assert gate['feature_names']==list(h.FEATURE_NAMES)
    assert isinstance(h.score(gate,rows[0],ar['0'],md['0']),float)


def test_calibration_reports_rejected_raw_candidate_instead_of_ar_fallback():
    rows=[dict(sample_id='a',text='abc',width=100,height=32)]
    ar={'a':prediction('a','abc')};md={'a':prediction('a','xxx')}
    result=h.calibrate(rows,ar,md,constant_gate(.2))
    assert result['threshold']==0 and not result['calibration_passes_ar_gate']
    assert result['chosen']['switched_lines']==1
    assert result['chosen']['metrics']['char_distance']==3
    assert any(t['threshold'] is None and t['metrics']['char_distance']==0 for t in result['trials'])


def test_historical_test_is_blocked_before_cache_or_test_labels(monkeypatch,tmp_path):
    ctx=dict(args=SimpleNamespace(allow_exploratory=False))
    monkeypatch.setattr(w,'load_validation',lambda c:({}, {}, {'hybrid_validation_passes_gate':False}))
    monkeypatch.setattr(w,'load_data',lambda *a,**k:pytest.fail('Test labels accessed before eligibility'))
    with pytest.raises(ValueError,match='No historical data loaded'):w.test(ctx)


def test_fit_and_validation_use_separate_roles_and_freeze_once(monkeypatch,tmp_path):
    rows=grouped_rows();roles=h.development_roles(rows)
    ar={r['sample_id']:prediction(r['sample_id'],'base') for r in rows}
    md={r['sample_id']:prediction(r['sample_id'],'candidate') for r in rows}
    ctx=dict(rows=rows,roles=roles,ar=ar,mdiff=md,run=tmp_path,binding_sha256='b')
    write(tmp_path/'check.json',{})
    monkeypatch.setattr(w,'verify_check',lambda c:{})
    seen={}
    def fitting(rs,a,d):seen['fit']={r['sample_id'] for r in rs};return constant_gate(0)
    def calibration(rs,a,d,g):seen['calibration']={r['sample_id'] for r in rs};return dict(threshold=0,calibration_passes_ar_gate=False,selection_reason='fixture')
    monkeypatch.setattr(w,'fit_gate',fitting);monkeypatch.setattr(w,'calibrate',calibration)
    w.fit(ctx)
    audit={sid for sid,role in roles['sample_roles'].items() if role=='audit'}
    assert not audit & seen['fit'] and not audit & seen['calibration']
    assert not seen['fit'] & seen['calibration']
    assert len(seen['fit']|seen['calibration']|audit)==60
    with pytest.raises(FileExistsError):w.fit(ctx)
    w.load_fit(ctx)
    write(tmp_path/'fit/model.json',{'changed':True})
    with pytest.raises(ValueError):w.load_fit(ctx)


def test_real_small_gate_validation_is_immutable_and_does_not_refit_audit(monkeypatch,tmp_path):
    rows=[dict(r,text='abcd') for r in grouped_rows()]
    ar={r['sample_id']:prediction(r['sample_id'],'abxx' if i%2 else 'abcd') for i,r in enumerate(rows)}
    md={r['sample_id']:prediction(r['sample_id'],'abcd' if i%2 else 'abxd') for i,r in enumerate(rows)}
    ctx=dict(rows=rows,roles=h.development_roles(rows),ar=ar,mdiff=md,run=tmp_path,binding_sha256='binding',
        binding=dict(original_primary_validation_passes_release_gate=False),
        parent=dict(baselines=dict(frozen_v2_2=ar,span_hybrid_verifier_v4=ar)))
    write(tmp_path/'check.json',{})
    monkeypatch.setattr(w,'verify_check',lambda c:{})
    w.fit(ctx)
    frozen={p.name:file_hash(p) for p in (tmp_path/'fit').iterdir()}
    w.validation(ctx)
    report=read(tmp_path/'validation/report.json')
    assert report['test_used'] is False
    assert sum(r['metrics']['lines'] for r in report['role_results'].values())==len(rows)
    assert {p.name:file_hash(p) for p in (tmp_path/'fit').iterdir()}==frozen
    w.load_validation(ctx)
    with pytest.raises(FileExistsError):w.validation(ctx)
    (tmp_path/'validation/predictions.jsonl').write_text('{"tampered":true}\n',encoding='utf-8')
    with pytest.raises(ValueError):w.load_validation(ctx)


@pytest.mark.parametrize('action',['Check','Fit','Validation','ValidationStats','Test'])
def test_cli_routing_and_exploratory_flag(action,tmp_path):
    args=parse_args(['--action',action,'--project',str(tmp_path),'--run-name','hybrid1','--dry-run'])
    assert args.action==action and args.dry_run
    assert args.runner.name=='run_ar_mdiff_hybrid_v1.ps1'
    assert args.test_cache_name=='mdar_20261031_x01'
