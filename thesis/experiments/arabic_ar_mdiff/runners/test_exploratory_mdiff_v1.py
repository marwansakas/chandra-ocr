"""Authorization can waive eligibility, never artifact integrity or coverage."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pytest
import torch
from PIL import Image

spec = importlib.util.spec_from_file_location('exploratory_eval', Path(__file__).with_name('evaluate_mdiff_exploratory_v1.py'))
x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x)


def fixture_validation(tmp_path):
    rows = [dict(sample_id='a', text='ab'), dict(sample_id='b', text='cd')]
    baseline = {r['sample_id']:dict(sample_id=r['sample_id'], text=t, status='completed')
                for r,t in zip(rows,['ab','xx'])}
    candidate = {r['sample_id']:dict(sample_id=r['sample_id'], text=t, status='completed')
                 for r,t in zip(rows,['ax','cd'])}
    ctx = dict(run=tmp_path/'md', args=SimpleNamespace(arm='mdiff'), binding_sha256='md-binding',
               parts=dict(validation=rows), baselines=dict(frozen_v2_2=baseline, span_hybrid_verifier_v4=baseline))
    peer = dict(run=tmp_path/'ar', args=SimpleNamespace(arm='ar'), binding_sha256='ar-binding')
    bests=[]
    arms={};conditions={}
    for c in (ctx,peer):
        x.write_jsonl(c['run']/'v001.jsonl', candidate.values())
        bests.append(dict(sha256=c['binding_sha256'],predictions='v001.jsonl',validation_metrics=x.e.metrics(rows,candidate)))
        arm=x.e.eligibility(rows,ctx['baselines'],candidate)
        name=x.e.model_name(c['args'].arm)
        arms[name]=arm
        conditions.update({name+'.'+k:v for k,v in arm['conditions'].items()})
    report=dict(partition='validation',test_used=False,binding_sha256=ctx['binding_sha256'],peer_binding_sha256=peer['binding_sha256'],
                arms=arms,gate_conditions=conditions,validation_passes_release_gate=False)
    x.write(ctx['run']/'validation/report.json',report)
    x.write_jsonl(ctx['run']/'validation/predictions.jsonl',candidate.values())
    seal=dict(binding_sha256=ctx['binding_sha256'],validation_report_sha256=x.file_hash(ctx['run']/'validation/report.json'),
              selected_checkpoint_sha256=bests[0]['sha256'],peer_selected_checkpoint_sha256=bests[1]['sha256'],
              policy_sha256=x.digest(x.POLICY),predictions_sha256=x.file_hash(ctx['run']/'validation/predictions.jsonl'),validation_passes_release_gate=False)
    x.write(ctx['run']/'validation/gate.json',seal)
    return ctx,peer,bests


def test_explicit_exploratory_check_retains_false_gate(tmp_path):
    ctx,peer,bests=fixture_validation(tmp_path)
    report,seal=x.verify_validation(ctx,peer,*bests)
    assert report['validation_passes_release_gate'] is False
    assert seal['validation_passes_release_gate'] is False
    assert x.read(ctx['run']/'validation/report.json')['validation_passes_release_gate'] is False


def test_integrity_and_actual_verdict_are_not_waived(tmp_path):
    ctx,peer,bests=fixture_validation(tmp_path)
    path=ctx['run']/'validation/report.json'
    report=x.read(path);report['validation_passes_release_gate']=True;x.write(path,report)
    with pytest.raises(ValueError): x.verify_validation(ctx,peer,*bests)
    seal=x.read(ctx['run']/'validation/gate.json')
    seal['validation_report_sha256']=x.file_hash(path)
    x.write(ctx['run']/'validation/gate.json',seal)
    with pytest.raises(ValueError,match='validation verdict'): x.verify_validation(ctx,peer,*bests)


def test_predictions_do_not_need_references_and_keep_image_failures(tmp_path):
    from thesis.diffusion.tokenizer import CharacterTokenizer
    from thesis.diffusion.images import ImageConfig
    dataset=tmp_path/'ground_truth_lines';dataset.mkdir()
    Image.new('RGB',(16,16),'white').save(dataset/'ok.png')
    rows=[dict(sample_id='a',image_path='ok.png'),dict(sample_id='b',image_path='missing.png')]
    ctx=dict(args=SimpleNamespace(drive=tmp_path,device='cpu'),image_config=ImageConfig(height=32,max_width=640,patch_size=4),
             tokenizer=CharacterTokenizer(['x']))
    class Dummy:
        arm='mdiff'
        def eval(self): return self
        def generate(self,pixels,widths,steps):
            return torch.tensor([[3,2]]),dict(decoder_passes=6,sampler_mode='masked_reflect_v1',sampler_steps=3)
    predictions=x.predict_images(Dummy(),rows,ctx,tmp_path/'predictions.jsonl')
    assert set(predictions)=={'a','b'}
    assert predictions['a']['text']=='x' and predictions['a']['status']=='completed'
    assert predictions['b']['text']=='' and predictions['b']['status']=='failed'
    assert len(x.jsonl(tmp_path/'predictions.jsonl'))==2
