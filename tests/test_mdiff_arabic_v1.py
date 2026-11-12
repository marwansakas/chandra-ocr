"""Scientific contracts: literal text, actual updates, denominator and gate safety."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thesis.diffusion.tokenizer import CharacterTokenizer,read_literal,training_text
from thesis.mdiff_arabic_v1 import artifacts as a
from thesis.mdiff_arabic_v1 import experiment as e
from thesis.mdiff_arabic_v1.model import Recognizer,structured_noise,correction_noise,normalized_ce,PAD,MASK,EOS
from thesis.scripts.run_mdiff_arabic_v1 import parse_args


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.set_num_threads(2);torch.manual_seed(123)


def test_literal_arabic_spaces_newline_pad_and_unknown(tmp_path):
    text='  أَـب 12،\nج '
    t=CharacterTokenizer(sorted(set(text)))
    ids=t.encode(text,32,eos_fill=False)
    assert t.decode(ids)==(text,True)
    assert ids[len(text)]==EOS and all(x==PAD for x in ids[len(text)+1:])
    assert training_text(text,'no_harakat_v1')=='  أـب 12،\nج '
    p=tmp_path/'label.txt';p.write_bytes((text+'\r\n').encode())
    assert read_literal(p)==text
    p.write_bytes(b' x \n\n');assert read_literal(p)==' x \n'
    with pytest.raises(ValueError,match='Out-of-vocabulary'): t.encode('Z',32)
    with pytest.raises(ValueError,match='overflow'): t.encode(text,len(text))
    with pytest.raises(ValueError,match='Unresolved'): t.decode([MASK,EOS])
    assert t.decode([t.ids['ج']]*5)==('ج'*5,False)


def test_mask_and_replacement_support_valid_positions_and_wrong_visible_tokens():
    targets=torch.tensor([[3,4,5,EOS,PAD,PAD]]*100)
    noisy,hidden,t=structured_noise(targets)
    assert hidden[:,:4].any(1).all() and not hidden[:,4:].any()
    assert noisy[:,4:].eq(MASK).all()
    assert torch.allclose(t,hidden.sum(1)/4)
    reflected,replaced,rate=correction_noise(targets,10)
    assert reflected[replaced].ne(targets[replaced]).all()
    assert (rate==0).any() and replaced[:,:4].any()
    assert reflected[:,4:].eq(MASK).all()
    assert not replaced[:,4:].any()
    assert replaced[:,3].any()  # wrong visible EOS can be corrected too


class Scripted(Recognizer):
    def __init__(self):
        nn.Module.__init__(self)
        self.arm='mdiff';self.capacity=4;self.decoder_passes=0
        self.encoder=nn.Identity()
        self.calls=[]
    def encode(self,pixels,widths):
        return torch.zeros(len(pixels),2,8),torch.zeros(len(pixels),2,dtype=torch.bool)
    def decode_logits(self,tokens,memory,memory_mask,t,causal=False):
        self.decoder_passes+=1;self.calls.append(tokens.clone())
        out=torch.full((len(tokens),4,8),-20.)
        # Deliberately wrong visible character on first denoise, corrected by
        # separate reflect call even though it was assigned high confidence.
        values=[6,4,EOS,EOS] if self.decoder_passes==1 else [3,4,EOS,EOS]
        for i,v in enumerate(values): out[:,i,v]=20.
        return out


def test_actual_six_pass_updates_correct_confident_visible_error_and_preserve():
    model=Scripted().eval()
    px=torch.zeros(1,3,8,12);w=torch.tensor([12])
    ids,info=model.generate(px,w,steps=3,trace=True)
    assert info['decoder_passes']==6
    assert info['trace'][0]['tokens'][0][0]==6
    assert info['trace'][1]['tokens'][0][0]==3
    assert model.calls[1][0,0]==6  # wrong token really visible to correction
    assert model.calls[2].eq(MASK).any()  # actual remasking between steps
    assert ids.tolist()==[[3,4,EOS,EOS]]
    model.decoder_passes=6
    good=torch.tensor([[3,4,EOS,PAD]])
    again,_=model.generate(px,w,initial=good,steps=3)
    assert again[0,:3].tolist()==good[0,:3].tolist()
    fresh=Scripted().eval()
    _,probe=fresh.generate(px,w,initial=good,steps=3,trace=True)
    assert probe['trace'][0]['tokens'][0][0]==3  # denoise cannot overwrite a visible token


def test_ar_causal_decoder_cannot_see_future_and_output_specials_excluded():
    m=Recognizer(12,capacity=6,arm='ar',dim=32,layers=1,dropout=0).eval()
    memory=torch.randn(1,5,32);mm=torch.tensor([[False]*4+[True]])
    left=torch.tensor([[MASK,3,4,5,6,7]])
    right=left.clone();right[:,3:]=11
    a1=m.decode_logits(left,memory,mm,torch.zeros(1),causal=True)
    a2=m.decode_logits(right,memory,mm,torch.zeros(1),causal=True)
    torch.testing.assert_close(a1[:,:3],a2[:,:3])
    assert a1.argmax(-1).ge(EOS).all()
    changed=memory.clone();changed[:,4]=1e5
    a3=m.decode_logits(left,changed,mm,torch.zeros(1),causal=True)
    torch.testing.assert_close(a1,a3)


def test_loss_mask_does_not_reward_pad_tail():
    target=torch.tensor([[3,EOS,PAD,PAD]])
    logits=torch.randn(1,4,8,requires_grad=True)
    loss=normalized_ce(logits,target,target.ne(PAD));loss.backward()
    assert logits.grad[:,:2].abs().sum()>0
    assert logits.grad[:,2:].abs().sum()==0


def sample_predictions(texts,**extra):
    return {str(i):dict(sample_id=str(i),text=t,status='completed',has_eos=True,**extra) for i,t in enumerate(texts)}


def test_unclamped_cer_and_failure_denominator_spaces():
    rows=[dict(sample_id='0',text='أ ب'),dict(sample_id='1',text='ج')]
    preds=sample_predictions(['أ  ب','']);preds['1'].update(status='failed',has_eos=False)
    result=a.metrics(rows,preds)
    assert result['reference_chars']==4 and result['char_distance']==2
    assert result['cer']==.5 and result['failed_lines']==1 and result['lines']==2
    assert result['space_insertions']==1
    preds=sample_predictions(['x'*30,'x'*30])
    assert a.metrics(rows,preds)['cer']>1
    with pytest.raises(ValueError,match='membership'): a.metrics(rows,{'0':preds['0']})


def test_equal_total_exact_count_cannot_hide_lost_line_and_gate_rejects():
    rows=[dict(sample_id='0',text='ab'),dict(sample_id='1',text='cd')]
    baseline=sample_predictions(['ab','xx']);candidate=sample_predictions(['ax','cd'])
    d=a.damage(rows,baseline,candidate)
    assert d['exact_lines_lost']==d['exact_lines_gained']==1
    gate=a.eligibility(rows,dict(frozen_v2_2=baseline,span_hybrid_verifier_v4=baseline),candidate)
    assert gate['conditions']['strictly_fewer_character_edits']
    assert not gate['conditions']['zero_baseline_exact_lines_damaged']
    assert not gate['validation_passes_release_gate']


def test_gate_requires_true_boolean_hashes_and_validation_only():
    report=dict(partition='validation',test_used=False,validation_passes_release_gate=True,gate_conditions=dict(a=True))
    seal=dict(binding_sha256='b',validation_report_sha256='r',selected_checkpoint_sha256='c',peer_selected_checkpoint_sha256='p')
    kw=dict(expected_binding='b',report_sha='r',checkpoint_sha='c',peer_checkpoint_sha='p')
    a.enforce_gate(report,seal,**kw)
    for changed in [dict(validation_passes_release_gate=False),dict(validation_passes_release_gate='true'),dict(test_used=True),dict(partition='test'),dict(gate_conditions=dict(a=False))]:
        with pytest.raises(ValueError): a.enforce_gate(dict(report,**changed),seal,**kw)
    for key in seal:
        with pytest.raises(ValueError,match='binding mismatch'): a.enforce_gate(report,dict(seal,**{key:'changed'}),**kw)


def test_test_action_stops_before_loading_test_data_on_failed_gate(monkeypatch,tmp_path):
    ctx=dict(run=tmp_path,binding_sha256='b')
    best=dict(sha256='c')
    monkeypatch.setattr(e,'selected',lambda ctx:best)
    monkeypatch.setattr(e,'peer_context',lambda ctx:ctx)
    def forbidden(*args,**kwargs): pytest.fail('Test data accessed before gate passed')
    monkeypatch.setattr(e,'load_data',forbidden)
    a.write(tmp_path/'validation/report.json',dict(partition='validation',test_used=False,validation_passes_release_gate=False))
    a.write(tmp_path/'validation/gate.json',{})
    with pytest.raises(ValueError,match='STOP'): e.test(ctx)


def test_failed_or_stale_readiness_cannot_start_train(monkeypatch,tmp_path):
    ctx=dict(checks=tmp_path,binding_sha256='b')
    a.write(tmp_path/'check.json',{})
    report=dict(passed=False,binding_sha256='b',check_sha256=a.file_hash(tmp_path/'check.json'))
    path=tmp_path/'overfit/report.json';a.write(path,report)
    a.write(tmp_path/'overfit_latest.json',dict(report=str(path),sha256=a.file_hash(path)))
    with pytest.raises(ValueError,match='readiness failed'): e.prerequisite(ctx,'overfit')
    report.update(passed=True,binding_sha256='stale');a.write(path,report)
    a.write(tmp_path/'overfit_latest.json',dict(report=str(path),sha256=a.file_hash(path)))
    with pytest.raises(ValueError,match='does not match'): e.prerequisite(ctx,'overfit')


def test_group_keeps_pages_and_lines_of_same_notebook_together():
    left=dict(sample_id='000003_0001_0001_Kfar__Yasif__Fourth__Grade__Notebook_5_Female_Fourth_Grade_Page_030_line_003')
    right=dict(sample_id='001245_0105_0105_Kfar__Yasif__Fourth__Grade__Notebook_5_Female_Fourth_Grade_Page_017_line_046')
    assert a.group(left)==a.group(right)
    assert 'notebook_proxy:' in a.group(left)


def test_selection_is_separate_from_eligibility_and_ties_choose_earliest():
    m=dict(char_distance=10,word_distance=5,exact_lines=2)
    assert e.training_key(m,2)<e.training_key(m,3)
    assert e.training_key(dict(m,char_distance=9),200)<e.training_key(m,2)
    assert e.training_key(dict(m,word_distance=4),5)<e.training_key(m,2)
    assert list(e.remaining_epochs(200,29))==[200]
    assert list(e.remaining_epochs(201,0))==[]
    assert list(e.remaining_epochs(50,30))==[]


@pytest.mark.parametrize('action',['Setup','Check','Smoke','Overfit','Train','Validation','ValidationStats','Test'])
def test_cli_action_routing_and_budgets(action):
    args=parse_args(['--action',action,'--project',str(Path.cwd()),'--run-name','md1','--arm','mdiff','--dry-run'])
    assert args.action==action and args.epochs==200 and args.patience==30 and args.dry_run


def test_environment_changes_explain_package_drift():
    with pytest.raises(ValueError,match='torch'):
        a.require_equal(dict(packages=dict(torch='new')),dict(packages=dict(torch='old')),'environment')


def test_validation_freezes_raw_paired_report_and_stats_reads_it(monkeypatch,tmp_path):
    rows=[dict(sample_id=str(i),text=t,image_sha256='image'+str(i)) for i,t in enumerate(['ab','cd'])]
    baselines={n:sample_predictions(['xx','yy']) for n in ['paddle','frozen_v2_2','span_hybrid_verifier_v4']}
    candidate=sample_predictions(['ab','cd'])
    contexts=[]
    bests={}
    for arm in ['mdiff','ar']:
        run=tmp_path/arm;checks=tmp_path/(arm+'_checks');run.mkdir();checks.mkdir()
        for path in ['tokenizer.json','dependencies.freeze.txt']:
            (run/path).write_text('fixture',encoding='utf-8')
        a.write(checks/'check.json',dict(fixture=True))
        a.write(run/'run.json',dict(encoder_initialization_sha256='shared'))
        a.write(run/'training_complete.json',dict(epoch=2))
        (run/'m002.pt').write_bytes(b'fixture checkpoint')
        a.write_jsonl(run/'v002.jsonl',candidate.values())
        ctx=dict(run=run,checks=checks,args=SimpleNamespace(arm=arm,device='cpu'),
            binding_sha256=arm,binding=dict(comparator_provenance={},environment={},source_sha256={}),
            parts=dict(validation=rows),baselines=baselines,evidence=dict(split_hashes={}))
        contexts.append(ctx)
        bests[arm]=dict(epoch=2,checkpoint='m002.pt',sha256=a.file_hash(run/'m002.pt'),predictions='v002.jsonl',
                       validation_metrics=a.metrics(rows,candidate))
    ctx,peer=contexts
    monkeypatch.setattr(e,'selected',lambda c:bests[c['args'].arm])
    monkeypatch.setattr(e,'peer_context',lambda c:peer)
    e.validation(ctx)
    report=a.read(ctx['run']/'validation/report.json')
    assert report['validation_passes_release_gate'] is True
    assert report['test_used'] is False and len(report['models'])==5
    assert report['models']['mdiff_arabic_raw']['cer']==0
    seal=a.read(ctx['run']/'validation/gate.json')
    assert seal['validation_report_sha256']==a.file_hash(ctx['run']/'validation/report.json')
    assert len(a.jsonl(ctx['run']/'validation/predictions.jsonl'))==10
    e.stats(ctx)
    with pytest.raises(FileExistsError): e.validation(ctx)
    (ctx['run']/'validation/predictions.jsonl').write_text('tampered',encoding='utf-8')
    with pytest.raises(ValueError,match='modified'): e.stats(ctx)


def test_geometry_protocol_is_train_only_deterministic_and_has_distinct_labels():
    rows=[dict(sample_id=f'sample{i}',text=f'word{i}',width=65+i%4,height=32) for i in range(8)]
    rows += [dict(sample_id=f'other{i}',text=f'long{i}',width=100,height=32) for i in range(4)]
    picked,manifest=a.tiny_examples(rows,a.GEOMETRY_PROTOCOL)
    shuffled=list(reversed(rows))
    again,other=a.tiny_examples(shuffled,a.GEOMETRY_PROTOCOL)
    assert [r['sample_id'] for r in picked]==[r['sample_id'] for r in again]
    assert manifest==other
    assert len({r['text'] for r in picked})==4
    assert manifest['encoder_input_width']==68
    assert manifest['thresholds_changed'] is False
    assert all((r['width']+3)//4==17 for r in picked)
    with pytest.raises(ValueError,match='four distinct'): a.tiny_examples(rows[:3],a.GEOMETRY_PROTOCOL)


def test_geometry_protocol_blanks_are_identical_and_geometry_mismatch_is_rejected(tmp_path):
    from PIL import Image
    rows=[]
    for i in range(4):
        p=tmp_path/f'i{i}.png';Image.new('RGB',(65+i,32),(i*40,20,255)).save(p)
        rows.append(dict(sample_id=f's{i}',text=f'word{i}',image_path=p.name,width=65+i,height=32))
    tokenizer=CharacterTokenizer(sorted(set(''.join(r['text'] for r in rows))))
    ctx=dict(args=SimpleNamespace(drive=tmp_path.parent),tokenizer=tokenizer,
             image_config=a.ImageConfig(height=32,max_width=640,patch_size=4),
             evidence=dict(tiny_diagnostic_protocol=dict(version=a.GEOMETRY_PROTOCOL)))
    # The dataset helper normally resolves drive/ground_truth_lines.
    from unittest.mock import patch
    original=e.batch
    with patch.object(e,'batch',side_effect=lambda rows,dataset,*args,**kw:original(rows,tmp_path,*args,**kw)):
        result=e.verify_diagnostic_geometry(ctx,rows)
        assert result['blank_encoder_inputs_identical'] is True
        assert result['encoder_input_shape']==[1,3,32,68]
        Image.new('RGB',(69,32),'white').save(tmp_path/'i3.png')
        with pytest.raises(ValueError,match='invariant failed'): e.verify_diagnostic_geometry(ctx,rows)


def test_new_runner_routes_explicit_protocol_and_expected_failure_is_concise(monkeypatch,capsys):
    import thesis.scripts.run_mdiff_arabic_v1 as cli
    argv=['--action','Overfit','--project',str(Path.cwd()),'--run-name','new1',
          '--readiness-protocol',a.GEOMETRY_PROTOCOL]
    args=cli.parse_args(argv)
    assert args.runner.name=='run_mdiff_arabic_v1_1.ps1'
    monkeypatch.setattr(e,'context',lambda args:None)
    def reject(ctx): raise e.ReadinessFailure('better_than_blank; report.json')
    monkeypatch.setattr(e,'overfit',reject)
    with pytest.raises(SystemExit) as ex: cli.main(argv)
    assert ex.value.code==2
    assert 'better_than_blank' in capsys.readouterr().err
