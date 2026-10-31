"""CPU/synthetic verification only; never run the real OCR experiments here."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thesis.ar_mdiff_fusion_v2 import logic as l,inference as i,shared as s,workflow as w
from thesis.diffusion.tokenizer import CharacterTokenizer
from thesis.mdiff_arabic_v1.artifacts import digest,file_hash,read,write,write_jsonl
from thesis.mdiff_arabic_v1.model import MASK,PAD,EOS,Recognizer
from thesis.scripts.run_ar_mdiff_fusion_v2 import parse_args,ACTIONS
from thesis.scripts import run_ar_mdiff_fusion_v2 as cli


def pred(sid,text,**kw):
    return dict(sample_id=sid,text=text,status='completed',has_eos=True,length_failure=False,**kw)


def scored(text,origins,a=-1.,d=-1.):
    return dict(text=text,origins=origins,ar_mean_logp=a,md_mean_logp=d)


def rec(sid='a'):
    return dict(sample_id=sid,ar=pred(sid,'ax'),mdiff=pred(sid,'ab'),candidates=[
        scored('ax',['ar'],-1.,-3.),scored('ab',['mdiff'],-2.,-.1),scored('ac',['span'],-.1,-.1)])


@pytest.mark.parametrize('a,b',[('abc','aXbc'),('aXbc','abc'),('abc','axc'),('','abc'),('abc',''),(' أ  ب ',' أب '),('abXdeYg','abZdeWg')])
def test_alignment_represents_insertions_deletions_and_literal_unicode(a,b):
    spans=l.disagreement_spans(a,b)
    assert l.splice(a,b,spans)==b
    for a0,b0,a1,b1 in spans:
        assert 0<=a0<=a1<=len(a) and 0<=b0<=b1<=len(b)
    if a=='abc' and b=='aXbc':assert spans==[(1,1,1,2)]


def test_span_candidates_can_keep_correct_parts_from_both_and_are_bounded():
    pool=l.candidates(pred('a','abXdeYg'),pred('a','abZdeWg'))
    texts={c['text'] for c in pool}
    assert {'abXdeYg','abZdeWg','abZdeYg','abXdeWg'}<=texts
    assert len(pool)<=l.RECIPE['candidate_limit']
    assert all(len(c['text'])<144 for c in pool)
    assert l.candidates(pred('a','same'),pred('a','same'))==[dict(text='same',origins=['ar','mdiff'],spans=[])]
    over=l.candidates(pred('a','a'*143),pred('a','a'*143+'b'))
    assert all(len(c['text'])+1<=144 for c in over)


def test_proposals_reject_failed_candidates_but_retain_empty_valid_lines():
    bad=pred('a','xx',seconds=1);bad.update(status='failed',has_eos=False)
    assert l.candidates(pred('a',''),bad)==[dict(text='',origins=['ar'],spans=[])]
    assert l.candidates(bad,bad)==[]


def test_method_one_excludes_splices_method_two_can_choose_them_and_ties_keep_ar():
    r=rec();cfg=dict(alpha=.5,margin=0.)
    assert l.choose(r,'dual_rescore',cfg)['text']=='ab'
    assert l.choose(r,'span_merge',cfg)['text']=='ac'
    r['candidates'][1].update(ar_mean_logp=-1.,md_mean_logp=-3.)
    assert l.choose(r,'dual_rescore',cfg)['text']=='ax'
    assert l.choose(rec(),'span_merge',dict(alpha=.5,margin=100))['text']=='ax'
    gain=l.choose(rec(),'dual_rescore',cfg)['score_gain']
    assert l.choose(rec(),'dual_rescore',dict(alpha=.5,margin=gain))['text']=='ax'  # strict > margin


def test_selection_ignores_references_and_invalid_scores_are_not_silently_used():
    r=rec();expected=l.choose(r,'span_merge',dict(alpha=.5,margin=0))
    r.update(text='REFERENCE MUST NOT MATTER',reference_literal='other')
    assert l.choose(r,'span_merge',dict(alpha=.5,margin=0))==expected
    r['candidates'][0]['md_mean_logp']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):l.choose(r,'span_merge',dict(alpha=.5,margin=0))


def test_audit_denominators_and_exact_damage_do_not_hide_behind_equal_total():
    rows=[dict(sample_id='a',text='ab'),dict(sample_id='b',text='cd')]
    ar={'a':pred('a','ab'),'b':pred('b','xx')};after={'a':pred('a','ax'),'b':pred('b','cd')}
    result=l.evaluate(rows,ar,after)
    assert result['changes_vs_ar']['exact_lines_lost']==result['changes_vs_ar']['exact_lines_gained']==1
    assert not result['passes_preservation_check']
    subset=l.evaluate(rows[:1],ar,after)
    assert subset['metrics']['lines']==1 and subset['metrics']['reference_chars']==2


def test_calibration_uses_only_its_subset_and_rejects_duplicate_records():
    rows=[dict(sample_id='a',text='ab')];records=[rec('a'),rec('audit')]
    first=l.calibrate(rows,records,'dual_rescore')
    changed=copy.deepcopy(records);changed[1]['candidates']=[];changed[1]['reference']='x'
    assert l.calibrate(rows,changed,'dual_rescore')==first
    with pytest.raises(ValueError,match='membership'):l.calibrate(rows,records+[rec('a')],'dual_rescore')


class ToyDecoder:
    capacity=6
    def __init__(self):self.calls=[]
    def decode_logits(self,tokens,memory,mm,t,causal=False):
        self.calls.append((tokens.clone(),t.clone(),causal))
        logits=torch.zeros((*tokens.shape,6))
        logits[...,0:2]=-10000.
        logits[...,2:]=torch.tensor([-.5,.1,.4,-.2])
        return logits


def test_md_pll_masks_each_scored_token_including_eos_and_hides_tail():
    tok=CharacterTokenizer(list('abc'));model=ToyDecoder();memory=torch.zeros(1,2,8);mm=torch.zeros(1,2,dtype=torch.bool)
    result=i.md_score(model,memory,mm,tok,'ab',batch_size=2)
    all_inputs=torch.cat([c[0] for c in model.calls])
    visible=torch.tensor([tok.ids['a'],tok.ids['b'],EOS,MASK,MASK,MASK])
    for pos,tokens in enumerate(all_inputs):
        expected=visible.clone();expected[pos]=MASK
        assert torch.equal(tokens,expected)
    assert all(torch.allclose(c[1],torch.full_like(c[1],1/3)) for c in model.calls)
    assert all(not c[2] for c in model.calls)
    other=i.md_score(ToyDecoder(),memory,mm,tok,'ab',batch_size=1)
    assert result['md_mean_logp']==pytest.approx(other['md_mean_logp'])
    assert result['md_sum_logp']==pytest.approx(result['md_mean_logp']*3)


def test_ar_scoring_shift_and_eos_denominator():
    tok=CharacterTokenizer(list('abc'));model=ToyDecoder();memory=torch.zeros(1,2,8);mm=torch.zeros(1,2,dtype=torch.bool)
    result=i.ar_score(model,memory,mm,tok,'ab')
    assert model.calls[0][0][0].tolist()==[MASK,tok.ids['a'],tok.ids['b'],EOS,PAD,PAD]
    assert model.calls[0][2] is True and model.calls[0][1].item()==0
    assert result['active_tokens']==3
    assert result['ar_sum_logp']==pytest.approx(3*result['ar_mean_logp'])
    empty=i.ar_score(ToyDecoder(),memory,mm,tok,'')
    assert empty['active_tokens']==1


def test_cached_memory_sampler_matches_original_sampler_without_reencoding():
    class Stub:
        capacity=6;arm='mdiff';training=False
        hide_tail=staticmethod(Recognizer.hide_tail)
        generate=Recognizer.generate
        def __init__(self):self.decoder_passes=0;self.encodes=0
        def encode(self,pixels,widths):
            self.encodes+=1
            return torch.zeros(1,2,8),torch.zeros(1,2,dtype=torch.bool)
        def decode_logits(self,tokens,memory,mm,t,**kw):
            self.decoder_passes+=1
            logits=torch.zeros((*tokens.shape,6));targets=torch.tensor([3,4,EOS,EOS,EOS,EOS])
            logits.scatter_(2,targets[None,:,None],5.)
            return logits
    model=Stub();original,info=model.generate(torch.zeros(1,3,2,2),torch.tensor([2]))
    m,mm=model.encode(None,None);count=model.encodes
    cached=i.generate_memory(model,m,mm)
    assert torch.equal(original,cached) and info['decoder_passes']==6
    assert model.encodes==count


def test_shared_backbone_is_frozen_but_adapter_learns(monkeypatch):
    def init(self,vocab_size,arm,**kw):
        nn.Module.__init__(self);self.arm=arm;self.capacity=6
        self.encoder=nn.Linear(4,8);self.visual=nn.Linear(8,8);self.embedding=nn.Embedding(vocab_size,8)
    def encode(self,pixels,widths):
        return self.visual(self.encoder(pixels))[:,None,:],torch.zeros(len(pixels),1,dtype=torch.bool)
    monkeypatch.setattr(Recognizer,'__init__',init);monkeypatch.setattr(Recognizer,'encode',encode)
    model=s.SharedRecognizer(6).train();before=s.backbone_hash(model)
    assert not model.encoder.training and not model.visual.training
    assert all(not p.requires_grad for module in (model.encoder,model.visual) for p in module.parameters())
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
    memory,_=model.encode(torch.ones(2,4),torch.ones(2,dtype=torch.long))
    memory.square().sum().backward()
    assert model.adapter[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.encoder.parameters())
    opt.step();assert s.backbone_hash(model)==before
    assert model.adapter[-1].weight.abs().sum()>0


def test_checkpoint_cleanup_protects_selected_and_external_files(tmp_path):
    folder=tmp_path/'shared';folder.mkdir();(folder/'state.pt').write_bytes(b'old')
    (folder/'model.pt').write_bytes(b'best');external=tmp_path/'model.pt';external.write_bytes(b'external')
    s.remove_obsolete_checkpoint(folder,'state.pt',['model.pt'])
    assert not (folder/'state.pt').exists()
    s.remove_obsolete_checkpoint(folder,'model.pt',['model.pt'])
    assert (folder/'model.pt').exists()
    with pytest.raises(ValueError,match='outside'):s.remove_obsolete_checkpoint(folder,'../model.pt',[])
    assert external.read_bytes()==b'external'


def test_shared_training_interrupt_resume_preserves_backbone_and_latest_checkpoint(monkeypatch,tmp_path):
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=nn.Linear(2,2);self.visual=nn.Linear(2,2);self.head=nn.Parameter(torch.ones(()))
            for module in (self.encoder,self.visual):
                for p in module.parameters():p.requires_grad_(False)
        def loss(self,px,widths,target):return (self.head*px.sum()-target.sum()).square(),{}
    monkeypatch.setattr(s,'seed_all',lambda seed:torch.manual_seed(seed))
    def initialize(ctx):
        torch.manual_seed(54101);model=Toy();return model,s.backbone_hash(model)
    monkeypatch.setattr(s,'initialized',initialize)
    calls=[];interrupt=[True]
    def batch(*a,**kw):
        calls.append(1)
        if interrupt[0] and len(calls)==3:raise KeyboardInterrupt()
        return torch.ones(1,2),torch.ones(1,dtype=torch.long),torch.ones(1,1)
    monkeypatch.setattr(s,'batch',batch)
    monkeypatch.setattr(s,'predict_shared',lambda model,rows,ctx:{r['sample_id']:pred(r['sample_id'],'ab') for r in rows})
    rows=[dict(sample_id=str(j),text='ab') for j in range(2)]
    ctx=dict(run=tmp_path/'run',args=SimpleNamespace(device='cpu',drive=tmp_path,resume=False),
        binding_sha256='fixture',binding=dict(shared_recipe=dict(l.RECIPE['shared'],maximum_epochs=3,patience=3,accumulation=2)),
        parts=dict(train=rows,validation=rows),tokenizer=None,image_config=None)
    with pytest.raises(KeyboardInterrupt):s.train(ctx)
    folder=ctx['run']/'shared';assert read(folder/'last.json')['epoch']==1
    assert not (folder/'complete.json').exists()
    ctx['args'].resume=True;interrupt[0]=False;s.train(ctx)
    last=read(folder/'last.json');complete=read(folder/'complete.json')
    assert last['epoch']==3 and complete['best']['epoch']==1
    assert len(list(folder.glob('e*/state.pt')))==1
    assert len(list(folder.glob('e*/model.pt')))==1
    state=torch.load(folder/last['path'],weights_only=True)
    model,_=initialize(ctx);model.load_state_dict(state['model'])
    assert s.backbone_hash(model)==complete['backbone_sha256']
    assert state['updates']==3 and (folder/complete['best']['checkpoint']).exists()


def fixture_ctx(tmp_path,n=3):
    rows=[dict(sample_id=str(j),text='ab',image_path=f'{j}.png',image_sha256=f'im{j}',pixels_sha256=f'px{j}',width=10,height=32) for j in range(n)]
    return dict(run=tmp_path/'run',args=SimpleNamespace(device='cpu',drive=tmp_path,action='Prepare',resume=False),
        binding_sha256='binding',binding={},parts=dict(validation=rows),tokenizer=CharacterTokenizer(list('abx')),
        image_config=None,ar={r['sample_id']:pred(r['sample_id'],'ax') for r in rows},
        mdiff={r['sample_id']:pred(r['sample_id'],'ab') for r in rows},ar_path=tmp_path/'ar.pt',md_path=tmp_path/'md.pt')


def mock_scoring(monkeypatch,ctx):
    monkeypatch.setattr(w,'verify_check',lambda c:{})
    monkeypatch.setattr(i,'load_primary',lambda *a:SimpleNamespace(encoder='encoder',visual='visual'))
    calls=[]
    def record(row,ar,md,*args,shared=False,**kw):
        assert 'text' not in row and 'label_path' not in row and 'reference_literal' not in ar
        calls.append((row['sample_id'],shared));result=rec(row['sample_id'])
        result.update(ar=ar,seconds=.1,encoder_calls=1 if shared else 2,peak_cuda_bytes=None)
        if shared:result['candidates']=[c for c in result['candidates'] if 'span' not in c['origins']]
        return result
    monkeypatch.setattr(i,'record',record)
    return calls,record


def test_preparation_resumes_finished_lines_and_detects_modified_cache(monkeypatch,tmp_path):
    ctx=fixture_ctx(tmp_path);calls,record=mock_scoring(monkeypatch,ctx)
    def interrupted(row,*a,**kw):
        if row['sample_id']=='1':raise KeyboardInterrupt()
        return record(row,*a,**kw)
    monkeypatch.setattr(i,'record',interrupted)
    with pytest.raises(KeyboardInterrupt):w.prepare(ctx)
    monkeypatch.setattr(i,'record',record);rows,records,manifest=w.prepare(ctx)
    assert [sid for sid,_ in calls].count('0')==1
    assert len(records)==3 and read(manifest)['lines']==3
    before=len(calls);w.prepare(ctx);assert len(calls)==before
    line=manifest.parent/'0000.json';value=read(line);value['payload']['ar']['text']='tampered'
    write(line,value)
    with pytest.raises(ValueError,match='sealed line cache'):w.prepare(ctx)


def test_calibration_audit_isolation_and_sealed_settings(monkeypatch,tmp_path):
    ctx=fixture_ctx(tmp_path,6);mock_scoring(monkeypatch,ctx)
    ctx['roles']=dict(sample_roles={str(j):'audit' if j>=4 else 'fit' for j in range(6)})
    w.prepare(ctx);seen=[];original=w.calibrate
    def calibration(rows,*a,**kw):seen.append({r['sample_id'] for r in rows});return original(rows,*a,**kw)
    monkeypatch.setattr(w,'calibrate',calibration);w.calibrate_frozen(ctx)
    assert seen==[{'0','1','2','3'}]*2
    folder=ctx['run']/'calibration/frozen';report=w.verify_published(folder,ctx)
    assert report['audit_sample_ids']==['4','5'] and report['calibration_lines']==4
    assert not report['test_used']
    write(folder/'selection.json',{'changed':True})
    with pytest.raises(ValueError,match='dependency'):w.verify_published(folder,ctx)


def test_test_references_cannot_be_opened_by_prepare_or_calibration(tmp_path):
    ctx=fixture_ctx(tmp_path)
    with pytest.raises(ValueError,match='explicit Test action'):w.test_inputs(ctx)


def test_check_refuses_changed_code_and_checks_source_snapshot(tmp_path):
    src=tmp_path/'src.py';src.write_text('original',encoding='utf-8')
    ctx=dict(run=tmp_path/'run',binding=dict(source_sha256={str(src):file_hash(src)},shared_recipe={}),
        binding_sha256='binding',roles=dict(line_counts={}))
    w.check(ctx);w.verify_check(ctx)
    changed=copy.deepcopy(ctx);changed['binding']['new']='change'
    with pytest.raises(ValueError,match='experiment binding'):w.verify_check(changed)
    (ctx['run']/'source/00_src.py').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='source copy'):w.verify_check(ctx)


def test_atomic_publication_restarts_after_interrupted_pending_jsonl(monkeypatch,tmp_path):
    target=tmp_path/'predictions.jsonl';canonical=w.canonical;calls=[]
    def interrupted(value):
        calls.append(1)
        if len(calls)==2:raise KeyboardInterrupt()
        return canonical(value)
    monkeypatch.setattr(w,'canonical',interrupted)
    with pytest.raises(KeyboardInterrupt):w.once_jsonl(target,[{'a':1},{'a':2}])
    assert not target.exists() and target.with_suffix('.jsonl.pending').exists()
    monkeypatch.setattr(w,'canonical',canonical);w.once_jsonl(target,[{'a':1},{'a':2}])
    assert w.jsonl(target)==[{'a':1},{'a':2}]


def test_run_lock_releases_after_interrupt_and_excludes_another_process(tmp_path):
    with w.run_lock(tmp_path):
        with pytest.raises(RuntimeError,match='Another process'):
            with w.run_lock(tmp_path):pass
    with pytest.raises(KeyboardInterrupt):
        with w.run_lock(tmp_path):raise KeyboardInterrupt()
    with w.run_lock(tmp_path):pass


def test_all_dispatches_steps_in_order_and_test_authorization_is_last(monkeypatch,tmp_path):
    seen=[];context=[]
    def make_context(args):
        ctx=dict(args=args,run=tmp_path/'run');context.append(ctx);return ctx
    monkeypatch.setattr(w,'context',make_context)
    sequence=[('check','Check'),('prepare_frozen','Prepare'),('calibrate_frozen','Calibrate'),
        ('train_shared','TrainShared'),('prepare_shared','PrepareShared'),('calibrate_shared','CalibrateShared'),('historical_test','Test')]
    for function,action in sequence:
        def step(ctx,action=action):
            assert ctx['args'].action==action;seen.append(action)
        monkeypatch.setattr(w,function,step)
    monkeypatch.setattr(sys,'argv',['entry','--project',str(tmp_path),'--action','All','--resume'])
    cli.main();assert seen==[action for _,action in sequence]
    assert context[0]['args'].resume is False  # Fresh training despite All -Resume.


def test_all_three_publish_exactly_752_synthetic_lines_without_real_models(monkeypatch,tmp_path):
    ctx=fixture_ctx(tmp_path,150);calls,_=mock_scoring(monkeypatch,ctx)
    ctx['roles']=dict(sample_roles={str(j):'audit' if j>=120 else 'fit' for j in range(150)})
    folder=ctx['run']/'shared';folder.mkdir(parents=True)
    (folder/'model.pt').write_bytes(b'SYNTHETIC FIXTURE ONLY')
    write_jsonl(folder/'validation.jsonl',[pred(str(j),'ab') for j in range(150)])
    best=dict(epoch=1,checkpoint='model.pt',sha256=file_hash(folder/'model.pt'),predictions='validation.jsonl',
        predictions_sha256=file_hash(folder/'validation.jsonl'))
    complete=dict(binding_sha256='binding',best=best);write(folder/'complete.json',complete)
    monkeypatch.setattr(s,'load_best',lambda c:(SimpleNamespace(encoder=None,visual=None),best,complete))
    w.prepare_frozen(ctx);w.calibrate_frozen(ctx);w.prepare_shared(ctx);w.calibrate_shared(ctx)
    rows=[dict(sample_id=f't{j}',text='ab',image_path=f't{j}.png',image_sha256=f'tim{j}',width=10,height=32) for j in range(752)]
    base={name:{r['sample_id']:pred(r['sample_id'],'ax') for r in rows}
        for name in ('paddle','frozen_v2_2','span_hybrid_verifier_v4','transformer_arabic_control','mdiff_arabic_raw')}
    monkeypatch.setattr(w,'test_inputs',lambda c:(rows,base));ctx['args'].action='Test'
    w.historical_test(ctx)
    out=ctx['run']/'historical_752';report=w.verify_published(out,ctx)
    assert set(report['configs'])=={'dual_rescore','span_merge','shared_encoder'}
    assert all(m['lines']==752 for m in report['models'].values())
    assert set(report['results'])=={'dual_rescore','span_merge','shared_encoder'}
    assert report['promotion_performed'] is False and report['original_gates_modified'] is False
    assert report['timing']['frozen_pool_seconds']==pytest.approx(75.2)
    before=len(calls);w.historical_test(ctx);assert len(calls)==before
    assert len([1 for _,is_shared in calls if is_shared])==150+752


@pytest.mark.parametrize('action',ACTIONS)
def test_cli_dry_run_all_actions_has_no_experiment_side_effects(tmp_path,action):
    entry=Path(__file__).resolve().parents[1]/'thesis/scripts/run_ar_mdiff_fusion_v2.py'
    result=subprocess.run([sys.executable,str(entry),'--project',str(tmp_path),'--action',action,'--dry-run'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    payload=json.loads(result.stdout);assert payload['test_lines']==752 and len(payload['methods'])==3
    assert not (tmp_path/'Diffusion').exists()


def test_cli_rejects_invalid_resume_recipe_and_paths(tmp_path):
    common=['--project',str(tmp_path)]
    for extra in (['--action','Test','--resume'],['--action','Check','--run-name','../bad'],['--action','Check','--epochs','2','--patience','3']):
        with pytest.raises(SystemExit):parse_args(common+extra)


@pytest.mark.skipif(sys.platform!='win32',reason='Windows PowerShell runner')
def test_ps51_wrapper_dry_run_and_native_exit_codes(tmp_path):
    ps=Path(r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe')
    runner=Path(__file__).resolve().parents[2]/'Diffusion/run_ar_mdiff_fusion_v2.ps1'
    result=subprocess.run([str(ps),'-NoProfile','-File',str(runner),'-Action','All','-Python',sys.executable,'-DryRun'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['test_lines']==752
    for code in (0,7):
        fake=tmp_path/f'python{code}.cmd';fake.write_text(f'@echo off\necho synthetic-stderr 1>&2\nexit /b {code}\n',encoding='ascii')
        result=subprocess.run([str(ps),'-NoProfile','-File',str(runner),'-Action','Check','-Python',str(fake),'-DryRun'],capture_output=True,text=True)
        assert (result.returncode==0)==(code==0)
        if code:assert 'exit code 7' in result.stderr
