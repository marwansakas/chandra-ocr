"""Bounded diagnostics and explicit, fail-closed experiment lifecycle."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time
import urllib.request

import torch

from .artifacts import *
from .model import Recognizer, load_encoder, tensor_hash, PAD, MASK, EOS


class ReadinessFailure(ValueError):
    """Expected failed diagnostic, with a complete saved report, not a crash."""


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def setup(args):
    """Download pinned official assets only; never mutate installed packages."""
    target=args.drive/'pretrained/mdiff_s_v1'
    target.mkdir(parents=True,exist_ok=True)
    for name,sha in ASSETS.items():
        dest=target/name
        if not dest.exists():
            temp=dest.with_suffix(dest.suffix+'.download')
            url=f'https://huggingface.co/topdu/MDiff4STR/resolve/{HF_REVISION}/mdiff4str_small/{name}'
            print('Downloading pinned asset:',url,flush=True)
            urllib.request.urlretrieve(url,temp)
            if file_hash(temp)!=sha: raise ValueError('Downloaded asset hash mismatch: '+name)
            temp.replace(dest)
        if file_hash(dest)!=sha: raise ValueError('Existing pretrained asset hash differs: '+str(dest))
    print('Pinned assets verified. Installed packages unchanged. Cache:',target)


def make_model(ctx, arm=None):
    seed_all()
    m=Recognizer(len(ctx['tokenizer']),arm=arm or ctx['args'].arm)
    transfer=load_encoder(m.encoder,ctx['asset_dir']/'best.pth')
    m=m.to(ctx['args'].device)
    return m,transfer


def context(args):
    seed_all()
    repo=args.repo
    parts,tokenizer,image_config,evidence=load_data(args.drive,readiness_protocol=args.readiness_protocol)
    asset_dir=args.drive/'pretrained/mdiff_s_v1'
    for name,sha in ASSETS.items():
        if not (asset_dir/name).is_file(): raise ValueError('Run Setup first: missing '+str(asset_dir/name))
        if file_hash(asset_dir/name)!=sha: raise ValueError('Pretrained asset hash mismatch: '+name)
    # Check the checkpoint-specific pair, not a generic configuration example.
    import yaml
    config=yaml.safe_load((asset_dir/'config.yml').read_text(encoding='utf-8'))
    from .model import ENCODER_CONFIG
    ec=config['Architecture']['Encoder']
    for k,v in ENCODER_CONFIG.items():
        if k=='in_channels': continue
        if ec.get(k)!=v: raise ValueError(f'Checkpoint-specific encoder configuration differs: {k}')
    env=environment(args.device)
    sources=source_files(repo,args.runner)
    baselines,comparators=verify_comparators(args.drive,parts['validation'],'validation')
    binding=dict(version=VERSION,run_name=args.run_name,peer_run_name=args.peer_run_name,arm=args.arm,
        readiness_protocol=args.readiness_protocol,
        recipe=RECIPE,overfit_criterion=OVERFIT_CRITERION,environment=env,source_sha256=sources,
        assets=ASSETS,upstream_commit=UPSTREAM_COMMIT,hf_revision=HF_REVISION,
        data=evidence,comparator_provenance=comparators)
    return dict(args=args,parts=parts,tokenizer=tokenizer,image_config=image_config,evidence=evidence,
        binding=binding,binding_sha256=digest(binding),asset_dir=asset_dir,baselines=baselines,
        run=args.drive/'runs'/args.run_name,checks=args.drive/'checks'/args.run_name)


def verify_check(ctx):
    frozen=read(ctx['checks']/'check.json')
    require_equal(ctx['binding']['environment'],frozen['binding']['environment'],'environment')
    require_equal(ctx['binding'],frozen['binding'],'experiment binding')
    if frozen['binding_sha256']!=ctx['binding_sha256']:
        raise ValueError('Check binding hash differs')
    if frozen.get('passed') is not True: raise ValueError('Check did not pass')
    return frozen


def check(ctx):
    if ctx['run'].exists(): raise FileExistsError('Run name already used: '+str(ctx['run']))
    path=ctx['checks']/'check.json'
    if path.exists():
        verify_check(ctx)
        print('Existing immutable Check matches:',path)
        return
    if ctx['evidence']['coverage_issues']:
        write(ctx['checks']/('check_issues_'+stamp()+'.json'),ctx['evidence']['coverage_issues'],exclusive=True)
        raise ValueError('Coverage issues; full details saved. No labels truncated or examples removed.')
    model,transfer=make_model(ctx)
    initial=tensor_hash(model.state_dict())
    peer,_=make_model(ctx,'ar' if ctx['args'].arm=='mdiff' else 'mdiff')
    if tensor_hash(peer.state_dict())!=initial: raise ValueError('Primary arms do not share exact initial parameters')
    del peer
    payload=dict(binding=ctx['binding'],binding_sha256=ctx['binding_sha256'],passed=True,
        transfer=transfer,model_initialization_sha256=initial,parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),test_used=False)
    # Snapshot dirty/untracked executable sources too; Git HEAD alone is insufficient.
    snapshot=ctx['checks']/'source'
    snapshot.mkdir(parents=True,exist_ok=False)
    for i,(source,sha) in enumerate(ctx['binding']['source_sha256'].items()):
        dest=snapshot/f'{i:02d}_{Path(source).name}'
        shutil.copyfile(source,dest)
        if file_hash(dest)!=sha: raise ValueError('Source snapshot copy differs')
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ctx['args'].repo,capture_output=True,text=True,check=True).stdout.strip()
    payload['git_revision']=git
    write(ctx['checks']/'source_manifest.json',ctx['binding']['source_sha256'],exclusive=True)
    write(ctx['checks']/'environment.json',ctx['binding']['environment'],exclusive=True)
    packages=ctx['binding']['environment']['packages']
    (ctx['checks']/'dependencies.freeze.txt').write_text(''.join(f'{k}=={v}\n' for k,v in packages.items()),encoding='utf-8')
    ctx['tokenizer'].save(ctx['checks']/'tokenizer.json')
    write(path,payload,exclusive=True)
    print(f"Check passed | arm={ctx['args'].arm} | counts={ctx['evidence']['counts']} | GPU={ctx['binding']['environment']['gpu']} | VRAM={ctx['binding']['environment']['gpu_total_bytes']} bytes")
    print('Encoder initialization:',transfer['encoder_initialization_sha256'],'| parameters:',payload['parameter_count'])
    print('Coverage issues: 0 | Source snapshot includes uncommitted edits | Test used: false')
    print('Check:',path)


def subset(ctx):
    ids=set(ctx['evidence']['tiny_sample_ids'])
    return [r for r in ctx['parts']['train'] if r['sample_id'] in ids]


def verify_diagnostic_geometry(ctx,rows):
    protocol=ctx['evidence']['tiny_diagnostic_protocol']
    if protocol['version']!='geometry_matched_v1':
        return dict(protocol='legacy_v1',blank_encoder_inputs_identical=False)
    encoded=[]
    for r in rows:
        px,widths,_=batch([r],ctx['args'].drive/'ground_truth_lines',ctx['tokenizer'],ctx['image_config'],'cpu')
        aligned=math.ceil(int(widths[0])/4)*4
        encoded.append(torch.ones_like(px[:,:,:,:aligned]))
    identical=bool(encoded) and all(torch.equal(x,encoded[0]) for x in encoded[1:])
    distinct=len({training_text(r['text'],'no_harakat_v1') for r in rows})==len(rows)==4
    if not identical or not distinct: raise ValueError('Geometry-matched readiness invariant failed')
    return dict(protocol=protocol['version'],blank_encoder_inputs_identical=identical,
        distinct_labels=len(rows),blank_input_sha256=tensor_hash({'blank':encoded[0]}),
        encoder_input_shape=list(encoded[0].shape),selection_manifest=protocol)


def autocast(device):
    return torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda')


def synchronize(device):
    if device=='cuda': torch.cuda.synchronize()


def predict(model, rows, ctx, *, image_mode='normal', initial_mode=None, steps=3, deadline=None):
    model.eval()
    device=ctx['args'].device
    results=[]
    for i,r in enumerate(rows):
        source=rows[(i+1)%len(rows)] if image_mode=='shuffled' else r
        started=time.perf_counter()
        try:
            if deadline is not None and started>=deadline:
                raise TimeoutError('Strict overfit diagnostic time budget exhausted')
            px,w,_=batch([source],ctx['args'].drive/'ground_truth_lines',ctx['tokenizer'],ctx['image_config'],device)
            initial=None
            if image_mode=='blank': px=torch.ones_like(px)
            if initial_mode:
                _,_,target=batch([r],ctx['args'].drive/'ground_truth_lines',ctx['tokenizer'],ctx['image_config'],device)
                initial=target.clone()
                chars=target[0].ge(3).nonzero().flatten()
                index=int(chars[len(chars)//2])
                if initial_mode=='missing': initial[0,index]=MASK
                elif initial_mode=='replaced': initial[0,index]=3+(int(target[0,index])-3+1)%(len(ctx['tokenizer'])-3)
                elif initial_mode!='correct': raise ValueError('Unknown controlled diagnostic')
            with autocast(device): ids,info=model.generate(px,w,steps=steps,initial=initial)
            text,has_eos=ctx['tokenizer'].decode(ids[0].tolist())
            synchronize(device)
            result=dict(sample_id=r['sample_id'],text=text,has_eos=has_eos,status='completed' if has_eos else 'failed',
                length_failure=not has_eos,seconds=time.perf_counter()-started,decoder_passes=info['decoder_passes'],
                sampler_mode=info['sampler_mode'],sampler_steps=info['sampler_steps'],error=None)
        except Exception as error:
            # Keep failed examples, reference denominator and explicit error.
            result=dict(sample_id=r['sample_id'],text='',has_eos=False,status='failed',length_failure=False,
                        seconds=time.perf_counter()-started,error=f'{type(error).__name__}: {error}')
            if device=='cuda': torch.cuda.empty_cache()
        results.append(result)
    return indexed(results,rows)


def smoke(ctx):
    frozen=verify_check(ctx)
    model,transfer=make_model(ctx)
    if transfer['encoder_initialization_sha256']!=frozen['transfer']['encoder_initialization_sha256']:
        raise ValueError('Smoke initialization differs')
    device=ctx['args'].device
    # A real mixed-length batch; encoder sees each unpadded line and decoder masks memory padding.
    rows=subset(ctx)[:2]
    diagnostic_geometry=verify_diagnostic_geometry(ctx,subset(ctx))
    px,w,target=batch(rows,ctx['args'].drive/'ground_truth_lines',ctx['tokenizer'],ctx['image_config'],device)
    for text in [' لغه  عربية! ١٢\n','ـأإآ، 9.1',*[training_text(r['text'],'no_harakat_v1') for r in rows]]:
        # Synthetic probes use existing approved training characters where possible;
        # OOV rejection itself is tested by the unit suite.
        if set(text)<=set(ctx['tokenizer'].characters):
            actual,eos=ctx['tokenizer'].decode(ctx['tokenizer'].encode(text,144,eos_fill=False))
            if actual!=text or not eos: raise ValueError('Tokenizer literal roundtrip failed')
    model.train()
    with autocast(device): loss,parts=model.loss(px,w,target)
    if not torch.isfinite(loss): raise ValueError('Nonfinite real minibatch loss')
    loss.backward()
    gradients={name:sum(float(p.grad.detach().float().abs().sum()) for p in module.parameters() if p.grad is not None)
               for name,module in [('encoder',model.encoder),('visual_projection',model.visual),('decoder',model.decoder),
                                   ('embedding',model.embedding),('output',model.output),('time',model.time)]}
    if any(not math.isfinite(v) or v<=0 for v in gradients.values()):
        raise ValueError('Missing/nonfinite intended module gradients: '+str(gradients))
    widest=max(ctx['parts']['train'],key=lambda r:r['width']/r['height'])
    model.zero_grad(set_to_none=True)
    largest_px,largest_w,largest_target=batch([widest],ctx['args'].drive/'ground_truth_lines',ctx['tokenizer'],ctx['image_config'],device)
    with autocast(device): largest_loss,_=model.loss(largest_px,largest_w,largest_target)
    if not torch.isfinite(largest_loss): raise ValueError('Nonfinite widest-TRAIN minibatch loss')
    largest_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    model.eval()
    with autocast(device):
        first,info=model.generate(px,w,steps=3,trace=True)
        second,_=model.generate(px,w,steps=3)
    if not torch.equal(first,second): raise ValueError('Deterministic repeated decoding differs')
    if model.arm=='mdiff' and (info['decoder_passes']!=6 or [x['kind'] for x in info['trace']]!=['denoise','correction']*3):
        raise ValueError('Iterative path did not execute all denoise/correction passes')
    path=ctx['checks']/'smoke'/stamp()/'report.json'
    report=dict(passed=True,binding_sha256=ctx['binding_sha256'],check_sha256=file_hash(ctx['checks']/'check.json'),
        real_minibatch_sample_ids=[r['sample_id'] for r in rows],loss=float(loss.detach()),loss_terms=parts,
        gradients=gradients,deterministic_repeat=True,inference=info,test_used=False,
        diagnostic_geometry=diagnostic_geometry,
        widest_training_image=dict(sample_id=widest['sample_id'],resized_width=int(largest_w[0]),loss=float(largest_loss.detach()),finite_backward=True),
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    write(path,report,exclusive=True)
    write(ctx['checks']/'smoke_latest.json',dict(report=str(path),sha256=file_hash(path)))
    print('Smoke passed | finite real minibatch gradients:',gradients)
    print('Decoder passes:',info['decoder_passes'],'| deterministic repeat: true | Test used: false')
    print('Report:',path)


def diagnostics(model,rows,ctx,deadline=None):
    predictions={mode:predict(model,rows,ctx,image_mode=mode,deadline=deadline) for mode in ('normal','shuffled','blank')}
    report={mode:metrics(rows,preds) for mode,preds in predictions.items()}
    if model.arm=='mdiff':
        for mode in ('missing','replaced','correct'):
            preds=predict(model,rows,ctx,initial_mode=mode,deadline=deadline)
            predictions[mode]=preds
            report[mode]=metrics(rows,preds)
    return report,predictions


def overfit(ctx):
    verify_check(ctx)
    args=ctx['args']
    rows=subset(ctx)
    diagnostic_geometry=verify_diagnostic_geometry(ctx,rows)
    device=args.device
    seed_all()
    model,_=make_model(ctx)
    # Same optimizer rates as the main recipe; no special huge preservation weights.
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=RECIPE['encoder_lr']),
        dict(params=[p for n,p in model.named_parameters() if not n.startswith('encoder.')],lr=RECIPE['decoder_lr'])],weight_decay=RECIPE['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=False)  # BF16 needs no scaling
    started=time.perf_counter()
    deadline=started+args.overfit_seconds
    # Reserve bounded time for decoded metrics; no teacher-forced-only pass.
    reserve=min(60.,args.overfit_seconds/3)
    completed=0
    failures=[]
    for update in range(args.overfit_updates):
        if time.perf_counter()>=deadline-reserve: break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Four examples cycled, one true microbatch per update; hard update/time cap.
        row=rows[update%len(rows)]
        px,w,target=batch([row],args.drive/'ground_truth_lines',ctx['tokenizer'],ctx['image_config'],device)
        with autocast(device): loss,terms=model.loss(px,w,target)
        if not torch.isfinite(loss):
            failures.append('nonfinite loss');break
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update()
        completed+=1
        if completed%100==0: print(f'Overfit update {completed}/{args.overfit_updates} | loss={float(loss.detach()):.4f}',flush=True)
    report,predictions=diagnostics(model,rows,ctx,deadline)
    criterion=OVERFIT_CRITERION
    normal=report['normal']
    conditions=dict(minimum_updates=completed>=criterion['minimum_updates'],
        normal_cer=normal['cer']<=criterion['cer_at_most'],normal_exact=normal['exact_line_accuracy']>=criterion['exact_at_least'],
        complete_eos=all(m['failed_lines']==0 and m['eos_failures']==0 for m in report.values()),
        better_than_blank=report['blank']['cer']-normal['cer']>=criterion['image_cer_advantage_at_least'],
        better_than_shuffled=report['shuffled']['cer']-normal['cer']>=criterion['image_cer_advantage_at_least'],finite_training=not failures,
        time_budget=time.perf_counter()<=deadline)
    if model.arm=='mdiff':
        conditions.update(missing_recovery=report['missing']['cer']<=criterion['correction_cer_at_most'] and report['missing']['exact_line_accuracy']>=criterion['correction_exact_at_least'],
            replaced_recovery=report['replaced']['cer']<=criterion['correction_cer_at_most'] and report['replaced']['exact_line_accuracy']>=criterion['correction_exact_at_least'],
            correct_state_preservation=report['correct']['exact_line_accuracy']>=criterion['preservation_exact_at_least'])
    directory=ctx['checks']/'overfit'/stamp()
    result=dict(passed=all(conditions.values()),conditions=conditions,criterion=criterion,
        binding_sha256=ctx['binding_sha256'],check_sha256=file_hash(ctx['checks']/'check.json'),
        sample_ids=[r['sample_id'] for r in rows],updates_completed=completed,update_limit=args.overfit_updates,
        time_limit_seconds=args.overfit_seconds,elapsed_seconds=time.perf_counter()-started,metrics=report,failures=failures,test_used=False,
        diagnostic_geometry=diagnostic_geometry,
        correct_input_lines_damaged=(len(rows)-report['correct']['exact_lines']) if model.arm=='mdiff' else None,
        controlled_tokens='training-derived mid-character missing/replacement; correct full-state preservation; diagnostic-only reference tokens never enter image-only production inference',
        interpretation='TRAIN-only dependency check; no guarantee of generalization or proof of visual grounding')
    write(directory/'report.json',result,exclusive=True)
    for mode,values in predictions.items(): write_jsonl(directory/(mode+'.jsonl'),values.values())
    write(ctx['checks']/'overfit_latest.json',dict(report=str(directory/'report.json'),sha256=file_hash(directory/'report.json')))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('Report:',directory/'report.json')
    if not result['passed']:
        failed=', '.join(k for k,v in conditions.items() if not v)
        raise ReadinessFailure(f'Overfit readiness failed: {failed}. Train remains blocked. Report: {directory / "report.json"}')


def prerequisite(ctx,kind):
    pointer=read(ctx['checks']/(kind+'_latest.json'))
    path=Path(pointer['report'])
    if file_hash(path)!=pointer['sha256']: raise ValueError(kind+' readiness report modified')
    report=read(path)
    if report.get('passed') is not True or report['binding_sha256']!=ctx['binding_sha256'] or report['check_sha256']!=file_hash(ctx['checks']/'check.json'):
        raise ValueError(kind+' readiness failed or does not match this experiment')
    return dict(report=str(path),sha256=file_hash(path))


def optimizer_for(model):
    return torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=RECIPE['encoder_lr'],base_lr=RECIPE['encoder_lr']),
        dict(params=[p for n,p in model.named_parameters() if not n.startswith('encoder.')],lr=RECIPE['decoder_lr'],base_lr=RECIPE['decoder_lr'])],weight_decay=RECIPE['weight_decay'])


def save_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    torch.save(value,temp)
    temp.replace(path)


def training_key(metrics,epoch):
    return (metrics['char_distance'],metrics['word_distance'],-metrics['exact_lines'],epoch)


def remaining_epochs(start_epoch,stale):
    # A crash between writing last.json and training_complete.json must not
    # resume past an already satisfied patience/max-epoch stop condition.
    return range(start_epoch,201) if stale<30 else range(0)


def train(ctx):
    frozen=verify_check(ctx)
    args=ctx['args'];run=ctx['run'];device=args.device
    if args.epochs!=200 or args.patience!=30:
        raise ValueError('Frozen primary recipe requires -Epochs 200 -Patience 30; use Overfit for bounded diagnostics')
    ready={k:prerequisite(ctx,k) for k in ('smoke','overfit')}
    if args.resume:
        if (run/'training_complete.json').exists(): raise ValueError('Completed run is immutable; cannot resume')
        record=read(run/'run.json')
        require_equal(record['binding_sha256'],ctx['binding_sha256'],'resume binding')
        require_equal(record['readiness'],ready,'resume readiness artifacts')
        last=read(run/'last.json')
        if file_hash(run/'last.pt')!=last['sha256']: raise ValueError('Interrupted run checkpoint hash mismatch')
    else:
        run.mkdir(parents=True,exist_ok=False)
        shutil.copytree(ctx['checks']/'source',run/'source')
        write(run/'run.json',dict(binding=ctx['binding'],binding_sha256=ctx['binding_sha256'],readiness=ready,
            check_sha256=file_hash(ctx['checks']/'check.json'),encoder_initialization_sha256=frozen['transfer']['encoder_initialization_sha256'],
            model_initialization_sha256=frozen['model_initialization_sha256'],parameter_count=frozen['parameter_count'],created_utc=stamp()),exclusive=True)
        shutil.copyfile(ctx['checks']/'tokenizer.json',run/'tokenizer.json')
        shutil.copyfile(ctx['checks']/'dependencies.freeze.txt',run/'dependencies.freeze.txt')
    model,transfer=make_model(ctx)
    if tensor_hash(model.state_dict())!=frozen['model_initialization_sha256']: raise ValueError('Actual training model initialization mismatch')
    optimizer=optimizer_for(model)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    best_key=None;best=None;stale=0;start_epoch=1;updates=0;train_seconds=0.;val_seconds=0.
    if args.resume:
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if state['binding_sha256']!=ctx['binding_sha256']: raise ValueError('Checkpoint resume binding mismatch')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start_epoch=state['epoch']+1;best_key=tuple(state['best_key']);best=state['best'];stale=state['stale'];updates=state['updates']
        train_seconds=state['train_seconds'];val_seconds=state['val_seconds']
        torch.set_rng_state(state['torch_rng'])
        if device=='cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
        # Epoch transaction: prediction/history files newer than last.json are
        # not reused. Refuse ambiguous partial epoch artifacts rather than append.
        if (run/f'v{start_epoch:03d}.jsonl').exists() or (run/f'e{start_epoch:03d}.json').exists() or (run/f'm{start_epoch:03d}.pt').exists():
            raise ValueError('Uncommitted epoch predictions exist; preserve this run and use a reviewed new run. Automatic replay refused.')
    train_rows=ctx['parts']['train'];validation=ctx['parts']['validation']
    updates_per_epoch=math.ceil(len(train_rows)/RECIPE['accumulation'])
    total_updates=200*updates_per_epoch;warmup=5*updates_per_epoch
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    last_epoch=start_epoch-1
    for epoch in remaining_epochs(start_epoch,stale):
        last_epoch=epoch
        for p in model.encoder.parameters(): p.requires_grad_(epoch>RECIPE['encoder_frozen_epochs'])
        model.train()
        order=torch.randperm(len(train_rows),generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
        began=time.perf_counter();sum_loss=0.;skipped=0
        for start in range(0,len(order),RECIPE['accumulation']):
            indices=order[start:start+RECIPE['accumulation']]
            optimizer.zero_grad(set_to_none=True)
            factor=min(1.,(updates+1)/warmup) if updates<warmup else .1+.9*.5*(1+math.cos(math.pi*(updates-warmup)/max(1,total_updates-warmup)))
            for g in optimizer.param_groups: g['lr']=g['base_lr']*factor
            for index in indices:
                px,w,target=batch([train_rows[index]],args.drive/'ground_truth_lines',ctx['tokenizer'],ctx['image_config'],device,epoch)
                with autocast(device): loss,_=model.loss(px,w,target)
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss; run preserved for diagnosis')
                scaler.scale(loss/len(indices)).backward()
                sum_loss+=float(loss.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            skipped+=scaler.get_scale()<previous_scale
            updates+=1
        train_seconds+=time.perf_counter()-began
        began=time.perf_counter()
        predictions=predict(model,validation,ctx)
        m=metrics(validation,predictions)
        val_seconds+=time.perf_counter()-began
        path=run/f'v{epoch:03d}.jsonl'
        write_jsonl(path,predictions.values())
        key=training_key(m,epoch)
        improved=best_key is None or key<best_key
        stale=0 if improved else stale+1
        checkpoint_path=run/f'm{epoch:03d}.pt'
        if improved or epoch%10==0 or stale>=30 or epoch==200:
            save_torch(checkpoint_path,dict(model=model.state_dict(),epoch=epoch,binding_sha256=ctx['binding_sha256']))
        if improved:
            best_key=key
            best=dict(epoch=epoch,checkpoint=checkpoint_path.name,sha256=file_hash(checkpoint_path),
                predictions=path.name,predictions_sha256=file_hash(path),validation_metrics=m,binding_sha256=ctx['binding_sha256'])
            write(run/'best.json',best)
        history=dict(epoch=epoch,train_loss=sum_loss/len(train_rows),validation=m,best_update=improved,stale_epochs=stale,
            optimizer_updates=updates,skipped_amp_updates=skipped,effective_batch_size=16,last_batch_size=len(order)%16 or 16,
            cumulative_train_seconds=train_seconds,cumulative_validation_seconds=val_seconds,
            training_decoder_passes=2*len(train_rows)*epoch,training_examples=len(train_rows)*epoch,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
        write(run/f'e{epoch:03d}.json',history,exclusive=True)
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),epoch=epoch,
            binding_sha256=ctx['binding_sha256'],best_key=best_key,best=best,stale=stale,updates=updates,
            train_seconds=train_seconds,val_seconds=val_seconds,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [])
        save_torch(run/'last.pt',state)
        write(run/'last.json',dict(epoch=epoch,sha256=file_hash(run/'last.pt'),binding_sha256=ctx['binding_sha256']))
        print(f"epoch={epoch:03d} train_loss={history['train_loss']:.4f} raw_val_CER={100*m['cer']:.3f}% exact={m['exact_lines']}/{len(validation)} failed={m['failed_lines']} best={improved} wait={stale}/30 updates={updates}",flush=True)
        if stale>=30: break
    write(run/'training_complete.json',dict(epoch=last_epoch,best=best,optimizer_updates=updates,
        train_seconds=train_seconds,validation_seconds=val_seconds,binding_sha256=ctx['binding_sha256'],test_used=False),exclusive=True)
    print('Training complete. Selected epoch:',best['epoch'],'| checkpoint:',run/best['checkpoint'],'| Test used: false')


def selected(ctx):
    checked=verify_check(ctx)
    run=ctx['run']
    record=read(run/'run.json')
    require_equal(record['binding_sha256'],ctx['binding_sha256'],'run binding')
    require_equal(record['check_sha256'],file_hash(ctx['checks']/'check.json'),'run check artifact')
    require_equal(record['encoder_initialization_sha256'],checked['transfer']['encoder_initialization_sha256'],'encoder initialization')
    require_equal(record['model_initialization_sha256'],checked['model_initialization_sha256'],'model initialization')
    require_equal(file_hash(run/'tokenizer.json'),file_hash(ctx['checks']/'tokenizer.json'),'tokenizer file')
    require_equal(file_hash(run/'dependencies.freeze.txt'),file_hash(ctx['checks']/'dependencies.freeze.txt'),'dependency lock')
    complete=read(run/'training_complete.json')
    if complete['binding_sha256']!=ctx['binding_sha256']: raise ValueError('Training completion binding differs')
    best=read(run/'best.json')
    require_equal(best,complete['best'],'frozen selection')
    if file_hash(run/best['checkpoint'])!=best['sha256'] or file_hash(run/best['predictions'])!=best['predictions_sha256']:
        raise ValueError('Selected checkpoint/predictions modified')
    histories=[read(p) for p in sorted(run.glob('e[0-9][0-9][0-9].json'))]
    if len(histories)!=complete['epoch'] or [h['epoch'] for h in histories]!=list(range(1,complete['epoch']+1)):
        raise ValueError('Missing per-epoch selection evidence')
    winner=min(histories,key=lambda h:training_key(h['validation'],h['epoch']))
    if winner['epoch']!=best['epoch']: raise ValueError('Selected checkpoint violates frozen selection rule')
    # Validate every epoch's raw denominators, not just copied aggregate fields.
    for history in histories:
        raw=indexed(jsonl(run/f"v{history['epoch']:03d}.jsonl"),ctx['parts']['validation'])
        require_equal(metrics(ctx['parts']['validation'],raw),history['validation'],'epoch validation evidence')
    return best


def peer_context(ctx):
    name=ctx['args'].peer_run_name
    if not name: raise ValueError('-PeerRunName is required for paired validation/test')
    args=copy.copy(ctx['args'])
    args.run_name=name;args.peer_run_name=ctx['args'].run_name
    args.arm='ar' if ctx['args'].arm=='mdiff' else 'mdiff'
    # Same verified data, environment, sources and assets. Only identities/arm differ.
    peer=dict(ctx,args=args,run=args.drive/'runs'/name,checks=args.drive/'checks'/name)
    peer['binding']=dict(ctx['binding'],run_name=name,peer_run_name=args.peer_run_name,arm=args.arm)
    peer['binding_sha256']=digest(peer['binding'])
    own=read(ctx['run']/'run.json');other=read(peer['run']/'run.json')
    for key in ('encoder_initialization_sha256','model_initialization_sha256'):
        if own[key]!=other[key]: raise ValueError('Matched arms initialization differs: '+key)
    return peer


def model_name(arm):
    return 'mdiff_arabic_raw' if arm=='mdiff' else 'transformer_arabic_control'


def display(report,path):
    print('Model                        Lines   CER %   WER %  Exact %   Sub   Ins   Del')
    for name,m in report['models'].items():
        print(f"{name:<28} {m['lines']:5d} {100*m['cer']:7.2f} {100*m['wer']:7.2f} {100*m['exact_line_accuracy']:8.2f} {m['substitutions']:5d} {m['insertions']:5d} {m['deletions']:5d}")
    for name,info in report['selected_models'].items():
        print(f"{name}: selected epoch={info['epoch']} | checkpoint={info['checkpoint_path']} | SHA256={info['sha256']} | Decoder={info['sampler_mode']} | Steps={info['sampler_steps']}")
        result=report['arms'][name]
        d=result['damage']['span_hybrid_verifier_v4']
        print(f"Exact lines lost/gained vs v4: {d['exact_lines_lost']} / {d['exact_lines_gained']} | union damaged={result['baseline_exact_union_lines_lost']}")
        print('Gate conditions:',json.dumps(result['conditions'],sort_keys=True))
    print('Validation gate:',report.get('validation_passes_release_gate'),'| Test used:',str(report['test_used']).lower())
    print('Report:',path)


def report_for(ctx,peer,own_best,peer_best,rows,baselines,predictions,partition,provenance):
    models={**baselines,**predictions}
    chosen={}
    arms={}
    for c,b in ((ctx,own_best),(peer,peer_best)):
        name=model_name(c['args'].arm)
        chosen[name]=dict(epoch=b['epoch'],checkpoint_path=str(c['run']/b['checkpoint']),sha256=b['sha256'],
            sampler_mode='masked_reflect_v1' if c['args'].arm=='mdiff' else 'greedy_causal',
            sampler_steps=3 if c['args'].arm=='mdiff' else None,
            encoder_initialization_sha256=read(c['run']/'run.json')['encoder_initialization_sha256'],
            run_manifest_sha256=file_hash(c['run']/'run.json'),tokenizer_sha256=file_hash(c['run']/'tokenizer.json'),
            dependency_lock_sha256=file_hash(c['run']/'dependencies.freeze.txt'),
            check_sha256=file_hash(c['checks']/'check.json'),training_complete=read(c['run']/'training_complete.json'))
        arms[name]=eligibility(rows,baselines,predictions[name])
        arms[name]['confidence_intervals']=grouped_intervals(rows,predictions[name],baselines['span_hybrid_verifier_v4'])
    gate={f'{name}.{condition}':v for name,a in arms.items() for condition,v in a['conditions'].items()}
    return dict(version=VERSION,partition=partition,models={name:metrics(rows,p) for name,p in models.items()},arms=arms,
        selected_models=chosen,validation_passes_release_gate=all(gate.values()) if partition=='validation' else None,
        gate_conditions=gate if partition=='validation' else {},policy=POLICY,seed=SEED,
        binding_sha256=ctx['binding_sha256'],peer_binding_sha256=peer['binding_sha256'],comparator_provenance=provenance,
        split_hashes=ctx['evidence']['split_hashes'],test_used=partition=='test',
        benchmark_status='exploratory/development-used; no untouched generalization claim',
        environment=ctx['binding']['environment'],code_sha256=ctx['binding']['source_sha256'],
        runtime_note='new model timing includes image loading/preprocessing and synchronized inference; historical comparator timing is unavailable',
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if ctx['args'].device=='cuda' else None)


def joined(rows,models):
    for r in rows:
        for name,preds in models.items():
            p=preds[r['sample_id']]
            yield dict(sample_id=r['sample_id'],group=group(r),image_sha256=r['image_sha256'],
                reference_literal=r['text'],reference_evaluation=training_text(r['text'],'no_harakat_v1'),model=name,
                **{k:v for k,v in p.items() if k!='sample_id'},
                per_line_metrics=score_pair(r['text'],p['text'],no_harakat_v1=True))


def validation(ctx):
    own=selected(ctx);peer=peer_context(ctx);other=selected(peer)
    output=ctx['run']/'validation'
    output.mkdir(parents=True,exist_ok=False)
    rows=ctx['parts']['validation'];baselines=ctx['baselines']
    predictions={model_name(ctx['args'].arm):indexed(jsonl(ctx['run']/own['predictions']),rows),
                 model_name(peer['args'].arm):indexed(jsonl(peer['run']/other['predictions']),rows)}
    # Recompute aggregate scores from the selected raw rows; no inference fallback.
    for c,b in ((ctx,own),(peer,other)):
        require_equal(metrics(rows,predictions[model_name(c['args'].arm)]),b['validation_metrics'],'selected raw metrics')
    report=report_for(ctx,peer,own,other,rows,baselines,predictions,'validation',ctx['binding']['comparator_provenance'])
    write_jsonl(output/'predictions.jsonl',joined(rows,{**baselines,**predictions}))
    report['predictions_sha256']=file_hash(output/'predictions.jsonl')
    report_path=output/'report.json'
    write(report_path,report,exclusive=True)
    seal=dict(binding_sha256=ctx['binding_sha256'],validation_report_sha256=file_hash(report_path),
        selected_checkpoint_sha256=own['sha256'],peer_selected_checkpoint_sha256=other['sha256'],
        predictions_sha256=report['predictions_sha256'],policy_sha256=digest(POLICY),
        validation_passes_release_gate=report['validation_passes_release_gate'])
    write(output/'gate.json',seal,exclusive=True)
    display(report,report_path)
    (output/'report.txt').write_text(readable(report,report_path),encoding='utf-8')


def readable(report,path):
    # Redirect only our display, never native stderr.
    import contextlib,io
    out=io.StringIO()
    with contextlib.redirect_stdout(out): display(report,path)
    return out.getvalue()


def stats(ctx):
    best=selected(ctx);peer=peer_context(ctx);other=selected(peer)
    path=ctx['run']/'validation/report.json'
    report=read(path);seal=read(ctx['run']/'validation/gate.json')
    if seal['validation_report_sha256']!=file_hash(path) or report['binding_sha256']!=ctx['binding_sha256']:
        raise ValueError('Validation report binding differs')
    if file_hash(ctx['run']/'validation/predictions.jsonl')!=report['predictions_sha256']:
        raise ValueError('Validation joined predictions modified')
    if best['sha256']!=seal['selected_checkpoint_sha256'] or other['sha256']!=seal['peer_selected_checkpoint_sha256']:
        raise ValueError('Validation checkpoint selection changed')
    display(report,path)


def test(ctx):
    """Gate verification must precede ANY test reference loading or inference."""
    own=selected(ctx);peer=peer_context(ctx);other=selected(peer)
    path=ctx['run']/'validation/report.json'
    report=read(path);seal=read(ctx['run']/'validation/gate.json')
    enforce_gate(report,seal,expected_binding=ctx['binding_sha256'],report_sha=file_hash(path),
                 checkpoint_sha=own['sha256'],peer_checkpoint_sha=other['sha256'])
    if seal['policy_sha256']!=digest(POLICY) or file_hash(ctx['run']/'validation/predictions.jsonl')!=seal['predictions_sha256']:
        raise ValueError('Frozen gate policy/predictions changed')
    # Recompute every gate condition from frozen raw validation artifacts, so
    # editing a boolean cannot authorize the comparison.
    for c,b in ((ctx,own),(peer,other)):
        preds=indexed(jsonl(c['run']/b['predictions']),ctx['parts']['validation'])
        actual=eligibility(ctx['parts']['validation'],ctx['baselines'],preds)
        if actual['conditions']!=report['arms'][model_name(c['args'].arm)]['conditions'] or not actual['validation_passes_release_gate']:
            raise ValueError('STOP: frozen validation gate did not recompute as passed')
    output=ctx['run']/'historical_test'
    if output.exists(): raise FileExistsError('Historical comparison already exists; never overwrite')
    parts,_,_,evidence=load_data(ctx['args'].drive,test_authorized=True,
                               readiness_protocol=ctx['args'].readiness_protocol)
    test_issues=[x for x in evidence['coverage_issues'] if x['partition']=='test']
    if test_issues:
        # Do not change settings or discard difficult examples; preserve failure evidence.
        write(ctx['run']/('historical_test_coverage_failure_'+stamp()+'.json'),test_issues,exclusive=True)
        raise ValueError('Historical test coverage failure; no inference/settings changes authorized')
    rows=parts['test']
    baselines,provenance=verify_comparators(ctx['args'].drive,rows,'test')
    output.mkdir(parents=True,exist_ok=False)
    predictions={}
    for c,b in ((ctx,own),(peer,other)):
        model,_=make_model(c)
        state=torch.load(c['run']/b['checkpoint'],map_location='cpu',weights_only=True)
        if state['binding_sha256']!=c['binding_sha256'] or state['epoch']!=b['epoch']:
            raise ValueError('Selected checkpoint metadata differs')
        model.load_state_dict(state['model'],strict=True)
        predictions[model_name(c['args'].arm)]=predict(model,rows,c)
        del model
        if c['args'].device=='cuda': torch.cuda.empty_cache()
    result=report_for(ctx,peer,own,other,rows,baselines,predictions,'test',provenance)
    result['authorized_by_gate_sha256']=file_hash(ctx['run']/'validation/gate.json')
    write_jsonl(output/'predictions.jsonl',joined(rows,{**baselines,**predictions}))
    result['predictions_sha256']=file_hash(output/'predictions.jsonl')
    write(output/'report.json',result,exclusive=True)
    (output/'report.txt').write_text(readable(result,output/'report.json'),encoding='utf-8')
    display(result,output/'report.json')
