"""Independent AR runs, epoch-transaction resume, and explicit historical Test."""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import shutil
import time

from PIL import Image
import torch

from thesis.diffusion.images import (ImageConfig, prepare_images, augment_image,
    AugmentationConfig, augmentation_seed)
from thesis.diffusion.tokenizer import CharacterTokenizer
from thesis.mdiff_arabic_v1.artifacts import (ASSETS, canonical, damage, digest, environment,
    file_hash, indexed, jsonl, metrics, read, write)
from thesis.mdiff_arabic_v1.experiment import autocast, save_torch, seed_all, stamp, synchronize, training_key
from thesis.mdiff_arabic_v1.model import Recognizer, load_encoder, tensor_hash
from .data import SEED, TRIALS, verify

RECIPE=dict(arm='ar',maximum_epochs=100,patience=10,accumulation=16,batch_size=1,
    warmup_epochs=5,encoder_frozen_epochs=5,decoder_lr=3e-4,encoder_lr=1e-5,
    minimum_lr_fraction=.1,weight_decay=.01,gradient_clip=1.,augmentation='trocr_views_v1',
    precision='bf16_on_cuda_fp32_on_cpu',seed=SEED,capacity=144,
    selection='minimum validation character edits, then word edits, then most exact lines, then earliest epoch')


@contextmanager
def run_lock(folder):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.active.lock').open('a+b') as handle:
        if handle.seek(0,2)==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:raise RuntimeError('This KHATT suite is already in use by another process') from e
        try:yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def sources(args):
    paths=list((args.repo/'thesis/khatt_ablation_v1').glob('*.py'))
    paths+=list((args.repo/'thesis/mdiff_arabic_v1').rglob('*.py'))
    paths += [args.repo/'thesis/diffusion'/f for f in ('data.py','images.py','tokenizer.py','metrics.py','paddle_conditioning.py')]
    paths += [args.repo/'thesis/scripts/common.py',args.repo/'thesis/scripts/run_khatt_ablation_v1.py',
              args.repo/'tests/test_khatt_ablation_v1.py',args.runner]
    return {str(p.resolve()):file_hash(p) for p in sorted(set(paths))}


def context(args):
    seed_all(SEED)
    prepared=verify(args)
    if args.epochs!=100 or args.patience!=10:
        raise ValueError('This suite requires -Epochs 100 -Patience 10')
    assets=args.drive/'pretrained/mdiff_s_v1'
    for name,sha in ASSETS.items():
        if file_hash(assets/name)!=sha:raise ValueError('Official encoder asset changed: '+name)
    env=environment(args.device)
    # Bind cached comparators without consulting their test scores for selection.
    baseline_paths=[args.drive/'hybrids/ar_md_fusion2_s54101/historical_752'/name
                    for name in ('report.json','predictions.jsonl','seal.json')]
    baseline_paths += [args.drive/'exploratory/mdar_20261031_x01'/name
                       for name in ('report.json','predictions.jsonl')]
    baseline_hashes={str(p):file_hash(p) for p in baseline_paths}
    binding=dict(version='khatt_ablation_v1',suite=args.suite_name,recipe=RECIPE,
        prepared_sha256=file_hash(args.suite/'prepared.json'),sources=sources(args),environment=env,
        pretrained_sha256=ASSETS,baselines=baseline_hashes,
        test_policy='User-invoked historical development-used comparison on fixed 752 children lines; no automatic promotion')
    return dict(args=args,prepared=prepared,binding=binding,binding_sha256=digest(binding),
        tokenizer=CharacterTokenizer.load(args.suite/'tokenizer.json'),
        config=ImageConfig(**prepared['image_config']),assets=assets)


def check(ctx):
    p=ctx['args'].suite/'check.json'
    value=dict(binding=ctx['binding'],binding_sha256=ctx['binding_sha256'])
    if p.exists():
        if read(p)!=value:raise ValueError('Suite source/environment binding changed; use a new SuiteName')
    else:
        # Reproducible code snapshot, including the active PowerShell launcher.
        snap=ctx['args'].suite/'source';snap.mkdir(exist_ok=True)
        for i,(name,sha) in enumerate(ctx['binding']['sources'].items()):
            target=snap/f'{i:03d}_{Path(name).name}'
            shutil.copyfile(name,target)
            if file_hash(target)!=sha:raise ValueError('Source snapshot copy mismatch')
        write(p,value)
    print('Verified CUDA/recipe/source/splits/pretrained assets. Validation: 150 children only.',flush=True)


def verify_images(rows):
    for r in rows:
        if file_hash(Path(r['image_path']))!=r['image_sha256']:
            raise ValueError('Image changed: '+r['sample_id'])
        if 'label_path' in r and file_hash(Path(r['label_path']))!=r['label_sha256']:
            raise ValueError('Child label file changed: '+r['sample_id'])


def image_batch(row,ctx,epoch=0):
    with Image.open(row['image_path']) as source:image=source.convert('RGB')
    if epoch:
        image,_=augment_image(image,AugmentationConfig(RECIPE['augmentation']),
            augmentation_seed(row['sample_id'],epoch,SEED),view_index=(epoch-1)%3)
    px,widths=prepare_images([image],ctx['config'])
    return px.to(ctx['args'].device),widths.to(ctx['args'].device)


@torch.no_grad()
def predict_one(model,row,ctx):
    # Reference-free inference: only sample identity and the image enter here.
    row={k:row[k] for k in ('sample_id','image_path','image_sha256')}
    synchronize(ctx['args'].device);started=time.perf_counter()
    px,widths=image_batch(row,ctx)
    with autocast(ctx['args'].device):ids,_=model.generate(px,widths)
    if ids.ndim!=2 or ids.shape[0]!=1:
        raise ValueError('Recognizer.generate must return one batch of token IDs plus metadata')
    text,has_eos=ctx['tokenizer'].decode(ids[0].cpu().tolist())
    synchronize(ctx['args'].device)
    return dict(sample_id=row['sample_id'],image_sha256=row['image_sha256'],text=text,
        status='completed',has_eos=has_eos,length_failure=not has_eos,seconds=time.perf_counter()-started)


def optimizer_for(model):
    return torch.optim.AdamW([
        dict(params=model.encoder.parameters(),lr=RECIPE['encoder_lr'],base_lr=RECIPE['encoder_lr']),
        dict(params=[p for n,p in model.named_parameters() if not n.startswith('encoder.')],
             lr=RECIPE['decoder_lr'],base_lr=RECIPE['decoder_lr'])],weight_decay=RECIPE['weight_decay'])


def remaining_epochs(epoch,stale):
    return range(epoch+1,101) if stale<10 else range(0)


def lr_factor(updates,per_epoch):
    warmup=5*per_epoch;total=100*per_epoch
    if updates<warmup:return min(1.,(updates+1)/warmup)
    return .1+.9*.5*(1+math.cos(math.pi*min(updates-warmup,total-warmup)/(total-warmup)))


def prune_states(folder,protected):
    # Only this new run's own transaction model/optimizer files are eligible.
    root=folder.resolve();protected={str((folder/p).resolve()) for p in protected}
    for attempt in folder.glob('e[0-9][0-9][0-9]_*'):
        if not attempt.is_dir():continue
        for name in ('model.pt','state.pt'):
            p=(attempt/name).resolve()
            if p.is_relative_to(root) and p.exists() and str(p) not in protected:p.unlink()


def selected(ctx,name):
    folder=ctx['args'].suite/'runs'/name
    complete=read(folder/'complete.json')
    if complete['binding_sha256']!=ctx['binding_sha256']:raise ValueError('Completed run binding mismatch: '+name)
    best=complete['best']
    for key,sha in (('checkpoint','sha256'),('predictions','predictions_sha256')):
        if file_hash(folder/best[key])!=best[sha]:raise ValueError('Selected artifact changed: '+name)
    return folder,best


def train_one(ctx,name):
    args=ctx['args'];folder=args.suite/'runs'/name
    if (folder/'complete.json').exists():
        _,best=selected(ctx,name)
        print(f'{name}: already complete; selected epoch {best["epoch"]}. Skipping.',flush=True)
        return
    if (folder/'run.json').exists() and not args.resume:
        raise ValueError('Interrupted run exists. Use -Resume to continue: '+name)
    rows=jsonl(args.suite/(name+'.jsonl'));val=jsonl(args.suite/'validation.jsonl')
    verify_images(rows+val)
    folder.mkdir(parents=True,exist_ok=True)
    seed_all(SEED)
    model=Recognizer(len(ctx['tokenizer']),capacity=144,arm='ar')
    transfer=load_encoder(model.encoder,ctx['assets']/'best.pth')
    initialization=tensor_hash(model.state_dict())
    record=dict(binding_sha256=ctx['binding_sha256'],trial=name,initialization_sha256=initialization,
        encoder_initialization_sha256=transfer['encoder_initialization_sha256'],recipe=RECIPE,
        counts=ctx['prepared']['counts'][name],initialization='official pretrained encoder; fresh Arabic head; no child-trained checkpoint')
    if (folder/'run.json').exists():
        if read(folder/'run.json')!=record:raise ValueError('Resume run identity mismatch')
    else:write(folder/'run.json',record)
    model.to(args.device);optimizer=optimizer_for(model)
    epoch=0;stale=0;updates=0;best=None;best_key=None
    pointer=folder/'last.json'
    if pointer.exists():
        last=read(pointer);p=folder/last['path']
        if file_hash(p)!=last['sha256']:raise ValueError('Resume checkpoint hash mismatch')
        state=torch.load(p,map_location='cpu',weights_only=True)
        if state['binding_sha256']!=ctx['binding_sha256'] or state['trial']!=name:
            raise ValueError('Resume checkpoint identity mismatch')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        epoch,stale,updates=state['epoch'],state['stale'],state['updates']
        best,best_key=state['best'],tuple(state['best_key'])
        torch.set_rng_state(state['torch_rng'])
        if args.device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'{name}: resuming after committed epoch {epoch}; interrupted epoch will replay.',flush=True)
    print(f'{name}: {len(rows)} TRAIN / {len(val)} children validation; CUDA={args.device=="cuda"}',flush=True)
    for epoch in remaining_epochs(epoch,stale):
        for p in model.encoder.parameters():p.requires_grad_(epoch>5)
        model.train();loss_sum=0.;started=time.perf_counter();last_print=started
        order=torch.randperm(len(rows),generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
        per_epoch=math.ceil(len(rows)/16)
        for offset in range(0,len(order),16):
            indices=order[offset:offset+16];optimizer.zero_grad(set_to_none=True)
            for g in optimizer.param_groups:g['lr']=g['base_lr']*lr_factor(updates,per_epoch)
            for index in indices:
                row=rows[index];px,widths=image_batch(row,ctx,epoch)
                target=torch.tensor([ctx['tokenizer'].encode(row['text'],144,eos_fill=False)],device=args.device)
                with autocast(args.device):loss,_=model.loss(px,widths,target)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss; saved epoch remains resumable')
                (loss/len(indices)).backward();loss_sum+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();updates+=1
            if time.perf_counter()-last_print>30:
                print(f'{name} epoch={epoch} TRAIN={min(offset+16,len(rows))}/{len(rows)} loss={loss_sum/min(offset+16,len(rows)):.4f}',flush=True)
                last_print=time.perf_counter()
        train_seconds=time.perf_counter()-started
        model.eval();predictions={}
        for r in val:predictions[r['sample_id']]=predict_one(model,r,ctx)
        m=metrics(val,predictions);key=training_key(m,epoch)
        improved=best_key is None or key<best_key;stale=0 if improved else stale+1
        attempt=folder/f'e{epoch:03d}_{stamp()}';attempt.mkdir()
        predpath=attempt/'validation.jsonl';predpath.write_bytes(b''.join(canonical(p) for p in predictions.values()))
        if improved:
            checkpoint=attempt/'model.pt'
            save_torch(checkpoint,dict(model=model.state_dict(),epoch=epoch,trial=name,binding_sha256=ctx['binding_sha256']))
            best_key=key
            best=dict(epoch=epoch,checkpoint=str(checkpoint.relative_to(folder)),sha256=file_hash(checkpoint),
                predictions=str(predpath.relative_to(folder)),predictions_sha256=file_hash(predpath),validation_metrics=m)
        write(attempt/'history.json',dict(epoch=epoch,train_loss=loss_sum/len(rows),validation=m,
            improved=improved,stale=stale,updates=updates,train_seconds=train_seconds,test_used=False))
        statepath=attempt/'state.pt'
        save_torch(statepath,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
            trial=name,binding_sha256=ctx['binding_sha256'],best=best,best_key=best_key,stale=stale,
            updates=updates,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else []))
        write(pointer,dict(path=str(statepath.relative_to(folder)),sha256=file_hash(statepath),epoch=epoch))
        write(folder/'best.json',best)
        prune_states(folder,[str(statepath.relative_to(folder)),best['checkpoint']])
        print(f'{name} epoch={epoch} CER={100*m["cer"]:.3f}% WER={100*m["wer"]:.3f}% exact={m["exact_lines"]}/150 best={improved} patience={stale}/10',flush=True)
        if stale>=10:break
    if best is None:raise ValueError('No completed epoch')
    write(folder/'complete.json',dict(binding_sha256=ctx['binding_sha256'],best=best,epoch=epoch,
        last_sha256=file_hash(pointer),test_used=False,stop_reason='patience' if stale>=10 else 'maximum_epochs'))
    print(name,'complete. Best epoch:',best['epoch'],'Test has not run.',flush=True)


def comparison(rows,by_model):
    span=indexed(by_model['span_merge'],rows)
    results={}
    for name,values in by_model.items():
        predictions=indexed(values,rows)
        results[name]=dict(metrics=metrics(rows,predictions),vs_span_merge=damage(rows,span,predictions))
    return results


def test_all(ctx):
    args=ctx['args']
    # Require all five finalized validation-selected checkpoints before accessing
    # historical references. Test scores never alter selection or training.
    selections={name:selected(ctx,name) for name in TRIALS}
    rows=jsonl(args.drive/'splits_70_05_25_v3/test.jsonl')
    identities=jsonl(args.suite/'test_identity.jsonl')
    if len(rows)!=752 or [r['sample_id'] for r in rows]!=[r['sample_id'] for r in identities]:
        raise ValueError('Historical test must be the exact fixed 752 lines')
    child=args.drive/'ground_truth_lines'
    from .data import child_row
    rows=[child_row(r,child) for r in rows];verify_images(rows)
    by_model={}
    original=jsonl(args.drive/'exploratory/mdar_20261031_x01/predictions.jsonl')
    for name in ('paddle','frozen_v2_2','span_hybrid_verifier_v4','transformer_arabic_control','mdiff_arabic_raw'):
        by_model[name]=[r for r in original if r['model']==name]
    fused=jsonl(args.drive/'hybrids/ar_md_fusion2_s54101/historical_752/predictions.jsonl')
    by_model['span_merge']=[r for r in fused if r['model']=='span_merge']
    output=args.suite/'historical_752';output.mkdir(exist_ok=True)
    for name,(folder,best) in selections.items():
        cache=output/name;cache.mkdir(exist_ok=True)
        expected=dict(binding_sha256=ctx['binding_sha256'],checkpoint_sha256=best['sha256'],lines=752)
        meta=cache/'identity.json'
        if meta.exists() and read(meta)!=expected:raise ValueError('Historical cache identity mismatch')
        write(meta,expected)
        seal=cache/'seal.json'
        if seal.exists():
            for filename,sha in read(seal)['files'].items():
                if file_hash(cache/filename)!=sha:raise ValueError('Completed historical cache changed: '+filename)
        predictions={};model=None
        cached_files={}
        for i,row in enumerate(rows,1):
            # Existing children IDs can be long: use short stable cache names.
            p=cache/(digest(row['sample_id'])+'.json')
            if p.exists():
                prediction=read(p)
                if prediction['sample_id']!=row['sample_id'] or prediction['image_sha256']!=row['image_sha256']:
                    raise ValueError('Historical line cache identity mismatch')
            else:
                if model is None:
                    saved=torch.load(folder/best['checkpoint'],map_location='cpu',weights_only=True)
                    if saved['binding_sha256']!=ctx['binding_sha256'] or saved['trial']!=name:
                        raise ValueError('Selected checkpoint payload mismatch')
                    model=Recognizer(len(ctx['tokenizer']),capacity=144,arm='ar')
                    model.load_state_dict(saved['model'],strict=True);model.to(args.device).eval()
                prediction=predict_one(model,row,ctx);write(p,prediction)
            predictions[row['sample_id']]=dict(prediction,model=name)
            cached_files[p.name]=file_hash(p)
            if i%50==0:print(f'{name} historical={i}/752',flush=True)
        by_model[name]=list(predictions.values())
        write(seal,dict(identity=expected,files=cached_files))
        del model
        if args.device=='cuda':torch.cuda.empty_cache()
    results=comparison(rows,by_model)
    report=dict(binding_sha256=ctx['binding_sha256'],scope='HISTORICAL DEVELOPMENT-USED; not a fresh held-out test',
        lines=752,results=results,selected=selections and {name:b for name,(_,b) in selections.items()},
        training_note='Mixed runs include 704 new child TRAIN lines; old AR/span do not. Different TRAIN sizes also imply different optimizer-update budgets.',
        automatic_promotion=False)
    write(output/'report.json',report)
    (output/'predictions.jsonl').write_bytes(b''.join(canonical(p) for values in by_model.values() for p in values))
    write(output/'seal.json',dict(binding_sha256=ctx['binding_sha256'],
        report_sha256=file_hash(output/'report.json'),predictions_sha256=file_hash(output/'predictions.jsonl')))
    stats(args)


def stats(args):
    output=args.suite/'historical_752';seal=read(output/'seal.json')
    if file_hash(output/'report.json')!=seal['report_sha256'] or file_hash(output/'predictions.jsonl')!=seal['predictions_sha256']:
        raise ValueError('Historical report/predictions changed after completion')
    report=read(output/'report.json')
    if report['binding_sha256']!=seal['binding_sha256'] or report['lines']!=752:
        raise ValueError('Historical report binding/line count mismatch')
    print('\nHISTORICAL DEVELOPMENT-USED COMPARISON — 752 children lines; frozen span baseline')
    print(f'{"Model":31} {"Lines":>5} {"CER %":>7} {"WER %":>7} {"Exact %":>8} {"Sub":>5} {"Ins":>5} {"Del":>5}')
    csvrows=['Model,Lines,CER_percent,WER_percent,Exact_percent,Sub,Ins,Del,Exact_lost_vs_span,Exact_gained_vs_span']
    order=('paddle','frozen_v2_2','span_hybrid_verifier_v4','mdiff_arabic_raw',
           'transformer_arabic_control','span_merge',*TRIALS)
    for name in order:
        result=report['results'][name]
        m=result['metrics'];d=result['vs_span_merge']
        print(f'{name:31} {m["lines"]:5d} {100*m["cer"]:7.2f} {100*m["wer"]:7.2f} {100*m["exact_line_accuracy"]:8.2f} {m["substitutions"]:5d} {m["insertions"]:5d} {m["deletions"]:5d}')
        csvrows.append(f'{name},{m["lines"]},{100*m["cer"]:.6f},{100*m["wer"]:.6f},{100*m["exact_line_accuracy"]:.6f},{m["substitutions"]},{m["insertions"]},{m["deletions"]},{d["exact_lines_lost"]},{d["exact_lines_gained"]}')
    path=args.suite/'historical_752/comparison.csv';path.write_text('\n'.join(csvrows)+'\n',encoding='utf-8-sig')
    print('Results:',path)
