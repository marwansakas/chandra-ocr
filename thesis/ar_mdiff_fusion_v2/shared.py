"""Frozen selected AR backbone, adapted MDiff head; AR weights never update."""
from __future__ import annotations

import math
from pathlib import Path
import time

import torch
from torch import nn

from thesis.mdiff_arabic_v1.model import Recognizer, tensor_hash
from thesis.mdiff_arabic_v1.artifacts import batch, digest, file_hash, metrics, read, require_equal, write, write_jsonl
from thesis.mdiff_arabic_v1.experiment import autocast, save_torch, seed_all, stamp, training_key
from .inference import predict_shared
from .logic import RECIPE


class SharedRecognizer(Recognizer):
    def __init__(self,vocab_size,**kwargs):
        super().__init__(vocab_size,arm='mdiff',**kwargs)
        dim=self.embedding.embedding_dim
        self.adapter=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
        for module in (self.encoder,self.visual):
            for p in module.parameters():p.requires_grad_(False)

    def train(self,mode=True):
        super().train(mode)
        self.encoder.eval();self.visual.eval()
        return self

    def adapt_memory(self,memory):return memory+self.adapter(memory)

    def encode(self,pixels,widths):
        with torch.no_grad():memory,mm=super().encode(pixels,widths)
        return self.adapt_memory(memory),mm


def backbone_hash(model):
    return tensor_hash({k:v for k,v in model.state_dict().items() if k.startswith(('encoder.','visual.'))})


def remove_obsolete_checkpoint(folder,relative,current_paths):
    """Only this new run's redundant state/model files, after a new commit."""
    root=folder.resolve();target=(folder/relative).resolve()
    protected={(folder/p).resolve() for p in current_paths}
    if not target.is_relative_to(root) or target.name not in ('state.pt','model.pt'):
        raise ValueError('Refusing checkpoint cleanup outside this shared run')
    if target not in protected and target.exists():target.unlink()


def initialized(ctx):
    seed_all(RECIPE['seed'])
    model=SharedRecognizer(len(ctx['tokenizer']))
    md=torch.load(ctx['md_path'],map_location='cpu',weights_only=True)['model']
    ar=torch.load(ctx['ar_path'],map_location='cpu',weights_only=True)['model']
    state=dict(md)
    state.update({k:v for k,v in ar.items() if k.startswith(('encoder.','visual.'))})
    state.update({k:v for k,v in model.state_dict().items() if k.startswith('adapter.')})
    model.load_state_dict(state,strict=True)
    expected=tensor_hash({k:v for k,v in ar.items() if k.startswith(('encoder.','visual.'))})
    require_equal(backbone_hash(model),expected,'shared AR backbone initialization')
    return model.to(ctx['args'].device),expected


def load_best(ctx):
    folder=ctx['run']/'shared';complete=read(folder/'complete.json');best=complete['best']
    require_equal(complete['binding_sha256'],ctx['binding_sha256'],'shared completion binding')
    for key,hashkey in [('checkpoint','sha256'),('predictions','predictions_sha256')]:
        require_equal(file_hash(folder/best[key]),best[hashkey],'shared best '+key)
    model=SharedRecognizer(len(ctx['tokenizer']))
    saved=torch.load(folder/best['checkpoint'],map_location='cpu',weights_only=True)
    require_equal(saved['binding_sha256'],ctx['binding_sha256'],'shared checkpoint binding')
    model.load_state_dict(saved['model'],strict=True)
    require_equal(backbone_hash(model),complete['backbone_sha256'],'shared frozen backbone')
    ar=torch.load(ctx['ar_path'],map_location='cpu',weights_only=True)['model']
    expected=tensor_hash({k:v for k,v in ar.items() if k.startswith(('encoder.','visual.'))})
    require_equal(complete['backbone_sha256'],expected,'shared backbone equals selected AR')
    model.to(ctx['args'].device).eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,best,complete


def train(ctx):
    folder=ctx['run']/'shared';args=ctx['args'];recipe=ctx['binding']['shared_recipe']
    if (folder/'complete.json').exists():
        _,best,_=load_best(ctx);print('Shared training already complete; selected epoch',best['epoch']);return
    folder.mkdir(parents=True,exist_ok=True)
    header=folder/'run.json'
    if header.exists():
        require_equal(read(header)['binding_sha256'],ctx['binding_sha256'],'shared training source binding')
        if not args.resume:raise ValueError('Shared training exists. Use -Resume to continue from its last completed epoch.')
    else:
        if args.resume:raise ValueError('No shared training run exists to resume')
        write(header,dict(binding_sha256=ctx['binding_sha256'],recipe=recipe,created_utc=stamp()))
    model,expected=initialized(ctx)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    start=1;best=None;best_key=None;stale=0;updates=0
    pointer=folder/'last.json'
    if args.resume and pointer.exists():
        last=read(pointer);require_equal(file_hash(folder/last['path']),last['sha256'],'shared resume state')
        state=torch.load(folder/last['path'],map_location='cpu',weights_only=True)
        require_equal(state['binding_sha256'],ctx['binding_sha256'],'shared resume binding')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        start=state['epoch']+1;best=state['best'];best_key=tuple(state['best_key']);stale=state['stale'];updates=state['updates']
        require_equal(backbone_hash(model),expected,'resumed frozen AR backbone')
    train_rows=ctx['parts']['train'];validation=ctx['parts']['validation']
    print(f'Shared encoder: frozen AR backbone; trainable head/adapter parameters={sum(p.numel() for p in parameters):,}',flush=True)
    for epoch in range(start,recipe['maximum_epochs']+1):
        if stale>=recipe['patience']:break
        # Restarting an interrupted epoch reproduces its ordering and dropout RNG.
        seed_all(RECIPE['seed']+epoch)
        model.train();began=time.perf_counter();total_loss=0.
        if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
        order=torch.randperm(len(train_rows),generator=torch.Generator().manual_seed(RECIPE['seed']+epoch)).tolist()
        warm=min(1.,epoch/max(1,recipe['warmup_epochs']))
        progress=max(0,epoch-recipe['warmup_epochs'])/max(1,recipe['maximum_epochs']-recipe['warmup_epochs'])
        factor=warm*(recipe['minimum_lr_fraction']+(1-recipe['minimum_lr_fraction'])*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:group['lr']=recipe['learning_rate']*factor
        for offset in range(0,len(order),recipe['accumulation']):
            indices=order[offset:offset+recipe['accumulation']];optimizer.zero_grad(set_to_none=True)
            for index in indices:
                px,widths,target=batch([train_rows[index]],args.drive/'ground_truth_lines',ctx['tokenizer'],ctx['image_config'],args.device,epoch)
                with autocast(args.device):loss,_=model.loss(px,widths,target)
                if not torch.isfinite(loss):raise ValueError('Nonfinite shared training loss; previous completed epoch remains resumable')
                (loss/len(indices)).backward();total_loss+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(parameters,recipe['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();updates+=1
            if offset//recipe['accumulation']%25==0:
                print(f'shared epoch={epoch} train={min(offset+len(indices),len(order))}/{len(order)} loss={total_loss/min(offset+len(indices),len(order)):.4f}',flush=True)
        train_seconds=time.perf_counter()-began
        require_equal(backbone_hash(model),expected,'AR backbone unchanged after optimizer updates')
        began=time.perf_counter();predictions=predict_shared(model,validation,ctx);m=metrics(validation,predictions)
        val_seconds=time.perf_counter()-began;key=training_key(m,epoch)
        improved=best_key is None or key<best_key;stale=0 if improved else stale+1
        # Each attempt gets its own directory. A crash preserves partial work;
        # the atomic last.json pointer commits only a fully completed epoch.
        attempt=folder/f'e{epoch:03d}_{stamp()}';attempt.mkdir(exist_ok=False)
        write_jsonl(attempt/'validation.jsonl',predictions.values())
        if improved:
            save_torch(attempt/'model.pt',dict(model=model.state_dict(),epoch=epoch,binding_sha256=ctx['binding_sha256']))
            best_key=key;best=dict(epoch=epoch,checkpoint=str((attempt/'model.pt').relative_to(folder)),
                sha256=file_hash(attempt/'model.pt'),predictions=str((attempt/'validation.jsonl').relative_to(folder)),
                predictions_sha256=file_hash(attempt/'validation.jsonl'),validation_metrics=m)
        write(attempt/'history.json',dict(epoch=epoch,train_loss=total_loss/len(train_rows),validation=m,
            improved=improved,stale=stale,updates=updates,train_seconds=train_seconds,validation_seconds=val_seconds,
            backbone_sha256=expected,peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None),exclusive=True)
        previous=read(pointer) if pointer.exists() else None
        previous_best=previous.get('best_checkpoint') if previous else None
        save_torch(attempt/'state.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
            binding_sha256=ctx['binding_sha256'],best=best,best_key=best_key,stale=stale,updates=updates))
        write(pointer,dict(path=str((attempt/'state.pt').relative_to(folder)),sha256=file_hash(attempt/'state.pt'),
            epoch=epoch,best_checkpoint=best['checkpoint']))
        # The pointer is now committed. Keep latest resume state + selected best;
        # per-epoch predictions/history remain, avoiding dozens of optimizer copies.
        protected=[str((attempt/'state.pt').relative_to(folder)),best['checkpoint']]
        if previous:remove_obsolete_checkpoint(folder,previous['path'],protected)
        if previous_best and previous_best!=best['checkpoint']:
            remove_obsolete_checkpoint(folder,previous_best,protected)
        print(f'shared epoch={epoch} CER={100*m["cer"]:.3f}% WER={100*m["wer"]:.3f}% exact={m["exact_lines"]}/150 best={improved} wait={stale}/{recipe["patience"]}',flush=True)
    if best is None:raise ValueError('No completed shared epoch/checkpoint')
    write(folder/'complete.json',dict(binding_sha256=ctx['binding_sha256'],best=best,backbone_sha256=expected,
        last_sha256=file_hash(pointer),test_used=False,recipe=recipe))
    print('Shared training complete. Selected epoch:',best['epoch'],'| Test not run.')
