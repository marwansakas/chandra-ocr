"""Image-conditioned scores; active tokens are hidden before MDiff scoring."""
from __future__ import annotations

import math
import time

from PIL import Image
import torch

from thesis.diffusion.data import resolve_relative
from thesis.diffusion.images import prepare_images
from thesis.mdiff_arabic_v1.model import Recognizer, PAD, MASK, EOS
from thesis.mdiff_arabic_v1.experiment import autocast, synchronize
from .logic import RECIPE, candidates, prediction_only


def image_batch(row,dataset,config,device):
    # Deliberately independent of batch(), which also encodes reference targets.
    with Image.open(resolve_relative(dataset,row['image_path'])) as image:
        px,widths=prepare_images([image.convert('RGB')],config)
    return px.to(device),widths.to(device)


def load_primary(path,tokenizer,arm,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=Recognizer(len(tokenizer),arm=arm)
    model.load_state_dict(saved['model'],strict=True)
    model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def active_ids(tokenizer,text,capacity):
    padded=tokenizer.encode(text,capacity,eos_fill=False)
    n=len(text)+1
    if padded[n-1]!=EOS or any(v!=PAD for v in padded[n:]):raise ValueError('Invalid candidate boundary')
    return padded,n


@torch.no_grad()
def ar_score(model,memory,mm,tokenizer,text):
    padded,n=active_ids(tokenizer,text,model.capacity)
    target=torch.tensor(padded,device=memory.device,dtype=torch.long)[None,:]
    shifted=torch.cat((torch.full_like(target[:,:1],MASK),target[:,:-1]),1)
    # Future padding cannot affect any active position under the causal mask.
    logits=model.decode_logits(shifted,memory,mm,memory.new_zeros(1),causal=True)
    logp=logits[0,:n].float().log_softmax(-1).gather(1,target[0,:n,None]).squeeze(1)
    if not torch.isfinite(logp).all():raise ValueError('Nonfinite AR score')
    return dict(ar_sum_logp=float(logp.sum()),ar_mean_logp=float(logp.mean()),active_tokens=n)


@torch.no_grad()
def md_score(model,memory,mm,tokenizer,text,batch_size=8):
    padded,n=active_ids(tokenizer,text,model.capacity)
    target=torch.tensor(padded,device=memory.device,dtype=torch.long)
    visible=target.masked_fill(target.eq(PAD),MASK)
    values=[]
    for start in range(0,n,batch_size):
        positions=torch.arange(start,min(start+batch_size,n),device=memory.device)
        tokens=visible[None,:].expand(len(positions),-1).clone()
        rows=torch.arange(len(positions),device=memory.device)
        tokens[rows,positions]=MASK  # Includes EOS; never score a visible target.
        t=memory.new_full((len(positions),),1./n)
        logits=model.decode_logits(tokens,memory.expand(len(positions),-1,-1),
            mm.expand(len(positions),-1),t,causal=False)
        logp=logits[rows,positions].float().log_softmax(-1).gather(1,target[positions,None]).squeeze(1)
        values.append(logp)
    values=torch.cat(values)
    if not torch.isfinite(values).all():raise ValueError('Nonfinite MDiff pseudo-score')
    return dict(md_sum_logp=float(values.sum()),md_mean_logp=float(values.mean()))


@torch.no_grad()
def generate_memory(model,memory,mm,steps=3):
    """The original masked/correction sampler with an already encoded image."""
    state=torch.full((len(memory),model.capacity),MASK,device=memory.device,dtype=torch.long)
    for k in range(steps):
        logits=model.decode_logits(state,memory,mm,memory.new_full((len(state),),(steps-k)/steps))
        state=torch.where(state.eq(MASK),logits.argmax(-1),state)
        logits=model.decode_logits(model.hide_tail(state),memory,mm,memory.new_zeros(len(state)))
        state=logits.argmax(-1)
        if k+1<steps:
            probabilities=logits.float().softmax(-1).gather(-1,state[...,None]).squeeze(-1)
            state=model.hide_tail(state)
            for row in range(len(state)):
                active=state[row].ne(MASK).nonzero().flatten()
                count=math.ceil(len(active)*(steps-k-1)/steps)
                order=probabilities[row,active].argsort(stable=True)
                state[row,active[order[:count]]]=MASK
    return state


@torch.no_grad()
def record(row,ar,md,models,tokenizer,dataset,image_config,device,*,shared=False):
    """Scoring/proposals have no access to a reference text or label path."""
    allowed=('sample_id','image_path','image_sha256','pixels_sha256','width','height')
    row={k:row[k] for k in allowed if k in row}
    ar=prediction_only(ar);md=prediction_only(md) if md is not None else None
    started=time.perf_counter();synchronize(device)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    a,d=models
    before=(a.decoder_passes,d.decoder_passes)
    with autocast(device):
        px,widths=image_batch(row,dataset,image_config,device)
        am,amm=a.encode(px,widths)
        if shared:
            # A single AR image encoding serves both heads in this experiment.
            dm,dmm=d.adapt_memory(am),amm
            ids=generate_memory(d,dm,dmm)
            text,eos=tokenizer.decode(ids[0].tolist())
            md=dict(sample_id=row['sample_id'],text=text,has_eos=eos,
                status='completed' if eos else 'failed',length_failure=not eos)
        else:
            dm,dmm=d.encode(px,widths)
        pool=candidates(ar,md)
        if shared:pool=[c for c in pool if any(o in ('ar','mdiff') for o in c['origins'])]
        for candidate in pool:
            candidate.update(ar_score(a,am,amm,tokenizer,candidate['text']))
            candidate.update(md_score(d,dm,dmm,tokenizer,candidate['text'],RECIPE['pll_batch_size']))
    synchronize(device)
    return dict(sample_id=row['sample_id'],ar=ar,mdiff=md,candidates=pool,
        seconds=time.perf_counter()-started,encoder_calls=1 if shared else 2,
        ar_decoder_calls=a.decoder_passes-before[0],md_decoder_calls=d.decoder_passes-before[1],
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)


@torch.no_grad()
def predict_shared(model,rows,ctx):
    model.eval();device=ctx['args'].device;out={}
    for row in rows:
        started=time.perf_counter()
        px,widths=image_batch(row,ctx['args'].drive/'ground_truth_lines',ctx['image_config'],device)
        with autocast(device):
            memory,mm=model.encode(px,widths)
            ids=generate_memory(model,memory,mm)
        text,eos=ctx['tokenizer'].decode(ids[0].tolist());synchronize(device)
        out[row['sample_id']]=dict(sample_id=row['sample_id'],text=text,has_eos=eos,
            status='completed' if eos else 'failed',length_failure=not eos,seconds=time.perf_counter()-started)
    return out
