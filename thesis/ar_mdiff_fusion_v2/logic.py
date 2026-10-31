"""Bounded proposals and selection; inference never consumes reference labels."""
from __future__ import annotations

import itertools
import math

from thesis.diffusion.metrics import align
from thesis.mdiff_arabic_v1.artifacts import damage, metrics

VERSION = 'ar_mdiff_fusion_suite_v2'
RECIPE = dict(version=VERSION, seed=54101, capacity=144,
    methods=['dual_rescore', 'span_merge', 'shared_encoder'],
    candidate_limit=10, maximum_spans=6, maximum_span_length=8,
    score='mean log probability over characters and first EOS; no PAD or repeated EOS',
    md_score='one masked position per conditional, including EOS; t=1/active_length; MASK tail',
    pll_batch_size=8, alpha_grid=[.25,.5,.75], margin_grid=[0.,.05,.15],
    selection='minimum calibration character edits, word edits, negative exact count, switches; then larger margin and lower alpha',
    calibration='120 development-used validation lines; grouped 30-line audit excluded from tuning',
    test_lines=752, test_policy='historical development-used exploratory comparison; no promotion; no gate bypass mutation',
    shared=dict(maximum_epochs=60,patience=10,learning_rate=.0001,minimum_lr_fraction=.1,
        warmup_epochs=2,accumulation=16,weight_decay=.01,gradient_clip=1.,
        encoder='selected AR encoder and visual projection frozen in eval mode',
        decoder='selected MDiff decoder initialization plus zero-initialized residual adapter',
        objective='unchanged MDiff masked/correction/tail loss; canonical TRAIN only',
        selection='minimum raw validation MDiff character edits, word edits, negative exact count, epoch'))


def valid(p):
    return (p.get('status') in ('completed','ok') and p.get('has_eos') is True
            and not p.get('length_failure',False) and isinstance(p.get('text'),str))


def prediction_only(p):
    return {k:p[k] for k in ('sample_id','text','status','has_eos','length_failure') if k in p}


def disagreement_spans(a,b):
    """Levenshtein spans, including zero-width insertions; logical Unicode order."""
    ai=bi=0;start=None;out=[]
    for op,_,x,y in align(a,b):
        if op=='equal':
            if start is not None:out.append((*start,ai,bi));start=None
        elif start is None:start=(ai,bi)
        ai+=x is not None;bi+=y is not None
    if start is not None:out.append((*start,ai,bi))
    if ai!=len(a) or bi!=len(b):raise ValueError('Invalid alignment extent')
    return out  # (AR start, MD start, AR end, MD end)


def splice(a,b,spans):
    out=[];cursor=0
    for a0,b0,a1,b1 in sorted(spans):
        if a0<cursor:raise ValueError('Overlapping span proposals')
        out.extend([a[cursor:a0],b[b0:b1]]);cursor=a1
    return ''.join([*out,a[cursor:]])


def candidates(ar,mdiff,capacity=144):
    """Only prediction text enters this function; no references or dictionaries."""
    out=[];seen={}
    def add(text,origin,spans=()):
        if len(text)+1>capacity:return
        if text in seen:
            if origin not in seen[text]['origins']:seen[text]['origins'].append(origin)
            return
        item=dict(text=text,origins=[origin],spans=[list(s) for s in spans])
        out.append(item);seen[text]=item
    if valid(ar):add(ar['text'],'ar')
    if valid(mdiff):add(mdiff['text'],'mdiff')
    if valid(ar) and valid(mdiff):
        spans=disagreement_spans(ar['text'],mdiff['text'])
        spans=[s for s in spans if max(s[2]-s[0],s[3]-s[1])<=RECIPE['maximum_span_length']]
        spans=sorted(spans,key=lambda s:(max(s[2]-s[0],s[3]-s[1]),s[0],s[1]))[:RECIPE['maximum_spans']]
        for s in spans:add(splice(ar['text'],mdiff['text'],[s]),'span',[s])
        # One bounded combination of the two shortest eligible disagreements.
        if len(spans)>1:add(splice(ar['text'],mdiff['text'],spans[:2]),'span',spans[:2])
    return out[:RECIPE['candidate_limit']]


def choose(record,method,config):
    alpha,margin=config['alpha'],config['margin']
    if not 0<=alpha<=1 or not math.isfinite(margin) or margin<0:raise ValueError('Invalid fusion configuration')
    pool=[c for c in record['candidates'] if method=='span_merge' or any(o in ('ar','mdiff') for o in c['origins'])]
    for c in pool:
        if not all(math.isfinite(c[k]) for k in ('ar_mean_logp','md_mean_logp')):raise ValueError('Nonfinite candidate score')
    baseline=next((c for c in pool if 'ar' in c['origins']),None)
    result=prediction_only(record['ar']);result.update(seconds=0.,alpha=alpha,margin=margin,
        selected_origin='ar',score_gain=0.,method=method)
    if not pool:return result
    def score(c):return (1-alpha)*c['ar_mean_logp']+alpha*c['md_mean_logp']
    best=max(pool,key=score)  # deterministic ties retain earlier AR candidate
    gain=score(best)-score(baseline) if baseline is not None else None
    if baseline is None or gain>margin:
        result.update(text=best['text'],status='completed',has_eos=True,length_failure=False,
            selected_origin='+'.join(best['origins']),score_gain=gain)
    return result


def evaluate(rows,ar,predictions):
    chosen={r['sample_id']:predictions[r['sample_id']] for r in rows}
    baseline={r['sample_id']:ar[r['sample_id']] for r in rows}
    m=metrics(rows,chosen);base=metrics(rows,baseline);changes=damage(rows,baseline,chosen)
    conditions=dict(lower_cer=m['char_distance']<base['char_distance'],
        no_worse_wer=m['word_distance']<=base['word_distance'],
        no_ar_exact_lines_lost=changes['exact_lines_lost']==0,
        complete_eos=m['failed_lines']==m['eos_failures']==m['length_failures']==0)
    return dict(metrics=m,ar_metrics=base,changes_vs_ar=changes,conditions=conditions,
        passes_preservation_check=all(conditions.values()),
        switched_lines=sum(predictions[r['sample_id']]['text']!=ar[r['sample_id']]['text'] for r in rows))


def select(records,method,config):
    return {r['sample_id']:choose(r,method,config) for r in records}


def calibrate(rows,records,method):
    """Only the caller-supplied calibration subset participates in the search."""
    ids={r['sample_id'] for r in rows};subset=[r for r in records if r['sample_id'] in ids]
    if len(subset)!=len(rows) or len({r['sample_id'] for r in subset})!=len(rows):raise ValueError('Calibration membership mismatch')
    ar={r['sample_id']:r['ar'] for r in subset};trials=[]
    for alpha,margin in itertools.product(RECIPE['alpha_grid'],RECIPE['margin_grid']):
        config=dict(alpha=alpha,margin=margin)
        trials.append(dict(config=config,**evaluate(rows,ar,select(subset,method,config))))
    def key(t):
        m=t['metrics'];c=t['config']
        return (m['char_distance'],m['word_distance'],-m['exact_lines'],t['switched_lines'],-c['margin'],c['alpha'])
    chosen=min(trials,key=key)
    return dict(method=method,config=chosen['config'],chosen=chosen,trials=trials,
        sample_ids=[r['sample_id'] for r in rows],test_used=False,
        policy='Report selected raw result even if preservation checks fail; no promotion or hidden fallback')
