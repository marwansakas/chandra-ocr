"""Label-free features and explicit development-only calibration/audit roles."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import math
import unicodedata

import numpy as np

from thesis.diffusion.metrics import align, score_pair
from thesis.mdiff_arabic_v1.artifacts import damage, group, metrics

VERSION = 'ar_mdiff_line_selector_v1'
FEATURE_NAMES = (
    'ar_length', 'mdiff_length', 'relative_length_difference', 'output_edit_ratio',
    'common_prefix_fraction', 'common_suffix_fraction', 'space_fraction_difference',
    'digit_fraction_difference', 'punctuation_fraction_difference',
    'unique_fraction_difference', 'max_run_fraction_difference', 'ar_has_eos',
    'mdiff_has_eos', 'image_log_aspect_ratio',
)
RECIPE = dict(version=VERSION, seed=54101, regularization=10.,
    fit_fraction=.6, calibration_fraction=.2, audit_fraction=.2,
    threshold_grid=[0., .025, .05, .1, .2, .4, 1.], raw_default_threshold=0.,
    target='(AR character edits - MDiff character edits) / max(1, predicted lengths)',
    selector='whole-line AR default; use valid MDiff only when fresh ridge score > threshold',
    eligibility='strictly fewer character edits, no more word edits, zero AR-exact lines lost, complete coverage/EOS',
    limitations='development-used validation; base checkpoints selected on these labels; not fresh generalization or fully unseen OOF evidence')


def valid(row):
    return (row.get('status') in ('ok','completed') and row.get('has_eos') is True
            and not row.get('length_failure',False) and isinstance(row.get('text'),str))


def features(record, ar, mdiff):
    """No reference, source-model accuracy or missing confidence substitutes."""
    a,d=ar['text'],mdiff['text'];n=max(1,len(a),len(d))
    common_prefix=0
    for x,y in zip(a,d):
        if x!=y: break
        common_prefix+=1
    common_suffix=0
    for x,y in zip(reversed(a),reversed(d)):
        if x!=y: break
        common_suffix+=1
    def fraction(text,predicate): return sum(predicate(c) for c in text)/max(1,len(text))
    def run(text):
        maximum=current=0;last=None
        for c in text:
            current=current+1 if c==last else 1;last=c;maximum=max(maximum,current)
        return maximum/max(1,len(text))
    values=[len(a)/144,len(d)/144,(len(d)-len(a))/n,
        sum(op!='equal' for op,*_ in align(a,d))/n,common_prefix/n,common_suffix/n,
        fraction(d,str.isspace)-fraction(a,str.isspace),
        fraction(d,str.isdigit)-fraction(a,str.isdigit),
        fraction(d,lambda c:unicodedata.category(c).startswith('P'))-fraction(a,lambda c:unicodedata.category(c).startswith('P')),
        len(set(d))/max(1,len(d))-len(set(a))/max(1,len(a)),run(d)-run(a),
        float(ar.get('has_eos') is True),float(mdiff.get('has_eos') is True),
        math.log(max(1,record['width'])/max(1,record['height']))]
    if len(values)!=len(FEATURE_NAMES) or not all(math.isfinite(x) for x in values):
        raise ValueError('Invalid hybrid feature vector')
    return values


def development_roles(rows):
    """Balance counts by entire notebook/writer group, using no text or errors."""
    groups=defaultdict(list)
    for row in rows: groups[group(row)].append(row['sample_id'])
    roles=['fit','calibration','audit'];targets=dict(zip(roles,[.6*len(rows),.2*len(rows),.2*len(rows)]))
    counts={r:0 for r in roles};assigned={}
    order=sorted(groups,key=lambda g:(-len(groups[g]),hashlib.sha256(('54101|'+g).encode()).hexdigest()))
    for key in order:
        role=min(roles,key=lambda r:(counts[r]/max(1,targets[r]),roles.index(r)))
        assigned[key]=role;counts[role]+=len(groups[key])
    if min(counts.values())<10: raise ValueError('Too few lines for fixed fit/calibration/audit roles')
    return dict(sample_roles={r['sample_id']:assigned[group(r)] for r in rows},
        group_roles=assigned,line_counts=counts,group_counts=dict(Counter(assigned.values())),
        rule='largest groups first; balance 60/20/20 counts; ties SHA256(seed|group), then role order; no label/error stratification')


def complementarity(rows,ar,mdiff):
    counts=Counter();oracle={};details=[]
    for row in rows:
        sid=row['sample_id'];a=score_pair(row['text'],ar[sid]['text'],no_harakat_v1=True)
        d=score_pair(row['text'],mdiff[sid]['text'],no_harakat_v1=True)
        counts['mdiff_better_lines']+=d['char_distance']<a['char_distance']
        counts['ar_better_lines']+=a['char_distance']<d['char_distance']
        counts['equal_character_distance_lines']+=a['char_distance']==d['char_distance']
        counts['mdiff_exact_ar_wrong']+=bool(d['exact_lines'] and not a['exact_lines'])
        counts['ar_exact_mdiff_wrong']+=bool(a['exact_lines'] and not d['exact_lines'])
        counts['both_exact']+=bool(a['exact_lines'] and d['exact_lines'])
        counts['both_wrong']+=not a['exact_lines'] and not d['exact_lines']
        winner='mdiff' if d['char_distance']<a['char_distance'] else 'ar'
        oracle[sid]=(mdiff if winner=='mdiff' else ar)[sid]
        details.append(dict(sample_id=sid,ar_char_edits=a['char_distance'],mdiff_char_edits=d['char_distance'],
            ar_exact=bool(a['exact_lines']),mdiff_exact=bool(d['exact_lines']),diagnostic_oracle_source=winner))
    am=metrics(rows,ar);dm=metrics(rows,mdiff);om=metrics(rows,oracle)
    return dict(counts=dict(counts),ar=am,mdiff=dm,oracle=om,
        oracle_character_edit_headroom=am['char_distance']-om['char_distance'],
        oracle_cer_percentage_point_headroom=100*(am['cer']-om['cer']),
        oracle_is_usable_model=False,warning='Reference-dependent line-selection upper bound only; never an inference policy'),details


def fit_gate(rows,ar,mdiff):
    matrix=np.asarray([features(r,ar[r['sample_id']],mdiff[r['sample_id']]) for r in rows],dtype=np.float64)
    target=np.asarray([(score_pair(r['text'],ar[r['sample_id']]['text'],no_harakat_v1=True)['char_distance']-
                        score_pair(r['text'],mdiff[r['sample_id']]['text'],no_harakat_v1=True)['char_distance'])/
                       max(1,len(ar[r['sample_id']]['text']),len(mdiff[r['sample_id']]['text'])) for r in rows])
    mean=matrix.mean(0);scale=matrix.std(0);scale[scale<1e-8]=1.
    design=np.column_stack((np.ones(len(rows)),(matrix-mean)/scale))
    penalty=np.eye(design.shape[1])*RECIPE['regularization'];penalty[0,0]=0.
    weights=np.linalg.solve(design.T@design+penalty,design.T@target)
    return dict(version=VERSION,feature_names=list(FEATURE_NAMES),mean=mean.tolist(),scale=scale.tolist(),
        intercept=float(weights[0]),coefficients=weights[1:].tolist(),regularization=RECIPE['regularization'],
        training_lines=len(rows),score_is_probability=False,target=RECIPE['target'])


def score(gate,record,ar,mdiff):
    if gate['version']!=VERSION or gate['feature_names']!=list(FEATURE_NAMES): raise ValueError('Wrong selector feature schema')
    value=gate['intercept']+float(((np.asarray(features(record,ar,mdiff))-gate['mean'])/gate['scale'])@gate['coefficients'])
    if not math.isfinite(value): raise ValueError('Nonfinite selector score')
    return value


def select(rows,ar,mdiff,gate,threshold):
    result={}
    for row in rows:
        sid=row['sample_id'];a,d=ar[sid],mdiff[sid]
        value=score(gate,row,a,d)
        use=False;reason='ar_default'
        if not valid(d): reason='mdiff_unavailable'
        elif not valid(a): use=True;reason='ar_unavailable'
        elif a['text']==d['text']: reason='identical_outputs'
        elif threshold is not None and value>threshold: use=True;reason='ridge_score_exceeds_threshold'
        source=d if use else a
        result[sid]=dict(sample_id=sid,text=source['text'],status=source['status'],has_eos=source.get('has_eos',False),
            length_failure=source.get('length_failure',False),selected_model='mdiff' if use else 'ar',
            gate_score=value,threshold=threshold,reason=reason)
    return result


def evaluate(rows,ar,predictions):
    chosen={r['sample_id']:predictions[r['sample_id']] for r in rows}
    reference={r['sample_id']:ar[r['sample_id']] for r in rows}
    m=metrics(rows,chosen);baseline=metrics(rows,reference);changes=damage(rows,reference,chosen)
    conditions=dict(strictly_fewer_character_edits_than_ar=m['char_distance']<baseline['char_distance'],
        no_more_word_edits_than_ar=m['word_distance']<=baseline['word_distance'],
        zero_ar_exact_lines_lost=changes['exact_lines_lost']==0,
        complete_successful_coverage=m['failed_lines']==m['missing_lines']==0,
        all_outputs_have_eos=m['eos_failures']==0,no_length_failures=m['length_failures']==0)
    return dict(metrics=m,ar_metrics=baseline,changes_vs_ar=changes,conditions=conditions,
        passes_ar_gate=all(conditions.values()),switched_lines=sum(predictions[r['sample_id']]['selected_model']=='mdiff' for r in rows))


def calibrate(rows,ar,mdiff,gate):
    trials=[]
    for threshold in [None,*RECIPE['threshold_grid']]:
        trial=evaluate(rows,ar,select(rows,ar,mdiff,gate,threshold));trial['threshold']=threshold;trials.append(trial)
    passing=[t for t in trials if t['passes_ar_gate']]
    if passing:
        chosen=min(passing,key=lambda t:(t['metrics']['char_distance'],t['metrics']['word_distance'],
            -t['metrics']['exact_lines'],t['switched_lines'],-t['threshold']))
        reason='best passing calibration threshold on frozen grid'
    else:
        chosen=next(t for t in trials if t['threshold']==RECIPE['raw_default_threshold'])
        reason='no threshold passed; report the predefined threshold-zero raw candidate as rejected, without hiding it behind AR fallback'
    return dict(threshold=chosen['threshold'],chosen=chosen,trials=trials,
        calibration_passes_ar_gate=chosen['passes_ar_gate'],selection_reason=reason)
