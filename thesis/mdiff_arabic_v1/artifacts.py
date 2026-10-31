"""Immutable identities, literal metrics, comparator provenance and frozen gate."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch

from thesis.diffusion.data import file_hash, image_info, resolve_relative
from thesis.diffusion.images import ImageConfig, prepare_images, augment_image, AugmentationConfig, augmentation_seed
from thesis.diffusion.metrics import aggregate, score_pair, align
from thesis.diffusion.tokenizer import CharacterTokenizer, read_literal, training_text

VERSION = 'mdiff_arabic_v1'
SEED = 54101
HF_REVISION = '70c320bb8ab539ecc08f820ef94ba7679b6e40f2'
UPSTREAM_COMMIT = '1ccfc6ee6161f7133b192e16af3a9265273997ba'
ASSETS = {'best.pth':'ebc96f11e5739eae98491f7c8c140157ed12b35dafa220511ef9d399cc20a937',
          'config.yml':'0abdd478db64fa06b0574f56f8c8620800bab511af0af94861b9ea73bcb219bf'}
POLICY = dict(selection='minimum validation character edits, then word edits, then most exact lines, then earliest epoch',
    promotion='strictly fewer character edits AND no more word edits vs frozen v4 AND zero damaged lines in union of v4/v2.2 exact lines',
    additional_integrity='complete successful coverage, EOS on every line, no overlength/OOV/image failures',
    text='no_harakat_v1: remove only U+064B-U+0652; all other codepoints and whitespace literal',
    test='historical development-used comparison; both primary arms must pass validation; no automatic promotion')
RECIPE = dict(version=VERSION, capacity=144, height=32, max_width=640, dim=256, layers=3,
    dropout=.1, augmentation='trocr_views_v1', batch_size=1, accumulation=16, workers=0,
    seed=SEED, maximum_epochs=200, patience=30, warmup_epochs=5, encoder_frozen_epochs=5,
    decoder_lr=.0003, encoder_lr=.00001, minimum_lr_fraction=.1, weight_decay=.01,
    gradient_clip=1., precision='bf16_on_cuda_fp32_on_cpu', sampler_steps=3,
    diffusion_training_decoder_passes=2, ar_training_decoder_passes=2,
    correction_weight=.5, masked_weight=.5, tail_weight=.1, policy=POLICY)
OVERFIT_CRITERION = dict(minimum_updates=100, maximum_updates=1200, maximum_seconds=900,
    samples=4, cer_at_most=.05, exact_at_least=.75, image_cer_advantage_at_least=.01,
    correction_cer_at_most=.05, correction_exact_at_least=.75, preservation_exact_at_least=.75)
GEOMETRY_PROTOCOL = 'geometry_matched_v1'


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',',':'))+'\n').encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('xb') as out: out.write(canonical(value))
    else:
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_bytes(canonical(value))
        temp.replace(path)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as out:
        for row in rows: out.write(canonical(row))


def environment(device):
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; environment changed')
    if device == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('Frozen CUDA recipe requires BF16 support; use an explicit new recipe/environment for other hardware')
    packages = {d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()}
    return dict(executable=str(Path(sys.executable).resolve()), python=sys.version,
        packages=dict(sorted(packages.items())), torch=torch.__version__, cuda=torch.version.cuda,
        device=device, gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None,
        bf16_supported=torch.cuda.is_bf16_supported() if device=='cuda' else None,
        torch_threads=2)


def source_files(repo, runner):
    paths = list((repo/'thesis/mdiff_arabic_v1').rglob('*.py'))
    paths += [repo/'thesis/mdiff_arabic_v1/vendor'/f for f in ('LICENSE','NOTICE.md')]
    paths += [repo/'thesis/diffusion'/f for f in ('data.py','tokenizer.py','images.py','metrics.py','paddle_conditioning.py')]
    paths += [repo/'thesis/scripts/common.py', repo/'thesis/scripts/run_mdiff_arabic_v1.py',
              repo/'tests/test_mdiff_arabic_v1.py', runner]
    return {str(p.resolve()):file_hash(p) for p in sorted(paths)}


def group(row):
    if row.get('writer_id'):
        return 'writer:'+str(row['writer_id'])
    sid = row['sample_id']
    body = sid.split('_',3)[-1]
    match = re.search(r'^(.+?)__(First|Second|Third|Fourth|Unknown)(?:__(?:Grade|grade))?__Notebook_([^_]+)(?:_(Male|Female))?', body)
    if not match:
        return 'page:'+sid.split('_',3)[-1].rsplit('_line_',1)[0]
    return 'notebook_proxy:'+'|'.join(v or 'unknown' for v in match.groups())


def page(row):
    return row['sample_id'].split('_',1)[-1].rsplit('_line_',1)[0]


def tiny_examples(train_rows, protocol='legacy_v1'):
    """Choose TRAIN diagnostics without model scores or development/test labels.

    Geometry matching uses the encoder's ceil(resized_width/4) extent. After
    blanking, all four encoder inputs then have exactly the same pixels/shape.
    Shuffling only changes content within that common extent. This removes the
    unique-width identifier present in the legacy four-example diagnostic.
    """
    candidates=[r for r in train_rows if 4<=len(training_text(r['text'],'no_harakat_v1'))<=18]
    key=lambda r:hashlib.sha256((str(SEED)+r['sample_id']).encode()).hexdigest()
    if protocol=='legacy_v1':
        examples=sorted(candidates,key=key)[:4]
        return examples,dict(version=protocol,geometry_matched=False)
    if protocol!=GEOMETRY_PROTOCOL: raise ValueError('Unknown readiness protocol')
    buckets=defaultdict(list)
    for r in candidates:
        width=max(1,round(r['width']*32/r['height']))
        buckets[math.ceil(width/4)].append(r)
    viable={w:rows for w,rows in buckets.items()
            if len({training_text(r['text'],'no_harakat_v1') for r in rows})>=4}
    if not viable: raise ValueError('No TRAIN geometry bucket with four distinct diagnostic labels')
    width=min(viable,key=lambda w:(-len({training_text(r['text'],'no_harakat_v1') for r in viable[w]}),w))
    ordered=sorted(viable[width],key=key)
    selected=[];labels=set();groups=set()
    # Prefer distinct notebook/writer groups, with a deterministic fallback.
    for require_new_group in (True,False):
        for r in ordered:
            label=training_text(r['text'],'no_harakat_v1')
            if label in labels or (require_new_group and group(r) in groups): continue
            selected.append(r);labels.add(label);groups.add(group(r))
            if len(selected)==4: break
        if len(selected)==4: break
    manifest=dict(version=protocol,geometry_matched=True,height=32,encoder_input_width=width*4,
        distinct_labels=len(labels),distinct_groups=len(groups),candidate_distinct_labels=len({training_text(r['text'],'no_harakat_v1') for r in viable[width]}),
        selection='largest TRAIN bucket by distinct eligible labels; tie lowest aligned width; SHA256(seed+sample_id), distinct labels, prefer distinct groups',
        sample_ids=[r['sample_id'] for r in selected],thresholds_changed=False,
        guarantee='all blank encoder inputs identical in shape/pixels; original images keep aspect ratio and existing <=3px alignment border')
    return selected,manifest


def load_data(drive, *, test_authorized=False, readiness_protocol='legacy_v1'):
    split, dataset = drive/'splits_70_05_25_v3', drive/'ground_truth_lines'
    lock = read(split/'split_lock.json')
    parts, hashes, ids, pixels = {}, {}, set(), {}
    for partition, count in [('train',2103), ('validation',150), ('test',752)]:
        path = split/(partition+'.jsonl')
        hashes[partition] = file_hash(path)
        if hashes[partition] != lock['hashes'][path.name]:
            raise ValueError('Canonical split hash mismatch: '+partition)
        rows = jsonl(path)
        if len(rows)!=count or len({r['sample_id'] for r in rows})!=count:
            raise ValueError('Canonical split count/duplicate mismatch: '+partition)
        for r in rows:
            sid = r['sample_id']
            if sid in ids: raise ValueError('Cross-split sample overlap: '+sid)
            ids.add(sid)
            pix = r['pixels_sha256']
            if pix in pixels and pixels[pix]!=partition: raise ValueError('Cross-split identical pixels')
            pixels[pix] = partition
            if partition=='test' and not test_authorized:
                # Identity-only audit: do not inspect test labels/lengths/OOVs.
                for key in list(r):
                    if key not in ('sample_id','writer_id','metadata_writer_candidate','pixels_sha256','image_sha256','image_path'):
                        del r[key]
                continue
            im = resolve_relative(dataset,r['image_path'])
            label = resolve_relative(dataset,r['label_path'])
            if file_hash(im)!=r['image_sha256'] or file_hash(label)!=r['label_sha256']:
                raise ValueError('Image/label binding mismatch: '+sid)
            if read_literal(label)!=r['text'] or image_info(im)['pixels_sha256']!=r['pixels_sha256']:
                raise ValueError('Pixel/literal binding mismatch: '+sid)
        parts[partition] = rows
    # Capacity and vocabulary are derived exclusively from TRAIN.
    texts = [training_text(r['text'],'no_harakat_v1') for r in parts['train']]
    tokenizer = CharacterTokenizer(sorted(set(''.join(texts)),key=ord))
    image_config = ImageConfig(height=32,max_width=640,patch_size=4)
    issues=[]
    for partition in ['train','validation']+(['test'] if test_authorized else []):
        for r in parts[partition]:
            text=training_text(r['text'],'no_harakat_v1')
            unknown=sorted(set(text)-set(tokenizer.characters), key=ord)
            if unknown: issues.append(dict(partition=partition,sample_id=r['sample_id'],kind='unsupported_characters',codepoints=[f'U+{ord(c):04X}' for c in unknown]))
            if len(text)+1>144: issues.append(dict(partition=partition,sample_id=r['sample_id'],kind='overlength',length=len(text)))
            if round(r['width']*32/r['height'])>640: issues.append(dict(partition=partition,sample_id=r['sample_id'],kind='image_limit'))
    overlaps={}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        overlaps[a+'_'+b]=dict(pages=len({page(r) for r in parts[a]} & {page(r) for r in parts[b]}),
                              notebook_or_writer_groups=len({group(r) for r in parts[a]} & {group(r) for r in parts[b]}))
    # A debug-only holdout within TRAIN keeps entire notebook proxies together.
    groups=sorted({group(r) for r in parts['train']},key=lambda g:hashlib.sha256((str(SEED)+g).encode()).hexdigest())
    holdout_groups=set(groups[:max(1,math.ceil(len(groups)*.1))])
    holdout=[r['sample_id'] for r in parts['train'] if group(r) in holdout_groups]
    examples,tiny_manifest=tiny_examples(parts['train'],readiness_protocol)
    evidence=dict(counts={p:len(v) for p,v in parts.items()}, split_hashes=hashes,
        split_lock_sha256=file_hash(split/'split_lock.json'), dataset_fingerprint=lock['dataset_fingerprint'],
        train_max_literal_length=max(len(r['text']) for r in parts['train']),
        train_max_resized_width=max(round(r['width']*32/r['height']) for r in parts['train']),
        geometry_margin='32px height, width640 vs training maximum489; capacity144 vs maximum109+EOS',
        vocabulary_origin='training no_harakat_v1 labels only',tokenizer=tokenizer.as_dict(),
        coverage_issues=issues,group_overlap=overlaps,writer_identity_verified=all(r.get('writer_id') for r in parts['train']),
        internal_debug_holdout=dict(sample_ids=holdout,groups=sorted(holdout_groups),scope='TRAIN-derived debugging only; primary training uses all canonical TRAIN'),
        tiny_sample_ids=[r['sample_id'] for r in examples],
        tiny_diagnostic_protocol=tiny_manifest,
        untouched_evaluation='none established: all canonical 3005 lines are historically development-used; other material has no verified untouched provenance',
        test_references_used=test_authorized)
    return parts, tokenizer, image_config, evidence


def batch(rows, dataset, tokenizer, config, device, epoch=0):
    images=[]
    for r in rows:
        with Image.open(resolve_relative(dataset,r['image_path'])) as image:
            image=image.convert('RGB')
        if epoch:
            image,_=augment_image(image,AugmentationConfig('trocr_views_v1'),augmentation_seed(r['sample_id'],epoch,SEED),view_index=(epoch-1)%3)
        images.append(image)
    pixels,widths=prepare_images(images,config)
    targets=torch.tensor([tokenizer.encode(training_text(r['text'],'no_harakat_v1'),144,eos_fill=False) for r in rows])
    return pixels.to(device), widths.to(device), targets.to(device)


def indexed(rows, expected):
    by_id={r['sample_id']:r for r in rows}
    if len(by_id)!=len(rows) or set(by_id)!={r['sample_id'] for r in expected}:
        raise ValueError('Prediction membership/duplicates differ from manifest')
    return by_id


def comparator_sources(drive, partition):
    frozen=drive/'baselines/agreement_refinement_v2_2_frozen_20260918'
    v4=drive/'runs/span_hybrid_verifier_v4_s53004'
    return dict(paddle=drive/f'paddle_70_05_25_v3/{partition}/predictions.jsonl',
        frozen_v2_2=frozen/f'agreement_refinement_v2_2/{partition}_predictions.jsonl',
        span_hybrid_verifier_v4=v4/('validation_predictions.jsonl' if partition=='validation' else 'test/predictions.jsonl'))


def verify_comparators(drive, rows, partition):
    """IDs alone are insufficient: verify split, source chain and image/label hashes."""
    from thesis.diffusion.paddle_conditioning import bind_paddle_predictions
    frozen=drive/'baselines/agreement_refinement_v2_2_frozen_20260918'
    v4=drive/'runs/span_hybrid_verifier_v4_s53004'
    lock=file_hash(drive/'splits_70_05_25_v3/split_lock.json')
    manifest=read(frozen/'manifest.json')
    if manifest['provenance']['split_lock_sha256']!=lock:
        raise ValueError('Frozen baseline split lock mismatch')
    # Verify immutable bundle, including model/policy and cached predictions.
    for rel, sha in manifest['copied_file_sha256'].items():
        if file_hash(frozen/rel)!=sha: raise ValueError('Frozen comparator modified: '+rel)
    bound,paddle_binding=bind_paddle_predictions(drive/f'paddle_70_05_25_v3/{partition}',rows,drive/'ground_truth_lines',drive/f'splits_70_05_25_v3/{partition}.jsonl')
    summary=read(frozen/f'agreement_refinement_v2_2/{partition}_run_summary.json')
    if summary['split_lock_sha256']!=lock or summary['completed']!=len(rows) or summary['failed']:
        raise ValueError('v2.2 prediction coverage/source mismatch')
    prepared=read(v4/'prepared.json')
    binding=prepared['binding']
    parent=binding['source_binding_v3']
    if binding['split_lock_sha256']!=lock or parent['split_lock_sha256']!=lock or parent['frozen_bundle_manifest_sha256']!=file_hash(frozen/'manifest.json'):
        raise ValueError('v4 input chain mismatch')
    if parent['frozen_validation_sha256']!=manifest['copied_file_sha256']['agreement_refinement_v2_2/validation_predictions.jsonl']:
        raise ValueError('v4 parent prediction chain mismatch')
    selected=read(v4/'selection.json')
    report=read(v4/'validation_report.json')
    if file_hash(v4/'prepared.json')!=report['prepared_sha256'] or file_hash(v4/'validation_report.json')!=selected['validation_report_sha256']:
        raise ValueError('v4 frozen selection/report mismatch')
    if file_hash(v4/f"verifier-{report['selected_verifier_epoch']:04d}.pt")!=report['model_sha256']:
        raise ValueError('v4 model mismatch')
    sources=comparator_sources(drive,partition)
    if partition=='test':
        test_report=read(v4/'test/report.json')
        # Existing test report identity is frozen with predictions below, never
        # used for selection or troubleshooting before this authorized action.
        if test_report.get('partition')!='test': raise ValueError('v4 historical test report mismatch')
    predictions={name:indexed(jsonl(path),rows) for name,path in sources.items()}
    predictions['paddle']={r['sample_id']:dict(sample_id=r['sample_id'],text=r['conditioning_text'],status='completed') for r in bound}
    for values in predictions.values():
        for r in values.values():
            if not isinstance(r.get('text',r.get('prediction')),str) or r.get('status') not in ('completed','ok'):
                raise ValueError('Frozen comparator has invalid/failed prediction')
    provenance=dict(files={str(p):file_hash(p) for p in sources.values()},paddle=paddle_binding,
        v4_model_sha256=report['model_sha256'],v4_prepared_sha256=file_hash(v4/'prepared.json'),
        v4_selection_sha256=file_hash(v4/'selection.json'),v4_report_sha256=file_hash(v4/'validation_report.json'),
        v2_bundle_sha256=file_hash(frozen/'manifest.json'),matching='canonical split lock + per-sample image/pixel/literal label hashes + Paddle input fingerprints + frozen v2/v4 input chains')
    return predictions, provenance


def metrics(rows, predictions):
    values=indexed(list(predictions.values()),rows)
    pairs=[]
    failed=0
    for r in rows:
        p=values[r['sample_id']]
        # Keep failed/no-EOS raw strings and every denominator. Missing rows are
        # rejected by indexed(), never silently omitted from scoring.
        pairs.append((r['text'],p.get('text',p.get('prediction',''))))
        failed+=p.get('status','completed') not in ('completed','ok')
    result=aggregate(pairs,no_harakat_v1=True)
    result.update(expected_lines=len(rows),completed_lines=len(rows)-failed,failed_lines=failed,missing_lines=0,
        eos_failures=sum(not p.get('has_eos',True) for p in values.values()),
        length_failures=sum(bool(p.get('length_failure',False)) for p in values.values()),
        runtime_seconds=sum(p.get('seconds',0) for p in values.values()))
    whitespace=Counter()
    for reference,prediction in pairs:
        ref=training_text(reference,'no_harakat_v1');pred=training_text(prediction,'no_harakat_v1')
        for op,_,expected,actual in align(ref,pred):
            if op!='equal' and ((expected is not None and expected.isspace()) or (actual is not None and actual.isspace())):
                whitespace['whitespace_'+op+'s']+=1
        whitespace['leading_whitespace_changed_lines']+=(len(ref)-len(ref.lstrip()))!=(len(pred)-len(pred.lstrip()))
        whitespace['trailing_whitespace_changed_lines']+=(len(ref)-len(ref.rstrip()))!=(len(pred)-len(pred.rstrip()))
    result['whitespace_diagnostics']=dict(whitespace)
    measured=[p['seconds'] for p in values.values() if 'seconds' in p]
    result['mean_seconds_per_line']=sum(measured)/len(measured) if measured else None
    return result


def damage(rows, baseline, predictions):
    lost,gained,improved,worsened,unchanged=[],[],[],[],[]
    for r in rows:
        sid=r['sample_id']
        before=score_pair(r['text'],baseline[sid]['text'],no_harakat_v1=True)
        after=score_pair(r['text'],predictions[sid]['text'],no_harakat_v1=True)
        if before['exact_lines'] and not after['exact_lines']: lost.append(sid)
        if after['exact_lines'] and not before['exact_lines']: gained.append(sid)
        (improved if after['char_distance']<before['char_distance'] else worsened if after['char_distance']>before['char_distance'] else unchanged).append(sid)
    return dict(exact_lines_lost=len(lost),exact_lines_gained=len(gained),lost_sample_ids=lost,gained_sample_ids=gained,
                improved_lines=len(improved),worsened_lines=len(worsened),unchanged_lines=len(unchanged))


def eligibility(rows, baselines, predictions):
    candidate=metrics(rows,predictions)
    v4=metrics(rows,baselines['span_hybrid_verifier_v4'])
    changes={name:damage(rows,baselines[name],predictions) for name in ('frozen_v2_2','span_hybrid_verifier_v4')}
    union=set(changes['frozen_v2_2']['lost_sample_ids'])|set(changes['span_hybrid_verifier_v4']['lost_sample_ids'])
    conditions=dict(strictly_fewer_character_edits=candidate['char_distance']<v4['char_distance'],
        no_more_word_edits=candidate['word_distance']<=v4['word_distance'],zero_baseline_exact_lines_damaged=len(union)==0,
        complete_successful_coverage=candidate['failed_lines']==candidate['missing_lines']==0,
        all_outputs_have_eos=candidate['eos_failures']==0,no_length_failures=candidate['length_failures']==0)
    return dict(validation_passes_release_gate=all(conditions.values()),conditions=conditions,raw_metrics=candidate,
        damage=changes,baseline_exact_union_lines_lost=len(union),baseline_exact_union_lost_sample_ids=sorted(union))


def grouped_intervals(rows, predictions, baseline, seed=SEED, repetitions=1000):
    groups=defaultdict(list)
    for r in rows:
        sid=r['sample_id']
        a=score_pair(r['text'],predictions[sid]['text'],no_harakat_v1=True)
        b=score_pair(r['text'],baseline[sid]['text'],no_harakat_v1=True)
        groups[group(r)].append((a['char_distance'],a['reference_chars'],a['char_distance']-b['char_distance']))
    sums=np.array([np.sum(v,axis=0) for v in groups.values()])
    if len(sums)<2: return dict(available=False,reason='fewer than two groups')
    rng=np.random.default_rng(seed)
    draws=sums[rng.integers(0,len(sums),size=(repetitions,len(sums)))].sum(1)
    return dict(available=True,groups=len(sums),group_type='writer if known else notebook proxy; identity not proven',
        repetitions=repetitions,seed=seed,cer_95=np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist(),
        paired_cer_delta_vs_v4_95=np.quantile(draws[:,2]/draws[:,1],[.025,.975]).tolist(),
        limitation='descriptive development-used benchmark interval; does not repair group leakage or repeated inspection')


def require_equal(current, frozen, label):
    if current!=frozen:
        if label=='environment':
            old,new=frozen.get('packages',{}),current.get('packages',{})
            changed={p:[old.get(p),new.get(p)] for p in old.keys()|new.keys() if old.get(p)!=new.get(p)}
            raise ValueError(f'Frozen environment differs; changed packages={changed}; old/new executable={frozen.get("executable")}/{current.get("executable")}. Use an explicit new experiment/environment; never refreeze this run.')
        raise ValueError('Frozen '+label+' differs; use a NEW run name after review')


def enforce_gate(report, seal, *, expected_binding, report_sha, checkpoint_sha, peer_checkpoint_sha):
    if report.get('partition')!='validation' or report.get('test_used') is not False:
        raise ValueError('Gate must be a validation-only report')
    if report.get('validation_passes_release_gate') is not True:
        raise ValueError('STOP: validation_passes_release_gate is not true')
    if not report.get('gate_conditions') or not all(v is True for v in report['gate_conditions'].values()):
        raise ValueError('STOP: a recorded gate condition failed')
    expected=dict(binding_sha256=expected_binding,validation_report_sha256=report_sha,
        selected_checkpoint_sha256=checkpoint_sha,peer_selected_checkpoint_sha256=peer_checkpoint_sha)
    for key,value in expected.items():
        if seal.get(key)!=value: raise ValueError('Frozen gate binding mismatch: '+key)
