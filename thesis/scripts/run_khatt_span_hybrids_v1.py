"""Compare MDiff and fine-tuned AR as correction sources for the mixed Transformer."""
from __future__ import annotations

import argparse
import csv
from itertools import product
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from thesis.ar_mdiff_fusion_v2 import inference, logic
from thesis.diffusion.images import ImageConfig
from thesis.diffusion.tokenizer import CharacterTokenizer
from thesis.khatt_ablation_v1 import workflow as ar
from thesis.khatt_ablation_v1.data import child_row
from thesis.mdiff_arabic_v1.artifacts import canonical, digest, environment, file_hash, indexed, jsonl, metrics, read, write
from thesis.mdiff_arabic_v1.experiment import autocast, seed_all


BASE = 'khatt100_ours'
MDIFF = 'mdiff_khatt100_ours'
FINETUNE = 'finetune_lr3e4'
DONORS = {'hybrid_span_mdiff': MDIFF, 'hybrid_span_finetune': FINETUNE}
CHECKPOINTS = {
    BASE: ('khatt_ablations/khatt_ar_v1fix1_s54101/runs/khatt100_ours', 40,
           'fed32927b03a240bdc855ec86ec0bfc541cad77ea5cbefd6d55f1ac74f5b9291', 'ar'),
    MDIFF: ('khatt_ablation_mdiff/khatt_mdiff_v1_s54101', 73,
            '41083edb98284841af52e6b5b41e0ddfd4748e010998053c89fc2c8a898f8495', 'mdiff'),
    FINETUNE: ('khatt_ar_finetune/khatt_ar_finetune_v1_s54101/finetune_lr3e4', 52,
               '3b57fbc08a3af292c4d08c200d5d437361bba91eeafae9e62e57f18044591611', 'ar'),
}
GRID = {'donor_weight': [0., .25, .5, .75, 1.], 'margin': [0., .025, .05, .10, .15, .25]}
POLICY = ('Require fewer character edits, fewer word edits, zero previously exact lines lost and full EOS coverage '
          'on the 150 children validation lines. Keep the base if no setting qualifies.')
# The original validation run used this revision. The only changes affecting
# Test are its cache path and verification of this specific predecessor.
PRE_SHORT_CACHE_RUNNER_SHA256 = '9159cfe7d1f21ccbff08eca51a3546e28779ac6fde19daf37921ec347c272438'


def verify(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f'File hash mismatch: {path}')


def context(project, device):
    repo = Path(__file__).resolve().parents[2]
    drive = project / 'Diffusion/drive_all'
    parent = drive / 'khatt_ablations/khatt_ar_v1fix1_s54101'
    output = drive / 'khatt_span_hybrids/khatt_span_hybrids_v1'
    provenance_path = repo / 'thesis/experiments/khatt_ablation_v1/ARTIFACTS.json'
    provenance = read(provenance_path)
    for name in ('validation.jsonl', 'tokenizer.json'):
        path = parent / name
        verify(path, provenance['local_artifacts'][path.relative_to(project).as_posix()]['sha256'])
    models = {}
    for name, (relative, epoch, sha256, arm) in CHECKPOINTS.items():
        folder = drive / relative
        complete = read(folder / 'complete.json')
        best = complete['best']
        if (best['epoch'], best['sha256']) != (epoch, sha256):
            raise ValueError(f'Selected checkpoint changed for {name}')
        verify(folder / best['checkpoint'], sha256)
        verify(folder / best['predictions'], best['predictions_sha256'])
        models[name] = dict(folder=folder, best=best, arm=arm, binding_sha256=complete['binding_sha256'])
    runtime = environment(device)
    runtime['packages'] = {name: runtime['packages'].get(name) for name in ('torch', 'numpy', 'pillow')}
    sources = [Path(__file__), project / 'Diffusion/run_khatt_span_hybrids_v1.ps1']
    sources.extend(repo / name for name in (
        'thesis/ar_mdiff_fusion_v2/inference.py', 'thesis/ar_mdiff_fusion_v2/logic.py',
        'thesis/khatt_ablation_v1/workflow.py', 'thesis/khatt_ablation_v1/data.py',
        'thesis/mdiff_arabic_v1/model.py', 'thesis/mdiff_arabic_v1/experiment.py',
        'thesis/mdiff_arabic_v1/vendor/svtrv2.py', 'thesis/mdiff_arabic_v1/vendor/common_upstream.py',
        'thesis/mdiff_arabic_v1/artifacts.py', 'thesis/diffusion/metrics.py',
        'thesis/diffusion/tokenizer.py', 'thesis/diffusion/images.py', 'thesis/diffusion/data.py',
    ))
    binding = dict(version='khatt_span_hybrids_v1', provenance_sha256=file_hash(provenance_path),
                   checkpoints={name: dict(epoch=m['best']['epoch'], sha256=m['best']['sha256'],
                                           validation_predictions_sha256=m['best']['predictions_sha256'],
                                           binding_sha256=m['binding_sha256']) for name, m in models.items()},
                   source_sha256={path.name: file_hash(path) for path in sources}, runtime=runtime,
                   grid=GRID, calibration_policy=POLICY)
    config = ImageConfig(**provenance['image_config'])
    tokenizer = CharacterTokenizer.load(parent / 'tokenizer.json')
    return dict(args=argparse.Namespace(device=device), project=project, drive=drive, parent=parent,
                output=output, provenance=provenance, models=models, config=config, tokenizer=tokenizer,
                binding=binding, binding_sha256=digest(binding))


def predictions_for(values, rows, *, require_image_hash=True):
    predictions = indexed(values, rows)
    for row in rows:
        prediction = predictions[row['sample_id']]
        if require_image_hash and prediction.get('image_sha256') != row['image_sha256']:
            raise ValueError('Saved prediction refers to a different image')
    return predictions


def validation_inputs(ctx):
    rows = jsonl(ctx['parent'] / 'validation.jsonl')
    if len(rows) != 150 or any(row['source'] != 'children' for row in rows):
        raise ValueError('Expected 150 children validation lines')
    ar.verify_images(rows)
    predictions = {name: predictions_for(jsonl(model['folder'] / model['best']['predictions']), rows)
                   for name, model in ctx['models'].items()}
    return rows, predictions


def historical_inputs(ctx):
    path = ctx['drive'] / 'splits_70_05_25_v3/test.jsonl'
    verify(path, ctx['provenance']['inputs'][path.relative_to(ctx['project']).as_posix()])
    identity = ctx['parent'] / 'test_identity.jsonl'
    verify(identity, ctx['provenance']['local_artifacts'][identity.relative_to(ctx['project']).as_posix()]['sha256'])
    rows = [child_row(row, ctx['drive'] / 'ground_truth_lines') for row in jsonl(path)]
    if len(rows) != 752 or [r['sample_id'] for r in rows] != [r['sample_id'] for r in jsonl(identity)]:
        raise ValueError('Expected the original 752 historical lines')
    parent = ctx['parent'] / 'historical_752/predictions.jsonl'
    verify(parent, ctx['provenance']['local_artifacts'][parent.relative_to(ctx['project']).as_posix()]['sha256'])
    saved = jsonl(parent)
    predictions = {name: predictions_for([p for p in saved if p['model'] == name], rows, require_image_hash=name != 'span_merge')
                   for name in (BASE, 'span_merge')}
    sources = {BASE: file_hash(parent), 'span_merge': file_hash(parent)}
    for name in (MDIFF, FINETUNE):
        model = ctx['models'][name]
        folder = model['folder'] if name == MDIFF else model['folder'].parent
        result = folder / 'historical_752'
        seal = read(result / 'seal.json')
        mdiff = name == MDIFF
        verify(result / 'report.json', seal['report_sha256' if mdiff else 'report.json'])
        verify(result / 'predictions.jsonl', seal['predictions_sha256' if mdiff else 'predictions.jsonl'])
        report = read(result / 'report.json')
        identity = report['identity']
        checkpoint_sha256 = identity['checkpoint_sha256'] if mdiff else identity['checkpoints'][name]
        manifest_key = 'historical_sha256' if mdiff else 'test_manifest_sha256'
        if (checkpoint_sha256 != model['best']['sha256'] or identity[manifest_key] != file_hash(path)
                or identity['binding_sha256'] != model['binding_sha256']):
            raise ValueError(f'Historical predictions do not match selected checkpoint for {name}')
        predictions[name] = predictions_for([p for p in jsonl(result / 'predictions.jsonl') if p['model'] == name], rows)
        sources[name] = file_hash(result / 'predictions.jsonl')
    return rows, predictions, sources


def propose(base, donor):
    if not (logic.valid(base) and logic.valid(donor)) or base['text'] == donor['text']:
        return []
    return [item for item in logic.candidates(base, donor)
            if 'span' in item['origins'] and item['text'] != base['text']]


@torch.inference_mode()
def score_images(pixels, widths, base, spans, models, tokenizer, donor_arm, device):
    texts = [base['text'], *(item['text'] for item in spans)]
    primary, donor = models
    with autocast(device):
        pm, pmask = primary.encode(pixels, widths)
        dm, dmask = donor.encode(pixels, widths)
        scored = []
        for text in texts:
            primary_score = inference.ar_score(primary, pm, pmask, tokenizer, text)['ar_mean_logp']
            if donor_arm == 'mdiff':
                donor_score = inference.md_score(donor, dm, dmask, tokenizer, text)['md_mean_logp']
            else:
                donor_score = inference.ar_score(donor, dm, dmask, tokenizer, text)['ar_mean_logp']
            scored.append(dict(text=text, base_score=primary_score, donor_score=donor_score))
    return scored


def load_models(ctx, donor):
    specs = (ctx['models'][name] for name in (BASE, donor))
    return tuple(inference.load_primary(m['folder'] / m['best']['checkpoint'], ctx['tokenizer'],
                                       m['arm'], ctx['args'].device) for m in specs)


def score_corpus(ctx, rows, inputs, variant, output, partition, identity):
    donor = DONORS[variant]
    cache = output / 'scores' / variant
    cache.mkdir(parents=True, exist_ok=True)
    models = None
    scores = {}
    for index, row in enumerate(rows, 1):
        sid = row['sample_id']
        base, alternative = (logic.prediction_only(inputs[name][sid]) for name in (BASE, donor))
        proposals = propose(base, alternative)
        if proposals:
            path = cache / (digest(sid) + '.json')
            key = digest(dict(identity=identity, variant=variant, sample_id=sid, image_sha256=row['image_sha256'],
                              base=base, donor=alternative))
            if path.exists():
                record = read(path)
                if record['key'] != key or record['sha256'] != digest(record['scores']):
                    raise ValueError('Score cache has a mismatched identity or digest')
            else:
                if models is None:
                    models = load_models(ctx, donor)
                image = {name: row[name] for name in ('sample_id', 'image_path', 'image_sha256')}
                pixels, widths = ar.image_batch(image, ctx)
                scored = score_images(pixels, widths, base, proposals, models, ctx['tokenizer'],
                                      ctx['models'][donor]['arm'], ctx['args'].device)
                record = dict(key=key, scores=scored, sha256=digest(scored))
                write(path, record)
            scores[sid] = record['scores']
        if index % 50 == 0 or index == len(rows):
            print(f'{variant} {partition} scored {index}/{len(rows)} lines', flush=True)
    return scores


def select(base, scores, config):
    output = dict(base)
    if config is None or not scores:
        return output
    weight = config['donor_weight']
    margin = config['margin']
    if scores[0]['text'] != output['text']:
        raise ValueError('Candidate ranking does not begin with the base transcription')
    if not all(math.isfinite(item[key]) for item in scores for key in ('base_score', 'donor_score')):
        raise ValueError('Nonfinite span score')
    def combined(item):
        return (1 - weight) * item['base_score'] + weight * item['donor_score']
    winner = max(scores, key=combined)
    if combined(winner) - combined(scores[0]) > margin:
        output.update(text=winner['text'], has_eos=True, length_failure=False, status='completed')
    return output


def apply_config(base, scores, config):
    return {sid: select(prediction, scores.get(sid), config) for sid, prediction in base.items()}


def fit(rows, base, scores):
    trials = []
    for weight, margin in product(GRID['donor_weight'], GRID['margin']):
        config = dict(donor_weight=weight, margin=margin)
        evaluated = logic.evaluate(rows, base, apply_config(base, scores, config))
        eligible = evaluated['passes_preservation_check'] and evaluated['metrics']['word_distance'] < evaluated['ar_metrics']['word_distance']
        trials.append(dict(config=config, metrics=evaluated['metrics'], eligible=eligible,
                           exact_lines_lost=evaluated['changes_vs_ar']['exact_lines_lost'],
                           exact_lines_gained=evaluated['changes_vs_ar']['exact_lines_gained'],
                           switched_lines=evaluated['switched_lines']))
    def rank(trial):
        m, config = trial['metrics'], trial['config']
        return (m['char_distance'], m['word_distance'], -m['exact_lines'], trial['switched_lines'],
                -config['margin'], config['donor_weight'])
    eligible = [trial for trial in trials if trial['eligible']]
    picked = min(eligible, key=rank) if eligible else None
    return dict(config=picked['config'] if picked else None, selected=picked,
                best_unconstrained=min(trials, key=rank), grid_results=trials)


def report_table(output, rows, predictions, variants, title):
    results = {name: metrics(rows, values) for name, values in predictions.items()}
    changes = {name: logic.evaluate(rows, predictions[BASE], predictions[name])['changes_vs_ar'] for name in variants}
    print(title)
    header = ('Model', 'Lines', 'CER %', 'WER %', 'Exact %', 'Sub', 'Ins', 'Del', 'No EOS', 'Exact +', 'Exact -')
    print('{:25} {:>5} {:>7} {:>7} {:>7} {:>5} {:>5} {:>5} {:>6} {:>7} {:>7}'.format(*header))
    with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for name, result in results.items():
            delta = changes.get(name, {})
            values = (name, result['lines'], *(100 * result[key] for key in ('cer', 'wer', 'exact_line_accuracy')),
                      *(result[key] for key in ('substitutions', 'insertions', 'deletions', 'eos_failures')),
                      delta.get('exact_lines_gained', '-'), delta.get('exact_lines_lost', '-'))
            print('{:25} {:5d} {:7.2f} {:7.2f} {:7.2f} {:5d} {:5d} {:5d} {:6d} {:>7} {:>7}'.format(*values))
            writer.writerow(values)
    return results, changes


def calibrate(ctx):
    rows, inputs = validation_inputs(ctx)
    output = ctx['output'] / 'validation'
    output.mkdir(exist_ok=True)
    configs = {}
    predictions = dict(inputs)
    for variant in DONORS:
        scores = score_corpus(ctx, rows, inputs, variant, output, 'validation', ctx['binding_sha256'])
        configs[variant] = fit(rows, inputs[BASE], scores)
        config = configs[variant]['config']
        print(f'{variant} selected {config if config is not None else "base fallback"}', flush=True)
        if config is None:
            candidate = configs[variant]['best_unconstrained']
            print(f'Best unconstrained validation CER = {100 * candidate["metrics"]["cer"]:.2f}% | '
                  f'WER = {100 * candidate["metrics"]["wer"]:.2f}% | '
                  f'Exact lines lost = {candidate["exact_lines_lost"]}', flush=True)
        predictions[variant] = apply_config(inputs[BASE], scores, config)
    results, changes = report_table(output, rows, predictions, DONORS, 'VALIDATION CALIBRATION | 150 lines')
    path = output / 'selection.json'
    write(path, dict(binding=ctx['binding'], binding_sha256=ctx['binding_sha256'], scope='children validation',
                     lines=150, variants=configs, results=results, changes_vs_base=changes))
    write(output / 'seal.json', dict(selection_sha256=file_hash(path)))
    print(f'Frozen validation selection = {path}', flush=True)


def load_selection(ctx):
    output = ctx['output'] / 'validation'
    path = output / 'selection.json'
    if not (output / 'seal.json').exists():
        raise ValueError('Run Calibrate before Test')
    seal = read(output / 'seal.json')
    verify(path, seal['selection_sha256'])
    selection = read(path)
    predecessor = dict(ctx['binding'], source_sha256={
        **ctx['binding']['source_sha256'], Path(__file__).name: PRE_SHORT_CACHE_RUNNER_SHA256,
    })
    if (selection['binding_sha256'] != digest(selection['binding'])
            or selection['binding'] not in (ctx['binding'], predecessor)
            or selection['scope'] != 'children validation' or selection['lines'] != 150
            or set(selection['variants']) != set(DONORS)):
        raise ValueError('Frozen validation selection does not match this comparison')
    for variant, result in selection['variants'].items():
        config = result['config']
        if config is not None and (config['donor_weight'] not in GRID['donor_weight'] or config['margin'] not in GRID['margin']
                                   or not result['selected']['eligible'] or config != result['selected']['config']):
            raise ValueError(f'Invalid frozen configuration for {variant}')
    return selection, seal['selection_sha256']


def historical_cache_root(output, identity):
    # Save space for the per-line filename and its atomic .json.tmp suffix.
    # The full identity is still checked in every cached record.
    return output / identity[:16]


def test(ctx):
    selection, selection_sha256 = load_selection(ctx)
    rows, inputs, source_hashes = historical_inputs(ctx)
    output = ctx['output'] / 'historical_752'
    output.mkdir(exist_ok=True)
    identity = digest(dict(binding_sha256=ctx['binding_sha256'], selection_sha256=selection_sha256, inputs=source_hashes))
    predictions = dict(inputs)
    for variant in DONORS:
        config = selection['variants'][variant]['config']
        scores = score_corpus(ctx, rows, inputs, variant, historical_cache_root(output, identity),
                              'historical', identity) if config is not None else {}
        print(f'{variant} uses {config if config is not None else "base fallback"}', flush=True)
        predictions[variant] = apply_config(inputs[BASE], scores, config)
    results, changes = report_table(output, rows, predictions, DONORS, 'HISTORICAL DEVELOPMENT COMPARISON | 752 lines')
    records = (dict(item, model=name) for name in DONORS for item in predictions[name].values())
    (output / 'predictions.jsonl').write_bytes(b''.join(canonical(item) for item in records))
    write(output / 'report.json', dict(scope='historical development-used comparison', lines=752,
                                      binding_sha256=ctx['binding_sha256'], selection_sha256=selection_sha256,
                                      validation_binding_sha256=selection['binding_sha256'],
                                      input_predictions_sha256=source_hashes, results=results, changes_vs_base=changes,
                                      configs={name: selection['variants'][name]['config'] for name in DONORS}))
    write(output / 'seal.json', {name: file_hash(output / name) for name in ('report.json', 'predictions.jsonl', 'comparison.csv')})
    print(f'Report file = {output / "report.json"}', flush=True)


def check(ctx):
    validation_inputs(ctx)
    device = ctx['args'].device
    pixels = torch.zeros((1, 3, ctx['config'].height, 32), device=device)
    widths = torch.tensor([32], device=device)
    base = dict(text='خط عربي', has_eos=True, status='completed')
    donor_prediction = dict(text='خط عملي', has_eos=True, status='completed')
    spans = propose(base, donor_prediction)
    if not spans:
        raise ValueError('Synthetic span proposal check found no candidates')
    for variant, donor in DONORS.items():
        models = load_models(ctx, donor)
        scored = score_images(pixels, widths, base, spans, models, ctx['tokenizer'], ctx['models'][donor]['arm'], device)
        select(base, scored, dict(donor_weight=.5, margin=.05))
        print(f'{variant}: checkpoint loading, span proposal and synthetic image scoring verified.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--action', choices=('Check', 'Calibrate', 'Test'), required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    seed_all(54101)
    ctx = context(args.project.resolve(), args.device)
    with ar.run_lock(ctx['output']):
        {'Check': check, 'Calibrate': calibrate, 'Test': test}[args.action](ctx)


if __name__ == '__main__':
    main()
