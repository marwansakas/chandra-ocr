"""Explicitly authorized historical comparison without altering the frozen gate.

This standalone evaluator is deliberately outside the bound training sources.
It retains all source/data/environment/checkpoint checks. Only the requirement
that validation eligibility be true is waived, with recorded human authorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / '.codex-pretrained-selection'))

import torch
from PIL import Image

from thesis.scripts.run_mdiff_arabic_v1 import parse_args as primary_args
from thesis.mdiff_arabic_v1 import experiment as e
from thesis.mdiff_arabic_v1.artifacts import (
    POLICY, canonical, digest, file_hash, indexed, jsonl, load_data, prepare_images,
    read, require_equal, resolve_relative, verify_comparators, write, write_jsonl,
)
from thesis.mdiff_arabic_v1.model import tensor_hash


def verify_validation(ctx, peer, own, other):
    path = ctx['run'] / 'validation/report.json'
    report = read(path)
    seal = read(ctx['run'] / 'validation/gate.json')
    expected = dict(binding_sha256=ctx['binding_sha256'],
        validation_report_sha256=file_hash(path), selected_checkpoint_sha256=own['sha256'],
        peer_selected_checkpoint_sha256=other['sha256'], policy_sha256=digest(POLICY),
        predictions_sha256=file_hash(ctx['run'] / 'validation/predictions.jsonl'))
    for key, value in expected.items():
        require_equal(seal.get(key), value, 'exploratory validation seal: ' + key)
    require_equal(report['binding_sha256'], ctx['binding_sha256'], 'validation binding')
    require_equal(report['peer_binding_sha256'], peer['binding_sha256'], 'peer validation binding')
    if report['partition'] != 'validation' or report['test_used'] is not False:
        raise ValueError('Expected a frozen validation-only report')
    actual_conditions = {}
    for c, best in ((ctx, own), (peer, other)):
        predictions = indexed(jsonl(c['run'] / best['predictions']), ctx['parts']['validation'])
        actual = e.eligibility(ctx['parts']['validation'], ctx['baselines'], predictions)
        name = e.model_name(c['args'].arm)
        require_equal(actual['conditions'], report['arms'][name]['conditions'], 'validation conditions')
        require_equal(actual['raw_metrics'], best['validation_metrics'], 'selected raw validation metrics')
        actual_conditions.update({name + '.' + k: v for k, v in actual['conditions'].items()})
    require_equal(actual_conditions, report['gate_conditions'], 'joint validation conditions')
    require_equal(all(actual_conditions.values()), report['validation_passes_release_gate'], 'validation verdict')
    require_equal(seal['validation_passes_release_gate'], report['validation_passes_release_gate'], 'sealed verdict')
    # Deliberately preserve false. Human authorization concerns evaluation only.
    return report, seal


def predict_images(model, rows, ctx, path):
    """Generate from images only; never encode test references into input targets."""
    model.eval()
    device = ctx['args'].device
    output = []
    with path.open('xb') as handle:
        for number, row in enumerate(rows, 1):
            started = time.perf_counter()
            try:
                source = resolve_relative(ctx['args'].drive / 'ground_truth_lines', row['image_path'])
                with Image.open(source) as image:
                    pixels, widths = prepare_images([image.convert('RGB')], ctx['image_config'])
                pixels, widths = pixels.to(device), widths.to(device)
                with e.autocast(device):
                    ids, info = model.generate(pixels, widths, steps=3)
                text, has_eos = ctx['tokenizer'].decode(ids[0].tolist())
                e.synchronize(device)
                prediction = dict(sample_id=row['sample_id'], text=text, has_eos=has_eos,
                    status='completed' if has_eos else 'failed', length_failure=not has_eos,
                    seconds=time.perf_counter()-started, decoder_passes=info['decoder_passes'],
                    sampler_mode=info['sampler_mode'], sampler_steps=info['sampler_steps'], error=None)
            except Exception as error:
                prediction = dict(sample_id=row['sample_id'], text='', has_eos=False, status='failed',
                    length_failure=False, seconds=time.perf_counter()-started,
                    error=f'{type(error).__name__}: {error}')
                if device == 'cuda': torch.cuda.empty_cache()
            output.append(prediction)
            handle.write(canonical(prediction))
            if number % 50 == 0 or number == len(rows):
                handle.flush()
                failures = sum(p['status'] != 'completed' for p in output)
                print(f'{e.model_name(model.arm)}: {number}/{len(rows)} | failed={failures}', flush=True)
    return indexed(output, rows)


def protect(ctx, peer, own, other):
    paths = set(ctx['binding']['source_sha256'])
    paths.update(str(ctx['args'].drive / 'splits_70_05_25_v3' / name)
                 for name in ('train.jsonl', 'validation.jsonl', 'test.jsonl', 'split_lock.json'))
    for c, best in ((ctx, own), (peer, other)):
        paths.update(str(c['run'] / name) for name in ('run.json', 'best.json', 'training_complete.json',
            'tokenizer.json', 'dependencies.freeze.txt', best['checkpoint'], best['predictions']))
    paths.update(str(ctx['run'] / 'validation' / name) for name in ('report.json', 'gate.json', 'predictions.jsonl'))
    return {str(p): file_hash(Path(p)) for p in sorted(paths)}


def run(args):
    base_args = primary_args(['--project', str(PROJECT), '--action', 'Test', '--run-name', 'ma1b_s54101',
        '--peer-run-name', 'ar1b_s54101', '--arm', 'mdiff', '--device', args.device,
        '--readiness-protocol', 'geometry_matched_v1'])
    print('Verifying frozen training, data, environment, checkpoint selection and validation bindings...', flush=True)
    ctx = e.context(base_args)
    own = e.selected(ctx)
    peer = e.peer_context(ctx)
    other = e.selected(peer)
    validation, seal = verify_validation(ctx, peer, own, other)
    output = base_args.drive / 'exploratory' / args.evaluation_name
    output.mkdir(parents=True, exist_ok=False)
    before = protect(ctx, peer, own, other)
    request = dict(evaluation_mode='user_authorized_exploratory_override',
        explicit_authorization=args.authorization_text,
        created_utc=e.stamp(), partition='test', expected_lines=752,
        validation_passes_release_gate=validation['validation_passes_release_gate'],
        validation_gate_overridden_for_evaluation_only=True, promotion_performed=False,
        validation_report_sha256=file_hash(ctx['run'] / 'validation/report.json'),
        gate_sha256=file_hash(ctx['run'] / 'validation/gate.json'),
        binding_sha256=ctx['binding_sha256'], peer_binding_sha256=peer['binding_sha256'],
        checkpoints={e.model_name(c['args'].arm): best for c, best in ((ctx, own), (peer, other))},
        evaluator_sha256=file_hash(Path(__file__)), protected_files=before,
        limitation='historical development-used benchmark; no untouched generalization claim; no promotion')
    write(output / 'request.json', request, exclusive=True)
    shutil.copyfile(__file__, output / 'evaluator.py')
    write(output / 'frozen_validation_report.json', validation, exclusive=True)
    write(output / 'frozen_gate.json', seal, exclusive=True)
    print('Exploratory output:', output, flush=True)
    parts, _, _, evidence = load_data(base_args.drive, test_authorized=True,
                                     readiness_protocol=base_args.readiness_protocol)
    rows = parts['test']
    if len(rows) != 752: raise ValueError('Canonical historical test must contain exactly 752 lines')
    issues = [r for r in evidence['coverage_issues'] if r['partition'] == 'test']
    write(output / 'coverage_issues.json', issues, exclusive=True)
    # Unsupported reference symbols are reported and scored literally; no test
    # vocabulary expansion. Image-limit failures remain in the full denominator.
    baselines, provenance = verify_comparators(base_args.drive, rows, 'test')
    write(output / 'comparator_provenance.json', provenance, exclusive=True)
    for name, predictions in baselines.items():
        write_jsonl(output / (name + '.jsonl'), predictions.values())
    predictions = {}
    memories = {}
    for c, best in ((ctx, own), (peer, other)):
        name = e.model_name(c['args'].arm)
        print(f'Loading {name}, selected epoch {best["epoch"]}...', flush=True)
        model, _ = e.make_model(c)
        require_equal(tensor_hash(model.state_dict()), read(c['checks'] / 'check.json')['model_initialization_sha256'], 'model initialization')
        state = torch.load(c['run'] / best['checkpoint'], map_location='cpu', weights_only=True)
        require_equal(state['binding_sha256'], c['binding_sha256'], 'checkpoint binding')
        require_equal(state['epoch'], best['epoch'], 'checkpoint epoch')
        model.load_state_dict(state['model'], strict=True)
        if args.device == 'cuda': torch.cuda.reset_peak_memory_stats()
        predictions[name] = predict_images(model, rows, c, output / (name + '.jsonl'))
        memories[name] = torch.cuda.max_memory_allocated() if args.device == 'cuda' else None
        del model, state
        if args.device == 'cuda': torch.cuda.empty_cache()
    report = e.report_for(ctx, peer, own, other, rows, baselines, predictions, 'test', provenance)
    order = ['paddle', 'frozen_v2_2', 'span_hybrid_verifier_v4', 'transformer_arabic_control', 'mdiff_arabic_raw']
    report['models'] = {name: report['models'][name] for name in order}
    for arm in report['arms'].values():
        arm['historical_comparison_conditions_descriptive_only'] = arm.pop('conditions')
        arm.pop('validation_passes_release_gate')
    report.update(evaluation_mode=request['evaluation_mode'],
        explicit_authorization=args.authorization_text,
        validation_gate_overridden_for_evaluation_only=True,
        validation_passes_release_gate=validation['validation_passes_release_gate'],
        gate_conditions=validation['gate_conditions'], promotion_performed=False,
        coverage_issues=issues, peak_gpu_memory_bytes_by_model=memories,
        evaluator_sha256=request['evaluator_sha256'], request_sha256=file_hash(output / 'request.json'),
        frozen_validation_report_sha256=request['validation_report_sha256'], frozen_gate_sha256=request['gate_sha256'])
    write_jsonl(output / 'predictions.jsonl', e.joined(rows, {**baselines, **predictions}))
    report['predictions_sha256'] = file_hash(output / 'predictions.jsonl')
    after = {p: file_hash(Path(p)) for p in before}
    require_equal(before, after, 'protected original artifacts after evaluation')
    report['original_artifacts_unchanged'] = True
    report['protected_files'] = after
    report_path = output / 'report.json'
    write(report_path, report, exclusive=True)
    text = 'USER-AUTHORIZED EXPLORATORY HISTORICAL TEST (752 lines)\n'
    text += 'Frozen validation gate remains ' + str(validation['validation_passes_release_gate']) + '. No promotion.\n\n'
    text += 'Model                        Lines   CER %   WER %  Exact %   Sub   Ins   Del\n'
    for name, m in report['models'].items():
        text += f'{name:<28} {m["lines"]:5d} {100*m["cer"]:7.2f} {100*m["wer"]:7.2f} {100*m["exact_line_accuracy"]:8.2f} {m["substitutions"]:5d} {m["insertions"]:5d} {m["deletions"]:5d}\n'
    for name, arm in report['arms'].items():
        d = arm['damage']['span_hybrid_verifier_v4']
        text += f'{name}: exact lost/gained vs v4={d["exact_lines_lost"]}/{d["exact_lines_gained"]}; failed={arm["raw_metrics"]["failed_lines"]}\n'
    text += f'Original artifacts unchanged: true | Test used: true\nReport: {report_path}\n'
    (output / 'report.txt').write_text(text, encoding='utf-8')
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-name', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--authorize-failed-gate', action='store_true')
    parser.add_argument('--authorization-text', required=True)
    args = parser.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,39}', args.evaluation_name):
        parser.error('Use a new short safe evaluation name')
    if not args.authorize_failed_gate or not args.authorization_text.strip():
        parser.error('Explicit user authorization for exploratory evaluation is required')
    run(args)


if __name__ == '__main__':
    main()
