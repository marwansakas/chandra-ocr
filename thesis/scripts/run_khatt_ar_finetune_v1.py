"""Compare two decoder learning rates for KHATT-to-children AR fine-tuning."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from thesis.diffusion.images import ImageConfig
from thesis.diffusion.tokenizer import CharacterTokenizer
from thesis.khatt_ablation_v1 import workflow as ar
from thesis.khatt_ablation_v1.data import child_row
from thesis.mdiff_arabic_v1.artifacts import canonical, damage, digest, environment, file_hash, indexed, jsonl, metrics, read, write
from thesis.mdiff_arabic_v1.experiment import seed_all, stamp, save_torch, training_key
from thesis.mdiff_arabic_v1.model import Recognizer
from thesis.scripts.run_khatt_mdiff_v1 import epoch_line, require_hash, restore_rng, rng_state, train_epoch


VARIANTS = {'finetune_lr1e4': 1e-4, 'finetune_lr3e4': 3e-4}
BASELINES = ('khatt100_only', 'khatt100_ours', 'span_merge')
RECIPE = dict(ar.RECIPE, decoder_lr_by_trial=VARIANTS)
del RECIPE['decoder_lr']


def source_digest(repo, provenance):
    files = {Path(__file__).relative_to(repo).as_posix(): file_hash(Path(__file__))}
    shared = 'thesis/scripts/run_khatt_mdiff_v1.py'
    files[shared] = file_hash(repo / shared)
    for name, expected in provenance['source_sha256'].items():
        prefix, relative = name.split(':', 1)
        if prefix == 'repository' and not relative.startswith('tests/') and relative != 'thesis/scripts/run_khatt_ablation_v1.py':
            require_hash(repo / relative, expected)
            files[relative] = expected
    return files


def context(args):
    repo = Path(__file__).resolve().parents[2]
    provenance = read(repo / 'thesis/experiments/khatt_ablation_v1/ARTIFACTS.json')
    args.drive = args.project / 'Diffusion/drive_all'
    args.parent = args.drive / 'khatt_ablations' / provenance['suite']
    args.suite = args.drive / 'khatt_ar_finetune' / 'khatt_ar_finetune_v1_s54101'
    parent = provenance['selected']['khatt100_only']
    args.starting_checkpoint = args.project / parent['checkpoint']
    require_hash(args.starting_checkpoint, parent['checkpoint_sha256'])
    files = {}
    for name in ('children_train.jsonl', 'validation.jsonl', 'tokenizer.json'):
        path = args.parent / name
        expected = provenance['local_artifacts'][path.relative_to(args.project).as_posix()]['sha256']
        require_hash(path, expected)
        files[name] = expected
    runtime = environment(args.device)
    observed_packages = dict(runtime['packages'])
    runtime['packages'] = {name: observed_packages.get(name) for name in ('torch', 'numpy', 'pillow')}
    binding = dict(version='khatt_ar_finetune_v1', recipe=RECIPE, runtime=runtime, inputs=files,
                   starting_checkpoint_sha256=parent['checkpoint_sha256'], starting_epoch=parent['selected_epoch'],
                   parent_binding=provenance['original_binding_sha256'], image_config=provenance['image_config'],
                   source_sha256=source_digest(repo, provenance),
                   launcher_sha256=file_hash(args.project / 'Diffusion/run_khatt_ar_finetune_v1.ps1'))
    return dict(args=args, binding=binding, binding_sha256=digest(binding), observed_packages=observed_packages,
                provenance=provenance, tokenizer=CharacterTokenizer.load(args.parent / 'tokenizer.json'),
                config=ImageConfig(**provenance['image_config']))


def load_inputs(ctx):
    args = ctx['args']
    train_rows = jsonl(args.parent / 'children_train.jsonl')
    validation = jsonl(args.parent / 'validation.jsonl')
    train_ids = {row['sample_id'] for row in train_rows}
    val_ids = {row['sample_id'] for row in validation}
    if ((len(train_rows), len(train_ids), len(validation), len(val_ids)) != (2807, 2807, 150, 150)
            or train_ids & val_ids or any(row['source'] != 'children' for row in train_rows + validation)):
        raise ValueError('Children training or validation membership differs from the expected split')
    ar.verify_images(train_rows + validation)
    for row in train_rows + validation:
        ctx['tokenizer'].encode(row['text'], RECIPE['capacity'], eos_fill=False)
    path = args.suite / 'check.json'
    if path.exists() and read(path)['binding_sha256'] != ctx['binding_sha256']:
        raise ValueError('Fine-tune binding changed. Restore the original inputs, code and runtime before resuming.')
    write(path, dict(binding=ctx['binding'], binding_sha256=ctx['binding_sha256'],
                     observed_packages=ctx['observed_packages'], children_train=2807, children_validation=150))
    return train_rows, validation


def make_model(ctx, from_parent=False):
    model = Recognizer(len(ctx['tokenizer']), capacity=RECIPE['capacity'], arm='ar')
    if from_parent:
        saved = torch.load(ctx['args'].starting_checkpoint, map_location='cpu', weights_only=True)
        parent = ctx['binding']
        if (saved['trial'], saved['epoch'], saved['binding_sha256']) != ('khatt100_only', parent['starting_epoch'], parent['parent_binding']):
            raise ValueError('Starting checkpoint is not the selected KHATT-only Transformer')
        model.load_state_dict(saved['model'], strict=True)
    return model.to(ctx['args'].device)


def optimizer_for(model, trial):
    optimizer = ar.optimizer_for(model)
    decoder = optimizer.param_groups[1]
    decoder['lr'] = decoder['base_lr'] = VARIANTS[trial]
    return optimizer


def selected(ctx, trial):
    folder = ctx['args'].suite / trial
    complete_path = folder / 'complete.json'
    if not complete_path.exists():
        raise ValueError(f'Finish training {trial} before testing.')
    complete = read(complete_path)
    if complete['binding_sha256'] != ctx['binding_sha256']:
        raise ValueError(f'Completed training binding mismatch: {trial}')
    require_hash(folder / 'last.json', complete['last_sha256'])
    best = complete['best']
    require_hash(folder / best['checkpoint'], best['sha256'])
    require_hash(folder / best['predictions'], best['predictions_sha256'])
    return folder, best


def train(ctx, trial, train_rows, validation):
    args = ctx['args']
    folder = args.suite / trial
    if (folder / 'complete.json').exists():
        _, best = selected(ctx, trial)
        print(f'{trial} is complete. Selected epoch = {best["epoch"]}.', flush=True)
        return
    last_path = folder / 'last.json'
    pending_path = folder / 'pending_validation.json'
    if (last_path.exists() or pending_path.exists()) and not args.resume:
        raise ValueError(f'Saved progress for {trial}. Add -Resume to continue.')
    last = read(last_path) if last_path.exists() else dict(epoch=0, stale=0, best=None)
    pending = read(pending_path) if pending_path.exists() else None
    if pending and pending['epoch'] == last['epoch'] and pending['path'] == last['path']:
        pending_path.unlink()
        pending = None
    if pending and pending['epoch'] != last['epoch'] + 1:
        raise ValueError('Pending validation is not for the next epoch')

    seed_all(RECIPE['seed'])
    resume_point = pending or (last if last['epoch'] else None)
    model = make_model(ctx, from_parent=not resume_point)
    optimizer = optimizer_for(model, trial)
    updates = 0
    if resume_point:
        path = folder / resume_point['path']
        require_hash(path, resume_point['sha256'])
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['trial'], saved['epoch'], saved['binding_sha256']) != (trial, resume_point['epoch'], ctx['binding_sha256']):
            raise ValueError('Resume checkpoint identity mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved, args.device)
        updates = saved['updates']
        del saved
    print(f'{trial} | Decoder LR = {VARIANTS[trial]:g} | Encoder LR = {RECIPE["encoder_lr"]:g}', flush=True)
    if pending:
        print(f'Resuming validation for epoch {pending["epoch"]}.', flush=True)
    elif resume_point:
        print(f'Resuming after completed epoch {last["epoch"]}.', flush=True)

    for epoch in ar.remaining_epochs(last['epoch'], last['stale']):
        if pending is None:
            started = time.perf_counter()
            updates, loss = train_epoch(ctx, model, optimizer, train_rows, epoch, updates)
            train_seconds = time.perf_counter() - started
            attempt = folder / f'e{epoch:03d}_{stamp()}'
            state_path = attempt / 'state.pt'
            save_torch(state_path, dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                       trial=trial, binding_sha256=ctx['binding_sha256'], updates=updates,
                                       **rng_state(args.device)))
            pending = dict(epoch=epoch, path=state_path.relative_to(folder).as_posix(), sha256=file_hash(state_path),
                           train_seconds=train_seconds, loss=loss)
            write(pending_path, pending)
        else:
            attempt = (folder / pending['path']).parent

        started = time.perf_counter()
        model.eval()
        predictions = [ar.predict_one(model, row, ctx) for row in validation]
        scores = metrics(validation, indexed(predictions, validation))
        elapsed = pending['train_seconds'] + time.perf_counter() - started
        best = last['best']
        improved = best is None or training_key(scores, epoch) < training_key(best['validation_metrics'], best['epoch'])
        stale = 0 if improved else last['stale'] + 1
        prediction_path = attempt / 'validation.jsonl'
        prediction_path.write_bytes(b''.join(canonical(item) for item in predictions))
        if improved:
            model_path = attempt / 'model.pt'
            save_torch(model_path, dict(epoch=epoch, model=model.state_dict(), trial=trial, binding_sha256=ctx['binding_sha256']))
            best = dict(epoch=epoch, checkpoint=model_path.relative_to(folder).as_posix(), sha256=file_hash(model_path),
                        predictions=prediction_path.relative_to(folder).as_posix(), predictions_sha256=file_hash(prediction_path),
                        validation_metrics=scores)
        write(attempt / 'history.json', dict(epoch=epoch, validation=scores, loss=pending['loss'], updates=updates,
                                             improved=improved, stale=stale, epoch_seconds=elapsed))
        last = dict(pending, best=best, stale=stale)
        write(last_path, last)
        pending_path.unlink()
        pending = None
        ar.prune_states(folder, [last['path'], best['checkpoint']])
        print(epoch_line(epoch, scores, elapsed), flush=True)
        if stale >= RECIPE['patience']:
            break

    write(folder / 'complete.json', dict(epoch=last['epoch'], best=last['best'], binding_sha256=ctx['binding_sha256'],
                                        last_sha256=file_hash(last_path), test_used=False,
                                        stop_reason='patience' if last['stale'] >= RECIPE['patience'] else 'maximum_epochs'))
    print(f'{trial} complete. Selected epoch = {last["best"]["epoch"]}.', flush=True)


def score_trial(ctx, rows, trial, folder, best, output):
    cache = output / trial
    cache.mkdir(exist_ok=True)
    model = None
    predictions = []
    for index, row in enumerate(rows, 1):
        path = cache / (digest(row['sample_id']) + '.json')
        if path.exists():
            prediction = read(path)
            if prediction['sample_id'] != row['sample_id'] or prediction['image_sha256'] != row['image_sha256']:
                raise ValueError('Historical prediction cache identity mismatch')
        else:
            if model is None:
                saved = torch.load(folder / best['checkpoint'], map_location='cpu', weights_only=True)
                if (saved['trial'], saved['epoch'], saved['binding_sha256']) != (trial, best['epoch'], ctx['binding_sha256']):
                    raise ValueError('Selected fine-tune checkpoint identity mismatch')
                model = make_model(ctx).eval()
                model.load_state_dict(saved['model'], strict=True)
                del saved
            prediction = ar.predict_one(model, row, ctx)
            write(path, prediction)
        predictions.append(dict(prediction, model=trial))
        if index % 100 == 0:
            print(f'{trial} historical test lines = {index}/{len(rows)}', flush=True)
    return predictions


def test(ctx):
    args, provenance = ctx['args'], ctx['provenance']
    selections = {trial: selected(ctx, trial) for trial in VARIANTS}
    test_manifest = args.drive / 'splits_70_05_25_v3/test.jsonl'
    require_hash(test_manifest, provenance['inputs'][test_manifest.relative_to(args.project).as_posix()])
    test_ids, baseline_path = args.parent / 'test_identity.jsonl', args.parent / 'historical_752/predictions.jsonl'
    for path in (test_ids, baseline_path):
        require_hash(path, provenance['local_artifacts'][path.relative_to(args.project).as_posix()]['sha256'])
    rows = [child_row(row, args.drive / 'ground_truth_lines') for row in jsonl(test_manifest)]
    if len(rows) != 752 or [row['sample_id'] for row in rows] != [row['sample_id'] for row in jsonl(test_ids)]:
        raise ValueError('Historical test membership mismatch')
    frozen = jsonl(baseline_path)
    by_model = {name: indexed([row for row in frozen if row['model'] == name], rows) for name in BASELINES}
    output = args.suite / 'historical_752'
    output.mkdir(exist_ok=True)
    identity = dict(binding_sha256=ctx['binding_sha256'], baseline_sha256=file_hash(baseline_path),
                    test_manifest_sha256=file_hash(test_manifest),
                    checkpoints={name: best['sha256'] for name, (_, best) in selections.items()})
    path = output / 'identity.json'
    if path.exists() and read(path) != identity:
        raise ValueError('Historical comparison cache belongs to a different run')
    write(path, identity)
    if (output / 'seal.json').exists():
        return stats(ctx)

    seed_all(RECIPE['seed'])
    new_predictions = []
    for trial, (folder, best) in selections.items():
        predictions = score_trial(ctx, rows, trial, folder, best, output)
        by_model[trial] = indexed(predictions, rows)
        new_predictions.extend(predictions)
    report = dict(identity=identity, scope='historical development-used comparison', lines=752,
                  selected_epochs={name: best['epoch'] for name, (_, best) in selections.items()},
                  validation_metrics={name: best['validation_metrics'] for name, (_, best) in selections.items()},
                  results={name: metrics(rows, predictions) for name, predictions in by_model.items()},
                  vs_mixed_ar={name: damage(rows, by_model['khatt100_ours'], by_model[name]) for name in VARIANTS})
    (output / 'predictions.jsonl').write_bytes(b''.join(canonical(row) for row in new_predictions))
    write(output / 'report.json', report)
    write(output / 'seal.json', {name: file_hash(output / name) for name in ('predictions.jsonl', 'report.json')})
    stats(ctx)


def stats(ctx):
    output = ctx['args'].suite / 'historical_752'
    seal = read(output / 'seal.json')
    for name in ('report.json', 'predictions.jsonl'):
        require_hash(output / name, seal[name])
    report = read(output / 'report.json')
    if report['identity']['binding_sha256'] != ctx['binding_sha256']:
        raise ValueError('Historical comparison binding mismatch')
    print('HISTORICAL DEVELOPMENT COMPARISON')
    print(f'{"Model":22} {"Epoch":>5} {"Lines":>5} {"CER %":>7} {"WER %":>7} {"Exact %":>8} {"Sub":>5} {"Ins":>5} {"Del":>5} {"No EOS":>6}')
    with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Model', 'Selected epoch', 'Lines', 'CER %', 'WER %', 'Exact %', 'Sub', 'Ins', 'Del', 'No EOS'])
        epochs = dict(khatt100_only=34, khatt100_ours=40, **report['selected_epochs'])
        for name in (*BASELINES, *VARIANTS):
            scores = report['results'][name]
            row = [name, str(epochs.get(name, '-')), scores['lines'],
                   *(100 * scores[key] for key in ('cer', 'wer', 'exact_line_accuracy')),
                   *(scores[key] for key in ('substitutions', 'insertions', 'deletions', 'eos_failures'))]
            print('{:22} {:>5} {:5d} {:7.2f} {:7.2f} {:8.2f} {:5d} {:5d} {:5d} {:6d}'.format(*row))
            writer.writerow(row)
    print(f'Report file = {output / "report.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', required=True, choices=('Check', 'Train', 'TrainAll', 'Test', 'Stats'))
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--trial', choices=tuple(VARIANTS))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if bool(args.trial) != (args.action == 'Train'):
        parser.error('A trial is required only with Train. Use TrainAll for both.')
    if args.resume and args.action not in ('Train', 'TrainAll'):
        parser.error('Resume is supported only with Train or TrainAll')
    args.project = args.project.resolve()
    ctx = context(args)
    with ar.run_lock(args.suite):
        if args.action in ('Check', 'Train', 'TrainAll'):
            train_rows, validation = load_inputs(ctx)
            if args.action == 'Check':
                model = make_model(ctx, from_parent=True)
                groups = [optimizer_for(model, trial).param_groups for trial in VARIANTS]
                print(f'Verified epoch-34 Transformer, {len(train_rows)} children training lines, '
                      f'{len(validation)} children validation lines, and optimizer rates '
                      f'{[(g[0]["base_lr"], g[1]["base_lr"]) for g in groups]}.', flush=True)
            else:
                trials = (args.trial,) if args.trial else VARIANTS
                for trial in trials:
                    train(ctx, trial, train_rows, validation)
        elif args.action == 'Test':
            test(ctx)
        else:
            stats(ctx)


if __name__ == '__main__':
    main()
