"""Train and compare one MDiff model on the winning KHATT + children mixture."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from thesis.diffusion.images import ImageConfig
from thesis.diffusion.tokenizer import CharacterTokenizer
from thesis.khatt_ablation_v1 import workflow as ar
from thesis.khatt_ablation_v1.data import child_row
from thesis.mdiff_arabic_v1.artifacts import (
    canonical, damage, digest, environment, file_hash, indexed, jsonl, metrics, read, write,
)
from thesis.mdiff_arabic_v1.experiment import autocast, save_torch, seed_all, stamp, synchronize, training_key
from thesis.mdiff_arabic_v1.model import Recognizer, load_encoder, tensor_hash

TRIAL = 'mdiff_khatt100_ours'
AR_TRIAL = 'khatt100_ours'
BASELINES = (AR_TRIAL, 'mdiff_arabic_raw', 'span_merge')
RECIPE = dict(ar.RECIPE, arm='mdiff', sampler_steps=3, denoise_weight=.5, correction_weight=.5,
              tail_eos_weight=.1, training_decoder_passes=2)
SOURCE_NAMES = ('khatt100_ours.jsonl', 'validation.jsonl', 'tokenizer.json', 'prepared.json')


def require_hash(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f'File hash mismatch: {path}')


def context(args):
    repo = Path(__file__).resolve().parents[2]
    provenance_file = repo / 'thesis/experiments/khatt_ablation_v1/ARTIFACTS.json'
    provenance = read(provenance_file)
    args.drive = args.project / 'Diffusion/drive_all'
    args.parent = args.drive / 'khatt_ablations' / provenance['suite']
    args.suite = args.drive / 'khatt_ablation_mdiff' / 'khatt_mdiff_v1_s54101'
    args.assets = args.drive / 'pretrained/mdiff_s_v1'
    for name in SOURCE_NAMES:
        path = args.parent / name
        require_hash(path, provenance['local_artifacts'][path.relative_to(args.project).as_posix()]['sha256'])
    for name, expected in provenance['pretrained_sha256'].items():
        require_hash(args.assets / name, expected)
    for name, expected in provenance['source_sha256'].items():
        prefix, relative = name.split(':', 1)
        if prefix == 'repository':
            require_hash(repo / relative, expected)
    observed = environment(args.device)
    # Bind dependencies used by this pipeline. Record the other installed
    # versions for reference without making them a condition of resume.
    runtime = dict(observed)
    runtime['packages'] = {name: observed['packages'].get(name) for name in ('torch', 'numpy', 'pillow')}
    launcher = args.project / 'Diffusion/run_khatt_mdiff_v1.ps1'
    binding = dict(version='khatt_mdiff_v1', recipe=RECIPE, runtime=runtime,
                   ar_provenance_sha256=file_hash(provenance_file),
                   runner_sha256=file_hash(Path(__file__)), launcher_sha256=file_hash(launcher))
    tokenizer = CharacterTokenizer.load(args.parent / 'tokenizer.json')
    return dict(args=args, binding=binding, binding_sha256=digest(binding), observed=observed,
                provenance=provenance, tokenizer=tokenizer, config=ImageConfig(**provenance['image_config']))


def check(ctx):
    args = ctx['args']
    train_rows = jsonl(args.parent / 'khatt100_ours.jsonl')
    validation = jsonl(args.parent / 'validation.jsonl')
    counts = {source: sum(row['source'] == source for row in train_rows) for source in ('khatt', 'children')}
    train_ids = {row['sample_id'] for row in train_rows}
    val_ids = {row['sample_id'] for row in validation}
    if (len(train_rows) != 12304 or counts != {'khatt': 9497, 'children': 2807}
            or len(train_ids) != 12304 or len(validation) != 150 or len(val_ids) != 150
            or train_ids & val_ids or any(row['source'] != 'children' for row in validation)):
        raise ValueError('Unexpected training or children validation membership')
    ar.verify_images(train_rows + validation)
    for row in train_rows + validation:
        ctx['tokenizer'].encode(row['text'], RECIPE['capacity'], eos_fill=False)
    check_file = args.suite / 'check.json'
    if check_file.exists() and read(check_file)['binding_sha256'] != ctx['binding_sha256']:
        raise ValueError('MDiff run binding changed. Keep the original code and training environment for resume.')
    write(check_file, dict(binding=ctx['binding'], binding_sha256=ctx['binding_sha256'],
                           observed_environment=ctx['observed'], train_counts=counts, validation_lines=len(validation)))
    return train_rows, validation


def make_model(ctx, initialize=False):
    model = Recognizer(len(ctx['tokenizer']), capacity=RECIPE['capacity'], arm='mdiff')
    if initialize:
        reference = ctx['provenance']['selected'][AR_TRIAL]
        transfer = load_encoder(model.encoder, ctx['args'].assets / 'best.pth')
        if (tensor_hash(model.state_dict()) != reference['model_initialization_sha256']
                or transfer['encoder_initialization_sha256'] != reference['encoder_initialization_sha256']):
            raise ValueError('Initial MDiff tensors differ from the matched AR run')
    return model.to(ctx['args'].device)


def epoch_line(epoch, scores, seconds):
    minutes, sec = divmod(round(seconds), 60)
    hours, minute = divmod(minutes, 60)
    return (f'Epoch = {epoch} | CER = {100 * scores["cer"]:.2f}% | WER = {100 * scores["wer"]:.2f}% | '
            f'Exact  = {100 * scores["exact_line_accuracy"]:.2f}% | Time for epoch = {hours:02d}:{minute:02d}:{sec:02d}')


def train_epoch(ctx, model, optimizer, rows, epoch, updates):
    device = ctx['args'].device
    accumulation = RECIPE['accumulation']
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(epoch > RECIPE['encoder_frozen_epochs'])
    model.train()
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(RECIPE['seed'] + epoch)).tolist()
    per_epoch = math.ceil(len(rows) / accumulation)
    loss_sum = 0.0
    last_progress = time.perf_counter()
    for offset in range(0, len(order), accumulation):
        indices = order[offset:offset + accumulation]
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * ar.lr_factor(updates, per_epoch)
        for index in indices:
            row = rows[index]
            pixels, widths = ar.image_batch(row, ctx, epoch)
            tokens = ctx['tokenizer'].encode(row['text'], RECIPE['capacity'], eos_fill=False)
            target = torch.tensor([tokens], device=device)
            with autocast(device):
                loss, _ = model.loss(pixels, widths, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite MDiff training loss')
            (loss / len(indices)).backward()
            loss_sum += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        now = time.perf_counter()
        if now - last_progress >= 60:
            print(f'Training epoch {epoch}: {min(offset + accumulation, len(rows))}/{len(rows)} lines', flush=True)
            last_progress = now
    synchronize(device)
    return updates, loss_sum / len(rows)


def rng_state(device):
    numpy_state = np.random.get_state()
    return dict(python_rng=random.getstate(), numpy_rng=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else [])


def restore_rng(state, device):
    random.setstate(state['python_rng'])
    kind, keys, position, gaussian, cached = state['numpy_rng']
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state['torch_rng'])
    if device == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def synthetic_smoke(ctx):
    seed_all(RECIPE['seed'])
    model = make_model(ctx, initialize=True)
    device = ctx['args'].device
    width = ctx['config'].max_width
    pixels = torch.zeros((1, 3, ctx['config'].height, width), device=device)
    widths = torch.tensor([width], device=device)
    target = ctx['tokenizer'].encode('خط عربي', RECIPE['capacity'], eos_fill=False)
    model.train()
    with autocast(device):
        loss, _ = model.loss(pixels, widths, torch.tensor([target], device=device))
    if not torch.isfinite(loss) or model.decoder_passes != RECIPE['training_decoder_passes']:
        raise ValueError('Synthetic diffusion loss or decoder pass check failed')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE['gradient_clip'], error_if_nonfinite=True)
    model.eval()
    with autocast(device):
        ids, metadata = model.generate(pixels, widths, steps=RECIPE['sampler_steps'])
    if ids.shape != (1, RECIPE['capacity']) or metadata['decoder_passes'] != 2 * RECIPE['sampler_steps']:
        raise ValueError('Synthetic diffusion output shape or decoder pass check failed')
    ctx['tokenizer'].decode(ids[0].cpu().tolist())


def train(ctx, rows, validation):
    args = ctx['args']
    folder = args.suite
    complete_file = folder / 'complete.json'
    if complete_file.exists():
        best = selected(ctx)
        print(f'Training complete. Selected epoch = {best["epoch"]}.', flush=True)
        return
    last_file, pending_file = folder / 'last.json', folder / 'pending_validation.json'
    if (last_file.exists() or pending_file.exists()) and not args.resume:
        raise ValueError('Saved progress found. Add -Resume to continue.')
    last = read(last_file) if last_file.exists() else dict(epoch=0, stale=0, best=None)
    pending = read(pending_file) if pending_file.exists() else None
    if pending and pending['epoch'] == last['epoch'] and pending['path'] == last['path']:
        # The last epoch was committed before an interruption cleared the pending pointer.
        pending_file.unlink()
        pending = None
    if pending and pending['epoch'] != last['epoch'] + 1:
        raise ValueError('Pending and committed epoch pointers disagree')

    seed_all(RECIPE['seed'])
    model = make_model(ctx, initialize=True)
    optimizer = ar.optimizer_for(model)
    updates = 0
    resume_point = pending or (last if last['epoch'] else None)
    if resume_point:
        path = folder / resume_point['path']
        require_hash(path, resume_point['sha256'])
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['binding_sha256'] != ctx['binding_sha256'] or state['epoch'] != resume_point['epoch']:
            raise ValueError('Resume checkpoint identity mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state, args.device)
        updates = state['updates']
        del state
        if pending:
            print(f'Resuming validation for epoch {pending["epoch"]}.', flush=True)
        else:
            print(f'Resuming after completed epoch {last["epoch"]}.', flush=True)

    for epoch in ar.remaining_epochs(last['epoch'], last['stale']):
        if pending is None:
            start = time.perf_counter()
            updates, loss = train_epoch(ctx, model, optimizer, rows, epoch, updates)
            train_seconds = time.perf_counter() - start
            attempt = folder / f'e{epoch:03d}_{stamp()}'
            state_path = attempt / 'state.pt'
            save_torch(state_path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                       updates=updates, binding_sha256=ctx['binding_sha256'], **rng_state(args.device)))
            pending = dict(epoch=epoch, path=state_path.relative_to(folder).as_posix(), sha256=file_hash(state_path),
                           train_seconds=train_seconds, loss=loss)
            write(pending_file, pending)
        else:
            attempt = (folder / pending['path']).parent

        validation_start = time.perf_counter()
        model.eval()
        predictions = [ar.predict_one(model, row, ctx) for row in validation]
        scores = metrics(validation, indexed(predictions, validation))
        elapsed = pending['train_seconds'] + time.perf_counter() - validation_start
        best = last['best']
        improved = best is None or training_key(scores, epoch) < training_key(best['validation_metrics'], best['epoch'])
        stale = 0 if improved else last['stale'] + 1
        prediction_path = attempt / 'validation.jsonl'
        prediction_path.write_bytes(b''.join(canonical(item) for item in predictions))
        if improved:
            model_path = attempt / 'model.pt'
            save_torch(model_path, dict(model=model.state_dict(), epoch=epoch, binding_sha256=ctx['binding_sha256'], trial=TRIAL))
            best = dict(epoch=epoch, checkpoint=model_path.relative_to(folder).as_posix(), sha256=file_hash(model_path),
                        predictions=prediction_path.relative_to(folder).as_posix(),
                        predictions_sha256=file_hash(prediction_path), validation_metrics=scores)
        write(attempt / 'history.json', dict(epoch=epoch, validation=scores, loss=pending['loss'], updates=updates,
                                             improved=improved, stale=stale, epoch_seconds=elapsed, test_used=False))
        last = dict(pending, best=best, stale=stale)
        write(last_file, last)
        pending_file.unlink()
        pending = None
        ar.prune_states(folder, [last['path'], best['checkpoint']])
        print(epoch_line(epoch, scores, elapsed), flush=True)
        if stale >= RECIPE['patience']:
            break

    write(complete_file, dict(epoch=last['epoch'], best=last['best'], binding_sha256=ctx['binding_sha256'],
                              last_sha256=file_hash(last_file), test_used=False,
                              stop_reason='patience' if last['stale'] >= RECIPE['patience'] else 'maximum_epochs'))
    print(f'Training complete. Selected epoch = {last["best"]["epoch"]}.', flush=True)


def selected(ctx):
    folder = ctx['args'].suite
    if not (folder / 'complete.json').exists():
        raise ValueError('Finish Train before testing the selected MDiff checkpoint.')
    complete = read(folder / 'complete.json')
    if complete['binding_sha256'] != ctx['binding_sha256']:
        raise ValueError('Completed MDiff run binding mismatch')
    require_hash(folder / 'last.json', complete['last_sha256'])
    best = complete['best']
    require_hash(folder / best['checkpoint'], best['sha256'])
    require_hash(folder / best['predictions'], best['predictions_sha256'])
    return best


def test(ctx):
    args, provenance = ctx['args'], ctx['provenance']
    best = selected(ctx)
    test_path = args.drive / 'splits_70_05_25_v3/test.jsonl'
    require_hash(test_path, provenance['inputs'][test_path.relative_to(args.project).as_posix()])
    identity = args.parent / 'test_identity.jsonl'
    baseline_path = args.parent / 'historical_752/predictions.jsonl'
    for path in (identity, baseline_path):
        require_hash(path, provenance['local_artifacts'][path.relative_to(args.project).as_posix()]['sha256'])
    rows = [child_row(row, args.drive / 'ground_truth_lines') for row in jsonl(test_path)]
    if len(rows) != 752 or [row['sample_id'] for row in rows] != [row['sample_id'] for row in jsonl(identity)]:
        raise ValueError('Test membership differs from the saved 752-line comparison')
    frozen = jsonl(baseline_path)
    predictions = {name: indexed([row for row in frozen if row['model'] == name], rows) for name in BASELINES}
    output = args.suite / 'historical_752'
    cache = output / 'line_cache'
    cache.mkdir(parents=True, exist_ok=True)
    test_identity = dict(binding_sha256=ctx['binding_sha256'], checkpoint_sha256=best['sha256'],
                         historical_sha256=file_hash(test_path), baseline_sha256=file_hash(baseline_path))
    identity_file = output / 'identity.json'
    if identity_file.exists() and read(identity_file) != test_identity:
        raise ValueError('Historical test cache belongs to a different checkpoint or dataset')
    write(identity_file, test_identity)
    if (output / 'seal.json').exists():
        stats(ctx)
        return
    model = None
    scored = []
    for index, row in enumerate(rows, 1):
        path = cache / (digest(row['sample_id']) + '.json')
        if path.exists():
            prediction = read(path)
            if prediction['sample_id'] != row['sample_id'] or prediction['image_sha256'] != row['image_sha256']:
                raise ValueError('Historical prediction cache line identity mismatch')
        else:
            if model is None:
                seed_all(RECIPE['seed'])
                saved = torch.load(args.suite / best['checkpoint'], map_location='cpu', weights_only=True)
                if (saved['epoch'], saved['binding_sha256'], saved['trial']) != (best['epoch'], ctx['binding_sha256'], TRIAL):
                    raise ValueError('Selected checkpoint identity mismatch')
                model = make_model(ctx).eval()
                model.load_state_dict(saved['model'], strict=True)
                del saved
            prediction = ar.predict_one(model, row, ctx)
            write(path, prediction)
        scored.append(dict(prediction, model=TRIAL))
        if index % 100 == 0:
            print(f'Historical test lines processed = {index}/752', flush=True)
    predictions[TRIAL] = indexed(scored, rows)
    results = {name: metrics(rows, values) for name, values in predictions.items()}
    report = dict(identity=test_identity, selected_epoch=best['epoch'], scope='historical development-used comparison',
                  results=results, mdiff_vs_ar=damage(rows, predictions[AR_TRIAL], predictions[TRIAL]),
                  mdiff_vs_span=damage(rows, predictions['span_merge'], predictions[TRIAL]))
    predictions_file = output / 'predictions.jsonl'
    predictions_file.write_bytes(b''.join(canonical(row) for row in scored))
    write(output / 'report.json', report)
    write(output / 'seal.json', dict(report_sha256=file_hash(output / 'report.json'),
                                     predictions_sha256=file_hash(predictions_file)))
    stats(ctx)


def stats(ctx):
    output = ctx['args'].suite / 'historical_752'
    seal = read(output / 'seal.json')
    require_hash(output / 'report.json', seal['report_sha256'])
    require_hash(output / 'predictions.jsonl', seal['predictions_sha256'])
    report = read(output / 'report.json')
    if report['identity']['binding_sha256'] != ctx['binding_sha256']:
        raise ValueError('Historical report binding mismatch')
    print(f'HISTORICAL DEVELOPMENT COMPARISON | MDiff selected epoch = {report["selected_epoch"]}')
    print(f'{"Model":24} {"Lines":>5} {"CER %":>7} {"WER %":>7} {"Exact %":>8} {"Sub":>5} {"Ins":>5} {"Del":>5} {"No EOS":>6}')
    with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Model', 'Lines', 'CER %', 'WER %', 'Exact %', 'Sub', 'Ins', 'Del', 'No EOS'])
        for name in (*BASELINES, TRIAL):
            m = report['results'][name]
            row = [name, m['lines'], *(100 * m[key] for key in ('cer', 'wer', 'exact_line_accuracy')),
                   *(m[key] for key in ('substitutions', 'insertions', 'deletions', 'eos_failures'))]
            print('{:24} {:5d} {:7.2f} {:7.2f} {:8.2f} {:5d} {:5d} {:5d} {:6d}'.format(*row))
            writer.writerow(row)
    print(f'Report file = {output / "report.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', required=True, choices=('Check', 'Train', 'Test', 'Stats'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and args.action != 'Train':
        parser.error('--resume requires Train')
    args.project = args.project.resolve()
    ctx = context(args)
    with ar.run_lock(args.suite):
        if args.action in ('Check', 'Train'):
            rows, validation = check(ctx)
            if args.action == 'Check':
                synthetic_smoke(ctx)
                print('Verified all 12,304 training and 150 validation inputs, matched initialization, '
                      'and synthetic loss, backward pass and decoding at maximum image width.', flush=True)
            else:
                train(ctx, rows, validation)
        elif args.action == 'Test':
            test(ctx)
        else:
            stats(ctx)


if __name__ == '__main__':
    main()
