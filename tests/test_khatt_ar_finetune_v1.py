"""Verify the two fine-tune arms and restart behavior using synthetic inputs."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from thesis.diffusion.tokenizer import CharacterTokenizer
from thesis.scripts import run_khatt_ar_finetune_v1 as runner


class ToyRecognizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.head = torch.nn.Linear(1, 1)

    def loss(self, pixels, widths, target):
        return (self.head(self.encoder(pixels)) + torch.rand(1) * .1).square().mean(), {}


class FineTuneTests(unittest.TestCase):
    def test_optimizer_groups_start_fresh_and_only_decoder_rate_differs(self):
        model = ToyRecognizer()
        optimizers = [runner.optimizer_for(model, trial) for trial in runner.VARIANTS]
        self.assertEqual([o.param_groups[0]['base_lr'] for o in optimizers], [1e-5, 1e-5])
        self.assertEqual([o.param_groups[1]['base_lr'] for o in optimizers], [1e-4, 3e-4])
        self.assertTrue(all(not o.state for o in optimizers))
        for index in range(2):
            groups = [{k: v for k, v in o.param_groups[index].items() if k not in ('lr', 'base_lr')}
                      for o in optimizers]
            self.assertEqual(groups[0], groups[1])

    def test_pending_validation_resume_keeps_the_same_optimizer_and_weights(self):
        rows = [dict(sample_id='train', text='ا')]
        validation = [dict(sample_id='validation', text='ا')]
        trial = 'finetune_lr3e4'
        train_epochs = []
        original_epoch = runner.train_epoch

        def update(ctx, model, optimizer, rows, epoch, updates):
            train_epochs.append(epoch)
            self.assertEqual(optimizer.param_groups[1]['base_lr'], 3e-4)
            return original_epoch(ctx, model, optimizer, rows, epoch, updates)

        def predict(model, row, ctx):
            return dict(sample_id=row['sample_id'], text='ا', status='completed', has_eos=True)

        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()) as stdout, \
                patch.object(runner, 'make_model', side_effect=lambda *a, **k: ToyRecognizer()), \
                patch.object(runner, 'train_epoch', side_effect=update), \
                patch.object(runner.ar, 'remaining_epochs', side_effect=lambda epoch, stale: range(epoch + 1, 3)), \
                patch.object(runner.ar, 'image_batch', return_value=(torch.ones(1, 1), torch.ones(1))), \
                patch.object(runner.ar, 'predict_one', side_effect=predict) as decoder:
            checkpoints = []
            for name in ('continuous', 'interrupted'):
                suite = Path(temp) / name
                ctx = dict(args=SimpleNamespace(suite=suite, device='cpu', resume=True),
                           tokenizer=CharacterTokenizer(['ا']), binding_sha256='synthetic')
                train_epochs.clear()
                if name == 'interrupted':
                    decoder.side_effect = KeyboardInterrupt
                    with self.assertRaises(KeyboardInterrupt):
                        runner.train(ctx, trial, rows, validation)
                    self.assertEqual(runner.read(suite / trial / 'pending_validation.json')['epoch'], 1)
                    decoder.side_effect = predict
                runner.train(ctx, trial, rows, validation)
                self.assertEqual(train_epochs, [1, 2])
                folder, best = runner.selected(ctx, trial)
                self.assertEqual(best['epoch'], 1)
                last = runner.read(folder / 'last.json')
                self.assertEqual(last['stale'], 1)
                checkpoints.append(torch.load(folder / last['path'], weights_only=True))
            self.assertIn('Resuming validation for epoch 1.', stdout.getvalue())
            self.assertIn('Epoch = 2 | CER = 0.00% | WER = 0.00% | Exact  = 100.00% | Time for epoch = ', stdout.getvalue())
        first, second = checkpoints
        self.assertEqual(first['updates'], second['updates'])
        torch.testing.assert_close(first['model'], second['model'], rtol=0, atol=0)
        torch.testing.assert_close(first['optimizer'], second['optimizer'], rtol=0, atol=0)
        self.assertTrue(torch.equal(first['torch_rng'], second['torch_rng']))

    def test_test_requires_both_completed_trials_before_reading_references(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, 'jsonl') as load_rows:
            ctx = dict(args=SimpleNamespace(suite=Path(temp)), provenance={})
            with self.assertRaisesRegex(ValueError, 'Finish training'):
                runner.test(ctx)
            load_rows.assert_not_called()


if __name__ == '__main__':
    unittest.main()
