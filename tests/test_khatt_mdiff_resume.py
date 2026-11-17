"""Check the MDiff epoch transaction without training on experiment data."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from thesis.diffusion.tokenizer import CharacterTokenizer
from thesis.scripts import run_khatt_mdiff_v1 as runner


class ToyRecognizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.head = torch.nn.Linear(1, 1)

    def loss(self, pixels, widths, target):
        return (self.head(self.encoder(pixels)) + torch.rand(1) * .1).square().mean(), {}


class ResumeTests(unittest.TestCase):
    def test_epoch_console_format(self):
        self.assertEqual(runner.epoch_line(4, dict(cer=.2992, wer=.6574, exact_line_accuracy=.246), 3723),
                         'Epoch = 4 | CER = 29.92% | WER = 65.74% | Exact  = 24.60% | Time for epoch = 01:02:03')

    def test_resume_reuses_saved_training_when_validation_was_interrupted(self):
        rows = [dict(sample_id='train', text='ا')]
        validation = [dict(sample_id='validation', text='ا')]
        train_epochs = []
        original = runner.train_epoch

        def train_epoch(ctx, model, optimizer, rows, epoch, updates):
            train_epochs.append(epoch)
            return original(ctx, model, optimizer, rows, epoch, updates)

        def predict(model, row, ctx):
            return dict(sample_id=row['sample_id'], text='ا', status='completed', has_eos=True)

        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()) as stdout, \
                patch.object(runner, 'make_model', side_effect=lambda *a, **k: ToyRecognizer()), \
                patch.object(runner, 'train_epoch', side_effect=train_epoch), \
                patch.object(runner.ar, 'remaining_epochs', side_effect=lambda epoch, stale: range(epoch + 1, 4)), \
                patch.object(runner.ar, 'image_batch', return_value=(torch.ones(1, 1), torch.ones(1))), \
                patch.object(runner.ar, 'predict_one', side_effect=predict) as decoder:
            final_states = []
            for name in ('continuous', 'interrupted'):
                suite = Path(temp) / name
                ctx = dict(args=SimpleNamespace(suite=suite, device='cpu', resume=True),
                           tokenizer=CharacterTokenizer(['ا']), binding_sha256='synthetic')
                train_epochs.clear()
                if name == 'interrupted':
                    decoder.side_effect = KeyboardInterrupt
                    with self.assertRaises(KeyboardInterrupt):
                        runner.train(ctx, rows, validation)
                    self.assertEqual(runner.read(suite / 'pending_validation.json')['epoch'], 1)
                    self.assertFalse((suite / 'last.json').exists())
                    decoder.side_effect = predict
                runner.train(ctx, rows, validation)
                self.assertEqual(train_epochs, [1, 2, 3])
                self.assertFalse((suite / 'pending_validation.json').exists())
                self.assertEqual(runner.selected(ctx)['epoch'], 1)
                last = runner.read(suite / 'last.json')
                self.assertEqual(last['stale'], 2)
                final_states.append(torch.load(suite / last['path'], weights_only=True))
            self.assertIn('Resuming validation for epoch 1.', stdout.getvalue())
        first, second = final_states
        self.assertEqual(first['updates'], second['updates'])
        self.assertTrue(torch.equal(first['torch_rng'], second['torch_rng']))
        for key in first['model']:
            torch.testing.assert_close(first['model'][key], second['model'][key], rtol=0, atol=0)
        for key, fields in first['optimizer']['state'].items():
            for field, value in fields.items():
                torch.testing.assert_close(value, second['optimizer']['state'][key][field], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
