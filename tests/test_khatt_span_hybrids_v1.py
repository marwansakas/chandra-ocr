"""Check the hybrid's margin, preservation requirement, and validation boundary."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from thesis.scripts import run_khatt_span_hybrids_v1 as runner


def prediction(text, has_eos=True):
    return dict(text=text, has_eos=has_eos, status='completed')


def candidates(base, alternative, gain):
    return [dict(text=base, base_score=-1., donor_score=-1.),
            dict(text=alternative, base_score=-1. + gain, donor_score=-1. + gain)]


class HybridTests(unittest.TestCase):
    def test_spans_only_and_failed_eos(self):
        self.assertFalse(runner.propose(prediction('خط عربي'), prediction('خط عملي', has_eos=False)))
        self.assertFalse(runner.propose(prediction('a' * 20), prediction('b' * 20)))
        self.assertTrue(runner.propose(prediction('خط عربي'), prediction('خط عملي')))

    def test_margin_is_strict(self):
        scores = candidates('abc', 'abd', .25)
        self.assertEqual(runner.select(prediction('abc'), scores, dict(donor_weight=.5, margin=.25))['text'], 'abc')
        self.assertEqual(runner.select(prediction('abc'), scores, dict(donor_weight=.5, margin=.15))['text'], 'abd')

    def test_fit_preserves_exact_lines(self):
        rows = [dict(sample_id='fixed', text='abcdef'), dict(sample_id='correct', text='hello')]
        base = {'fixed': dict(prediction('xyzuvw'), sample_id='fixed'),
                'correct': dict(prediction('hello'), sample_id='correct')}
        scores = {'fixed': candidates('xyzuvw', 'abcdef', .5), 'correct': candidates('hello', 'hella', .125)}
        fit = runner.fit(rows, base, scores)
        self.assertTrue(fit['selected']['eligible'])
        self.assertEqual(runner.apply_config(base, scores, fit['config'])['correct']['text'], 'hello')
        self.assertEqual(runner.apply_config(base, scores, fit['config'])['fixed']['text'], 'abcdef')
        scores['correct'] = candidates('hello', 'hella', .5)
        self.assertIsNone(runner.fit(rows, base, scores)['config'])

    def test_test_does_not_read_historical_inputs_without_calibration(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, 'historical_inputs') as historical:
            with self.assertRaisesRegex(ValueError, 'Run Calibrate before Test'):
                runner.test(dict(output=Path(temp)))
            historical.assert_not_called()

    def test_only_original_runner_revision_may_reuse_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            path = output / 'validation/selection.json'
            current = dict(source_sha256={'run_khatt_span_hybrids_v1.py': 'current', 'logic.py': 'same'})
            ctx = dict(output=output, binding=current, binding_sha256=runner.digest(current))
            old_sources = dict(current['source_sha256'])
            old_sources['run_khatt_span_hybrids_v1.py'] = runner.PRE_SHORT_CACHE_RUNNER_SHA256
            binding = dict(source_sha256=old_sources)
            variants = {variant: dict(config=None) for variant in runner.DONORS}
            for should_load in (True, False):
                runner.write(path, dict(binding=binding, binding_sha256=runner.digest(binding),
                                        scope='children validation', lines=150, variants=variants))
                runner.write(path.parent / 'seal.json', dict(selection_sha256=runner.file_hash(path)))
                if should_load:
                    self.assertEqual(runner.load_selection(ctx)[0]['variants'], variants)
                    old_sources['logic.py'] = 'different'
                else:
                    with self.assertRaisesRegex(ValueError, 'does not match'):
                        runner.load_selection(ctx)


if __name__ == '__main__':
    unittest.main()
