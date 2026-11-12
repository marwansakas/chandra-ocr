"""Four-run selection and abandonment tests; no real OCR or training."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from thesis import khatt_resume_compat_v1 as compat
from thesis import khatt_subset_eval_v1 as subset
from thesis.mdiff_arabic_v1.artifacts import read,write
from thesis.scripts.resume_khatt_ablation_v1 import parse_args


class SubsetTests(unittest.TestCase):
    def context(self,root):
        test=root/'repo/tests/test_khatt_subset_eval_v1.py'
        test.parent.mkdir(parents=True);test.write_text('# synthetic helper\n')
        suite=root/'suite';suite.mkdir()
        return dict(args=SimpleNamespace(suite=suite,repo=root/'repo',action='Check'),binding_sha256='original')

    def selected(self,ctx,name):
        self.assertIn(name,subset.INCLUDED)
        return ctx['args'].suite/'runs'/name,dict(epoch={'khatt100_only':34,'khatt100_ours':40,
            'khatt50_ours':34,'khatt30_ours':87}[name],checkpoint=name+'.pt',sha256=name+'-hash')

    def test_prepare_includes_four_completed_validation_best_and_preserves_10(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.context(Path(temp));suite=ctx['args'].suite
            ten=suite/'runs/khatt10_ours';ten.mkdir(parents=True)
            write(ten/'best.json',dict(epoch=15,sha256='untouched'))
            old=(ten/'best.json').read_bytes()
            with patch.object(subset.w,'selected',side_effect=self.selected) as select, \
                 patch.object(subset.w,'test_all') as test,patch.object(subset.w,'train_one') as train:
                request=subset.prepare(ctx)
                self.assertEqual([c.args[1] for c in select.call_args_list],list(subset.INCLUDED))
                test.assert_not_called();train.assert_not_called()
            self.assertEqual((ten/'best.json').read_bytes(),old)
            self.assertFalse((ten/'complete.json').exists())
            value=read(request)
            self.assertEqual(list(value['decision']['included_trials']),list(subset.INCLUDED))
            self.assertEqual(value['decision']['historical_test_lines'],752)
            self.assertEqual(value['selected']['khatt30_ours']['epoch'],87)
            self.assertEqual(subset.decision(ctx)['abandoned_trials'],['khatt10_ours'])

    def test_missing_included_checkpoint_does_not_mark_abandoned_or_test(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.context(Path(temp))
            with patch.object(subset.w,'selected',side_effect=FileNotFoundError('missing')), \
                 patch.object(subset.w,'test_all') as test:
                with self.assertRaises(FileNotFoundError):subset.prepare(ctx)
                test.assert_not_called()
            self.assertIsNone(subset.decision(ctx))

    def test_trial_override_restored_after_exception_without_mutating_original(self):
        original=subset.w.TRIALS;before=dict(original)
        with self.assertRaisesRegex(RuntimeError,'failure'):
            with subset.evaluation_trials():
                self.assertEqual(tuple(subset.w.TRIALS),subset.INCLUDED)
                self.assertIn(subset.EXCLUDED,original)
                raise RuntimeError('failure')
        self.assertIs(subset.w.TRIALS,original);self.assertEqual(original,before)

    def test_evaluation_calls_existing_evaluator_with_four_then_seals_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.context(Path(temp));suite=ctx['args'].suite
            def fake_test(actual):
                self.assertIs(actual,ctx)
                self.assertEqual(tuple(subset.w.TRIALS),subset.INCLUDED)
                output=suite/'historical_752';output.mkdir()
                write(output/'report.json',dict(results={n:{} for n in subset.INCLUDED}))
                (output/'predictions.jsonl').write_text('synthetic outputs')
            with patch.object(subset.w,'selected',side_effect=self.selected), \
                 patch.object(subset.w,'test_all',side_effect=fake_test):
                subset.evaluate(ctx,'TestAll')
            result=read(suite/'test_plan_without10/result_binding.json')
            self.assertEqual(result['included_trials'],list(subset.INCLUDED))
            self.assertEqual(result['abandoned_trials'],[subset.EXCLUDED])
            self.assertIn(subset.EXCLUDED,subset.w.TRIALS)

    def test_trainall_skips_abandoned_and_explicit_train_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.context(Path(temp));args=ctx['args'];args.action='TrainAll'
            audit=args.suite/'audit';audit.mkdir();write(audit/'compatibility.json',{})
            with patch.object(subset.w,'selected',side_effect=self.selected):subset.prepare(ctx)
            with patch.object(compat,'compatibility_context',return_value=(ctx,audit)), \
                 patch.object(compat,'verify_progress',return_value={}), \
                 patch.object(compat.w,'train_one') as train,patch.object(compat.w,'test_all') as test:
                compat.run(args)
                self.assertEqual([c.args[1] for c in train.call_args_list],list(subset.INCLUDED))
                test.assert_not_called();train.reset_mock()
                args.action='Train';args.trial='khatt10_ours'
                with self.assertRaisesRegex(ValueError,'abandoned'):compat.run(args)
                train.assert_not_called()

    def test_cli_new_flag_does_not_change_original_parser(self):
        args=parse_args(['--action','TestAll','--project','.','--skip-10-percent'])
        self.assertTrue(args.skip_10_percent);self.assertEqual(args.action,'TestAll')
        with self.assertRaises(SystemExit):
            parse_args(['--action','TrainAll','--project','.','--skip-10-percent'])


if __name__=='__main__':unittest.main()
