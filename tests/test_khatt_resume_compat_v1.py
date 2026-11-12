"""Compatibility checks use synthetic metadata; no real training/inference."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from thesis import khatt_resume_compat_v1 as c
from thesis.mdiff_arabic_v1.artifacts import digest, file_hash, read, write


def binding():
    return dict(environment=dict(packages={'torch':'2.10.0+cu126','numpy':'2.3.2'},
        device='cuda',cuda='12.6',gpu='GPU',python='3.13'),sources={'model.py':'model-hash'},
        recipe={'maximum_epochs':100,'patience':10},prepared_sha256='data-hash',
        baselines={'predictions.jsonl':'frozen-hash'})


class CompatibilityTests(unittest.TestCase):
    def test_exact_environment_or_only_reviewed_addition(self):
        old=binding();actual=copy.deepcopy(old)
        self.assertEqual(c.compatible_additions(old,actual),{})
        actual['environment']['packages']['opencv-python']='5.0.0.93'
        before=copy.deepcopy(actual)
        self.assertEqual(c.compatible_additions(old,actual),{'opencv-python':'5.0.0.93'})
        self.assertEqual(actual,before)
        self.assertNotIn('opencv-python',old['environment']['packages'])

    def test_training_dependency_code_data_and_device_changes_still_fail(self):
        old=binding()
        mutations=[lambda a:a['environment']['packages'].update(torch='3.0'),
            lambda a:a['environment']['packages'].pop('numpy'),
            lambda a:a['environment'].update(device='cpu'),
            lambda a:a['sources'].update({'model.py':'changed'}),
            lambda a:a['recipe'].update(patience=20),
            lambda a:a.update(prepared_sha256='changed'),
            lambda a:a['baselines'].update({'predictions.jsonl':'changed'})]
        for mutate in mutations:
            actual=copy.deepcopy(old)
            actual['environment']['packages']['opencv-python']='5.0.0.93'
            mutate(actual)
            with self.subTest(actual=actual),self.assertRaisesRegex(ValueError,'rejected changes'):
                c.compatible_additions(old,actual)

    def test_unreviewed_package_versions_and_existing_package_changes_fail(self):
        for name,version in (('opencv-python','6.0'),('some-other-library','1.0')):
            actual=binding();actual['environment']['packages'][name]=version
            with self.assertRaisesRegex(ValueError,'Unreviewed added package'):
                c.compatible_additions(binding(),actual)
        old=binding();old['environment']['packages']['opencv-python']='4.0'
        actual=copy.deepcopy(old);actual['environment']['packages']['opencv-python']='5.0.0.93'
        with self.assertRaisesRegex(ValueError,'rejected changes'):c.compatible_additions(old,actual)

    def test_static_or_dynamic_opencv_references_and_loaded_module_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.py'
            path.write_text('import math\n',encoding='utf-8')
            c.verify_opencv_unused([path])
            for source in ('import cv2\n','from cv2 import imread\n','x=importlib.import_module("cv2")\n'):
                path.write_text(source,encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'references OpenCV'):c.verify_opencv_unused([path])
            path.write_text('import math\n',encoding='utf-8')
            with patch.dict(c.sys.modules,{'cv2':object()}):
                with self.assertRaisesRegex(ValueError,'OpenCV is loaded'):c.verify_opencv_unused([path])

    def test_original_check_preserved_and_observed_environment_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);suite=root/'suite';suite.mkdir()
            repo=root/'repo';project=root/'project'
            for path in (repo/'thesis/scripts/resume_khatt_ablation_v1.py',
                         repo/'tests/test_khatt_resume_compat_v1.py',project/'Diffusion/resume_khatt_ablation_v1.ps1'):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# helper\n')
            source=root/'model.py';source.write_text('import math\n')
            old=binding();old['sources']={str(source):file_hash(source)}
            saved=dict(binding=old,binding_sha256=digest(old));write(suite/'check.json',saved)
            original_bytes=(suite/'check.json').read_bytes()
            actual=copy.deepcopy(old);actual['environment']['packages']['opencv-python']='5.0.0.93'
            args=SimpleNamespace(suite=suite,repo=repo,project=project)
            ctx=dict(args=args,binding=actual,binding_sha256=digest(actual))
            with patch.object(c.w,'context',return_value=ctx):
                result,audit=c.compatibility_context(args)
            self.assertEqual((suite/'check.json').read_bytes(),original_bytes)
            self.assertEqual(result['binding_sha256'],saved['binding_sha256'])
            self.assertEqual(result['binding'],old)
            self.assertEqual(result['observed_binding'],actual)
            evidence=read(audit/'compatibility.json')
            self.assertEqual(evidence['observed_binding'],actual)
            self.assertEqual(evidence['observed_binding_sha256'],digest(actual))
            self.assertEqual(evidence['reviewed_added_packages'],{'opencv-python':'5.0.0.93'})
            self.assertEqual(len(list((audit/'source').iterdir())),4)

    def test_checkpoint_hashes_checked_without_loading_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'runs/khatt100_only';folder.mkdir(parents=True)
            write(folder/'run.json',dict(binding_sha256='original'))
            (folder/'state.pt').write_bytes(b'synthetic checkpoint')
            write(folder/'last.json',dict(path='state.pt',epoch=35,sha256=file_hash(folder/'state.pt')))
            ctx=dict(args=SimpleNamespace(suite=root),binding_sha256='original')
            progress=c.verify_progress(ctx)
            self.assertEqual(progress['khatt100_only']['resume_epoch'],36)
            self.assertEqual(progress['khatt100_ours']['status'],'not_started')
            (folder/'state.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint hash mismatch'):c.verify_progress(ctx)

    def test_check_action_never_trains_and_trainall_does_not_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audit=root/'audit';audit.mkdir();write(audit/'compatibility.json',{})
            args=SimpleNamespace(suite=root,action='Check')
            ctx=dict(args=args,binding_sha256='original')
            with patch.object(c,'compatibility_context',return_value=(ctx,audit)), \
                 patch.object(c,'verify_progress',return_value={}), \
                 patch.object(c.w,'train_one') as train,patch.object(c.w,'test_all') as test:
                c.run(args);train.assert_not_called();test.assert_not_called()
                args.action='TrainAll';c.run(args)
                self.assertEqual([x.args[1] for x in train.call_args_list],list(c.w.TRIALS))
                test.assert_not_called()


if __name__=='__main__':unittest.main()
