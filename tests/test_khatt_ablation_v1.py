"""CPU-only protocol tests. Never trains or evaluates real OCR checkpoints."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import torch

from thesis.khatt_ablation_v1 import data, workflow as w
from thesis.mdiff_arabic_v1.artifacts import canonical, file_hash, read, write
from thesis.scripts.run_khatt_ablation_v1 import parse_args, main


class ProtocolTests(unittest.TestCase):
    def test_all_codes_and_logical_ligatures(self):
        text,delimiter=data.decode_cells(['laaa','sp','laae','laah','laam','tee','teE','',';','',''])
        self.assertEqual(text,'لا لألإلآةة');self.assertTrue(delimiter)
        self.assertEqual(data.decode_cells(['ma','','aa','sp','n0','com','qts'])[0],'ما 0,?')
        self.assertEqual(data.CODE_MAP['ka'],'ق');self.assertEqual(data.CODE_MAP['ke'],'ك')
        self.assertEqual(data.CODE_MAP['aa'],'ا');self.assertEqual(data.CODE_MAP['ae'],'أ')
        with self.assertRaises(ValueError):data.decode_cells(['aa',';','ba'])
        with self.assertRaises(ValueError):data.decode_cells(['unknown'])

    def test_five_nested_subsets_and_updated_children_counts(self):
        adult=[dict(sample_id='khatt:'+str(i)) for i in range(9497)]
        subsets=data.nested_subsets(adult)
        self.assertEqual({p:len(r) for p,r in subsets.items()},{10:949,30:2849,50:4748,100:9497})
        self.assertEqual(subsets,data.nested_subsets(list(reversed(adult))))
        sets={p:{r['sample_id'] for r in rows} for p,rows in subsets.items()}
        self.assertTrue(sets[10]<sets[30]<sets[50]<sets[100])
        counts={n:len(subsets[p])+(2807 if ours else 0) for n,(p,ours) in data.TRIALS.items()}
        self.assertEqual(list(counts.values()),[9497,12304,7555,5656,3756])

    def test_patience_epoch_limits_and_schedule(self):
        self.assertEqual(list(w.remaining_epochs(99,0)),[100])
        self.assertEqual(list(w.remaining_epochs(100,0)),[])
        self.assertEqual(list(w.remaining_epochs(11,10)),[])
        self.assertAlmostEqual(w.lr_factor(0,10),1/50)
        self.assertAlmostEqual(w.lr_factor(49,10),1)
        self.assertAlmostEqual(w.lr_factor(1000,10),.1)
        self.assertEqual(w.RECIPE['maximum_epochs'],100)
        self.assertEqual(w.RECIPE['patience'],10)

    def test_manifest_and_source_conversion_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'Train.csv';source.write_text('original')
            manifest=root/'validation.jsonl';manifest.write_bytes(canonical(dict(sample_id='child')))
            prepared=dict(version='khatt_ablation_v1',khatt_mapping_sha256=data.digest(data.CODE_MAP),
                inputs={str(source):file_hash(source)},files={manifest.name:file_hash(manifest)})
            write(root/'prepared.json',prepared);args=SimpleNamespace(suite=root)
            self.assertEqual(data.verify(args),prepared)
            manifest.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):data.verify(args)
            manifest.write_bytes(canonical(dict(sample_id='child')));source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'input changed'):data.verify(args)

    def test_only_owned_obsolete_checkpoint_files_pruned(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);keep=root/'e001_one';old=root/'e002_two'
            for folder in (keep,old):
                folder.mkdir()
                for name in ('state.pt','model.pt','history.json','validation.jsonl'):(folder/name).write_text('x')
            outside=root/'old_baseline.pt';outside.write_text('never delete')
            w.prune_states(root,['e001_one/model.pt','e002_two/state.pt'])
            self.assertTrue((keep/'model.pt').exists());self.assertTrue((old/'state.pt').exists())
            self.assertFalse((keep/'state.pt').exists());self.assertFalse((old/'model.pt').exists())
            self.assertTrue(outside.exists());self.assertTrue((old/'history.json').exists())

    def test_inference_never_receives_reference(self):
        from thesis.diffusion.tokenizer import CharacterTokenizer
        tokenizer=CharacterTokenizer(['ا'])
        class Toy:
            def generate(self,px,widths):
                return torch.tensor([[3,2]]),dict(decoder_passes=2,sampler_mode='greedy_causal')
        ctx=dict(args=SimpleNamespace(device='cpu'),tokenizer=tokenizer)
        row=dict(sample_id='x',image_path='x.png',image_sha256='hash',text='SECRET',label_path='SECRET')
        def fake_batch(actual,context):
            self.assertNotIn('text',actual);self.assertNotIn('label_path',actual)
            return torch.zeros(1,3,32,4),torch.tensor([4])
        with patch.object(w,'image_batch',side_effect=fake_batch):
            result=w.predict_one(Toy(),row,ctx)
        self.assertEqual(result['text'],'ا');self.assertTrue(result['has_eos'])

    def test_real_recognizer_return_contract_on_synthetic_cpu_image(self):
        # Execute the real encoder/decoder with random weights and one synthetic
        # image. No dataset images, trained checkpoint, or optimizer is used.
        from thesis.diffusion.tokenizer import CharacterTokenizer
        from thesis.mdiff_arabic_v1.model import Recognizer
        torch.set_num_threads(2)
        tokenizer=CharacterTokenizer(['ا'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model=Recognizer(len(tokenizer),capacity=4,arm='ar').eval()
            ctx=dict(args=SimpleNamespace(device='cpu'),tokenizer=tokenizer)
            row=dict(sample_id='synthetic',image_path='unused',image_sha256='synthetic')
            with patch.object(w,'image_batch',return_value=(torch.zeros(1,3,32,32),torch.tensor([32]))):
                prediction=w.predict_one(model,row,ctx)
        self.assertIsInstance(prediction['text'],str)
        self.assertIsInstance(prediction['has_eos'],bool)
        self.assertEqual(prediction['status'],'completed')
        self.assertLessEqual(len(prediction['text']),4)
        self.assertTrue(set(prediction['text'])<={'ا'})

    def test_inference_rejects_wrong_batch_shape(self):
        class WrongBatch:
            def generate(self,px,widths):return torch.tensor([[3,2],[3,2]]),{}
        ctx=dict(args=SimpleNamespace(device='cpu'))
        row=dict(sample_id='x',image_path='unused',image_sha256='synthetic')
        with patch.object(w,'image_batch',return_value=(torch.zeros(1,3,32,4),torch.tensor([4]))):
            with self.assertRaisesRegex(ValueError,'one batch of token IDs'):
                w.predict_one(WrongBatch(),row,ctx)

    def test_table_keeps_failed_lines_and_frozen_span_damage(self):
        rows=[dict(sample_id='a',text='ا'),dict(sample_id='b',text='ب')]
        span=[dict(sample_id='a',text='ا'),dict(sample_id='b',text='ب')]
        other=[dict(sample_id='a',text='ا'),dict(sample_id='b',text='',status='failed',has_eos=False)]
        results=w.comparison(rows,dict(span_merge=span,khatt100_only=other))
        result=results['khatt100_only']
        self.assertEqual(result['metrics']['lines'],2);self.assertEqual(result['metrics']['failed_lines'],1)
        self.assertEqual(result['metrics']['reference_chars'],2)
        self.assertEqual(result['vs_span_merge']['exact_lines_lost'],1)
        with self.assertRaises(ValueError):w.comparison(rows,dict(span_merge=span,broken=other[:1]))

    def test_trainall_cannot_call_test(self):
        args=parse_args(['--action','TrainAll','--project','.','--resume'])
        with tempfile.TemporaryDirectory() as temp:
            args.suite=Path(temp)
            calls=[]
            with patch('thesis.scripts.run_khatt_ablation_v1.parse_args',return_value=args), \
                 patch.object(w,'context',return_value={}),patch.object(w,'check'), \
                 patch.object(w,'train_one',side_effect=lambda ctx,name:calls.append(name)), \
                 patch.object(w,'test_all',side_effect=AssertionError('Test must never run automatically')):
                main()
            self.assertEqual(calls,list(data.TRIALS))

    def test_test_requires_all_completed_runs_before_reading_references(self):
        with patch.object(w,'selected',side_effect=FileNotFoundError('Not trained')), \
             patch.object(w,'jsonl',side_effect=AssertionError('Test references accessed too soon')):
            with self.assertRaises(FileNotFoundError):w.test_all(dict(args=SimpleNamespace()))

    def test_epoch_transaction_resume_replays_interruption_and_stops_at_patience(self):
        # Tiny synthetic CPU network, not Recognizer or any real OCR data/model.
        class Toy(torch.nn.Module):
            def __init__(self,*args,**kwargs):
                super().__init__();self.encoder=torch.nn.Linear(1,1);self.head=torch.nn.Linear(1,1)
            def loss(self,px,widths,target):
                # Include RNG consumption so state restoration is actually tested.
                value=self.head(self.encoder(px))+torch.rand(1)*.01
                loss=(value-.25).square().mean()
                return loss,{}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            results=[]
            for label,interrupt in (('continuous',False),('interrupted',True)):
                suite=root/label;suite.mkdir()
                row=dict(sample_id='toy',text='ا',image_path='unused',image_sha256='toy')
                for name in ('khatt100_only','validation'):(suite/(name+'.jsonl')).write_bytes(canonical(row))
                from thesis.diffusion.tokenizer import CharacterTokenizer
                ctx=dict(args=SimpleNamespace(suite=suite,device='cpu',resume=False),
                    tokenizer=CharacterTokenizer(['ا']),assets=root,binding_sha256='toy-binding',
                    prepared=dict(counts={'khatt100_only':dict(total=1,validation=1)}))
                training_calls=0
                def toy_batch(actual,context,epoch):
                    nonlocal training_calls
                    training_calls+=1
                    if interrupt and training_calls==2:raise KeyboardInterrupt('Synthetic interrupted epoch')
                    return torch.ones(1,1),torch.tensor([1])
                with ExitStack() as stack:
                    stack.enter_context(patch.object(w,'Recognizer',Toy))
                    stack.enter_context(patch.object(w,'load_encoder',return_value=dict(encoder_initialization_sha256='toy')))
                    stack.enter_context(patch.object(w,'verify_images'))
                    stack.enter_context(patch.object(w,'image_batch',side_effect=toy_batch))
                    stack.enter_context(patch.object(w,'predict_one',return_value=dict(sample_id='toy',text='ا',has_eos=True)))
                    stack.enter_context(patch('builtins.print'))
                    if interrupt:
                        with self.assertRaises(KeyboardInterrupt):w.train_one(ctx,'khatt100_only')
                        self.assertEqual(read(suite/'runs/khatt100_only/last.json')['epoch'],1)
                        ctx['args'].resume=True
                    w.train_one(ctx,'khatt100_only')
                folder=suite/'runs/khatt100_only';complete=read(folder/'complete.json')
                self.assertEqual(complete['epoch'],11);self.assertEqual(complete['best']['epoch'],1)
                self.assertEqual(complete['stop_reason'],'patience');self.assertFalse(complete['test_used'])
                pointer=read(folder/'last.json')
                state=torch.load(folder/pointer['path'],map_location='cpu',weights_only=True)
                results.append((state,w.tensor_hash(state['model'])))
                # Only one resume optimizer and one selected model are retained.
                self.assertEqual(len(list(folder.rglob('state.pt'))),1)
                self.assertEqual(len(list(folder.rglob('model.pt'))),1)
            self.assertEqual(results[0][1],results[1][1])
            self.assertTrue(torch.equal(results[0][0]['torch_rng'],results[1][0]['torch_rng']))
            for key in results[0][0]['optimizer']['state']:
                a=results[0][0]['optimizer']['state'][key];b=results[1][0]['optimizer']['state'][key]
                for field in a:self.assertTrue(torch.equal(a[field],b[field]))

    def test_cli_rejects_old_recipe_and_ambiguous_train(self):
        for extra in (['--epochs','200'],['--patience','30']):
            with self.assertRaises(SystemExit):parse_args(['--action','TrainAll','--project','.']+extra)
        with self.assertRaises(SystemExit):parse_args(['--action','Train','--project','.'])


if __name__=='__main__':unittest.main()
