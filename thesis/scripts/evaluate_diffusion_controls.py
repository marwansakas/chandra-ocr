"""Measure correct, shuffled-image and blank-image controls on one frozen partition."""
import argparse, json, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True); p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--split-root',type=Path,required=True); p.add_argument('--partition',choices=('validation','test'),default='test')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--steps',type=int,default=16)
    p.add_argument('--require-provenance',action='store_true',help='Reject checkpoints whose frozen data/split/tokenizer differ from this evaluation')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    import torch
    from PIL import Image
    from thesis.diffusion.data import file_hash, load_frozen, resolve_relative, json_bytes
    from thesis.diffusion.images import prepare_images
    from thesis.diffusion.decoding_diagnostics import greedy_decode
    from thesis.diffusion.metrics import aggregate
    from thesis.diffusion.runtime import get_device, load_checkpoint, seed_everything, verify_checkpoint_provenance
    from thesis.diffusion.tokenizer import training_text
    torch.set_num_threads(2); seed_everything(args.seed); device=get_device(args.device)
    model, tokenizer, image_config, metadata=load_checkpoint(args.model,device); model.eval()
    parts, lock, split_tokenizer=load_frozen(args.split_root,args.dataset_root)
    split_hash = file_hash(args.split_root/'split_lock.json')
    if args.require_provenance:
        verify_checkpoint_provenance(metadata, tokenizer, split_tokenizer, lock, split_hash)
    records=[{**r,'text':training_text(r['text'],model.config.text_policy)} for r in parts[args.partition]]
    images=[]
    for r in records:
        with Image.open(resolve_relative(args.dataset_root,r['image_path'])) as image: images.append(image.convert('RGB').copy())
    rng=random.Random(args.seed)
    shuffled = list(range(len(images)))
    if len(shuffled) > 1:
        shift = rng.randrange(1, len(shuffled))
        shuffled = shuffled[shift:] + shuffled[:shift]
    controls={'correct':list(range(len(images))),'shuffled':shuffled,'blank':[None]*len(images)}
    report={'partition':args.partition,'samples':len(records),'dataset_fingerprint':lock['dataset_fingerprint'],
            'split_lock_sha256':split_hash,'checkpoint_sha256':file_hash(args.model/'model.pt'),
            'decoding':'greedy_one_pass','steps':args.steps,'shuffle_seed':args.seed,
            'shuffled_image_mapping':[{"sample_id": records[index]['sample_id'], "image_sample_id": records[source]['sample_id']} for index, source in enumerate(shuffled)], 'controls':{}}
    with torch.inference_mode():
        for name, indices in controls.items():
            predictions=[]
            for start in range(0,len(records),8):
                batch=[]
                for pos in range(start,min(start+8,len(records))):
                    image=Image.new('RGB',images[pos].size,'white') if name=='blank' else images[indices[pos]]
                    batch.append((image,records[pos]))
                pixels,widths=prepare_images([x[0] for x in batch],image_config)
                ids,_=greedy_decode(model,pixels.to(device),widths.to(device),method='one_pass')
                for r,row in zip((x[1] for x in batch),ids.cpu().tolist()):
                    text,has_eos=tokenizer.decode(row); predictions.append((r['text'],text,has_eos))
            score=aggregate([(r,p) for r,p,_ in predictions])
            report['controls'][name]={'strict':score,'cer_percent':100*score['cer'],'exact_crops':score['exact_lines'],
                                      'no_eos':sum(not eos for _,_,eos in predictions),
                                      'mean_prediction_length':statistics.mean(len(p) for _,p,_ in predictions)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_bytes(json_bytes(report)); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
