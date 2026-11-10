"""Verify committed checkpoint/tokenizer bytes without loading models or inference."""
import hashlib
import json
from pathlib import Path


def main():
    repo=Path(__file__).resolve().parents[2]
    manifest=json.loads((repo/'thesis/experiments/arabic_ocr_best/manifest.json').read_text(encoding='utf-8'))
    for name,item in {**manifest['models'],**manifest['tokenizers']}.items():
        path=repo/item.get('checkpoint',item.get('path'))
        with path.open('rb') as handle:
            if handle.read(80).startswith(b'version https://git-lfs.github.com/spec/'):
                raise ValueError(f'{name}: LFS pointer only. Run git lfs pull first.')
            handle.seek(0);actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if actual!=item['sha256'] or ('bytes' in item and path.stat().st_size!=item['bytes']):
            raise ValueError('Checkpoint/tokenizer hash or size mismatch: '+name)
        print('Verified:',name)
    print('All selected artifacts verified. No model inference or training run.')


if __name__=='__main__':main()
