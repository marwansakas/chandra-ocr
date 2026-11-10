"""Audited KHATT transcription conversion and immutable, nested TRAIN subsets."""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

from PIL import Image

from thesis.diffusion.data import image_info
from thesis.diffusion.tokenizer import ARABIC_INVENTORY, CharacterTokenizer, read_literal, training_text
from thesis.mdiff_arabic_v1.artifacts import canonical, digest, file_hash, jsonl, read, write

SEED = 54101
# Literal line breaks and arithmetic inequality are supported explicitly; the
# three standalone ≠ non-transcription labels remain excluded as before.
DECLARED_INVENTORY = ARABIC_INVENTORY + '\r\n\t≠'
TRIALS = {'khatt100_only': (100, False), 'khatt100_ours': (100, True),
          'khatt50_ours': (50, True), 'khatt30_ours': (30, True), 'khatt10_ours': (10, True)}
# Official KHATT readme, Arabic-Latin lookup, pp.5-6:
# https://catalog.ldc.upenn.edu/docs/LDC2015T23/readme_KHATT_v1.0.pdf
CODE_MAP = dict(zip(
    'hh am ae ah aa ba ta teE th ja ha kh da dh ra za se sh sa de to zha ay gh fa ka ke la ma na he wa wl ya ee al'.split(),
    'ءآأإابتةثجحخدذرزسشصضطظعغفقكلمنهوؤيىئ'))
CODE_MAP.update({f'n{i}': str(i) for i in range(10)})
CODE_MAP.update({'atr':'@', 'col':':', 'dbq':'"', 'com':',', 'qts':'?', 'exc':'!',
    'dot':'.', 'bro':'(', 'brc':')', 'fsl':'/', 'bsl':'\\', 'equ':'=', 'hyp':'-',
    'usc':'_', 'scr':'#', 'per':'%', 'sp':' '})
# Mirror-specific case alias and lam-alif compounds. Compounds are two logical
# codepoints, never Arabic presentation forms. Cross-checked against source
# images including AHTD3A0085_Para2_5 (laam), AHTD3A0104_Para2_1 (laah).
CODE_MAP.update({'tee':'ة', 'laaa':'لا', 'laae':'لأ', 'laah':'لإ', 'laam':'لآ'})


def decode_cells(cells):
    nonempty = [x for x in cells if x != '']
    delimiter = bool(nonempty and nonempty[-1] == ';')
    if delimiter:
        nonempty.pop()
    if ';' in nonempty:
        raise ValueError('KHATT structural semicolon is only allowed at the end')
    unknown = sorted(set(nonempty) - CODE_MAP.keys())
    if unknown:
        raise ValueError('Unknown KHATT codes: ' + repr(unknown))
    text = ''.join(CODE_MAP[x] for x in nonempty)
    if not text.strip():
        raise ValueError('Empty KHATT transcription')
    return text, delimiter


def nested_subsets(rows, seed=SEED):
    ranked = sorted(rows, key=lambda r: (hashlib.sha256(f'{seed}\0{r["sample_id"]}'.encode()).hexdigest(), r['sample_id']))
    return {p: ranked[:math.floor(len(rows)*p/100)] for p in (10,30,50,100)}


def child_row(row, root):
    row = dict(row)
    image, label = root/row['image_path'], root/row['label_path']
    if file_hash(image) != row['image_sha256'] or file_hash(label) != row['label_sha256']:
        raise ValueError('Child image/label changed: '+row['sample_id'])
    if read_literal(label) != row['text']:
        raise ValueError('Child literal label mismatch')
    return dict(row, image_path=str(image.resolve()), label_path=str(label.resolve()), source='children',
                text=training_text(row['text'], 'no_harakat_v1'))


def prepare(args):
    folder, drive = args.suite, args.drive
    if (folder/'prepared.json').exists():
        verify(args)
        print('Existing prepared suite verified:', folder)
        return
    folder.mkdir(parents=True, exist_ok=True)
    split, child = drive/'splits_70_05_25_v3', drive/'ground_truth_lines'
    lock = read(split/'split_lock.json')
    inputs = {str(split/'split_lock.json'):file_hash(split/'split_lock.json')}
    parts, all_ids, held_pixels = {}, set(), set()
    for part, count in (('train',2103),('validation',150),('test',752)):
        p = split/(part+'.jsonl')
        if file_hash(p) != lock['hashes'][p.name]:
            raise ValueError('Canonical split changed: '+part)
        inputs[str(p)] = file_hash(p)
        rows = jsonl(p)
        if len(rows)!=count or len({r['sample_id'] for r in rows})!=count:
            raise ValueError('Canonical count/duplicates: '+part)
        for r in rows:
            if r['sample_id'] in all_ids:
                raise ValueError('Cross-split ID overlap')
            all_ids.add(r['sample_id'])
        if part == 'test':
            # Training preparation retains identity only, never test references.
            parts[part] = [{k:r[k] for k in ('sample_id','image_sha256','pixels_sha256','image_path')} for r in rows]
        else:
            parts[part] = [child_row(r,child) for r in rows]
        if part != 'train':
            held_pixels.update(r['pixels_sha256'] for r in rows)
    new, excluded = [], []
    for p in sorted(child.iterdir()):
        if not p.is_dir() or p.name in all_ids:
            continue
        image, label = p/'image.png', p/'ground_truth.txt'
        if not image.exists() or not label.exists():
            raise ValueError('Incomplete child pair: '+str(p))
        literal = read_literal(label)
        if literal == '≠':
            excluded.append(dict(sample_id=p.name, reason='standalone non-transcription marker ≠'))
            continue
        info = image_info(image)
        if info['pixels_sha256'] in held_pixels:
            raise ValueError('New child image duplicates validation/test pixels: '+p.name)
        with Image.open(image) as im:
            width,height=im.size
        new.append(dict(sample_id=p.name,image_path=str(image.resolve()),label_path=str(label.resolve()),
            image_sha256=file_hash(image),label_sha256=file_hash(label),pixels_sha256=info['pixels_sha256'],
            text=training_text(literal,'no_harakat_v1'),source='children',width=width,height=height))
    if len(new)!=704:
        raise ValueError(f'Expected 704 newly synced child lines, found {len(new)}; use a new reviewed suite for changed data')
    children = parts['train']+new
    if len(children)!=2807 or any(r['pixels_sha256'] in held_pixels for r in children):
        raise ValueError('Child TRAIN membership/held-out image overlap')
    khatt = args.project/'Diffusion/datasets/KHATT/data'
    csvpath = khatt/'Train.csv'
    inputs[str(csvpath)] = file_hash(csvpath)
    images = {}
    for p in (khatt/'Train_deskewed').rglob('*.jpg'):
        if p.stem in images:
            raise ValueError('Ambiguous KHATT image stem')
        images[p.stem]=p
    adults, used, delimiters = [], set(), 0
    with csvpath.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f); next(reader)
        for number,cells in enumerate(reader,2):
            stem=Path(cells[0]).stem
            if stem in used or stem not in images:
                raise ValueError('Duplicate/missing KHATT image: '+stem)
            used.add(stem);text, delimiter=decode_cells(cells[1:]);delimiters+=delimiter
            p=images[stem]
            with Image.open(p) as im:
                width,height=im.size
            adults.append(dict(sample_id='khatt:'+stem,image_path=str(p.resolve()),image_sha256=file_hash(p),
                text=text,source='khatt',width=width,height=height,csv_row=number))
    if len(adults)!=9497:
        raise ValueError('Expected all 9497 annotated KHATT Train.csv lines')
    # Declared inventory, independent of children validation/test or training
    # references: identical tokenizer/head initialization across all five trials.
    tokenizer=CharacterTokenizer(sorted(set(DECLARED_INVENTORY+''.join(CODE_MAP.values())),key=ord))
    all_train=adults+children
    for r in all_train+parts['validation']:
        tokenizer.encode(r['text'],144,eos_fill=False)
    max_width=max(round(r['width']*32/r['height']) for r in all_train)
    width_limit=math.ceil(max_width*1.05/4)*4
    width_limit=max(640,width_limit)
    subsets=nested_subsets(adults)
    files={}
    def emit(name,rows):
        p=folder/name
        # A prior interrupted preparation is replayable, never partially reused.
        p.write_bytes(b''.join(canonical(r) for r in rows))
        files[name]=file_hash(p)
    emit('validation.jsonl',parts['validation'])
    emit('test_identity.jsonl',parts['test'])
    emit('children_train.jsonl',children)
    emit('khatt_train.jsonl',adults)
    summaries={}
    for name,(percent,ours) in TRIALS.items():
        rows=subsets[percent]+(children if ours else [])
        emit(name+'.jsonl',rows)
        summaries[name]=dict(khatt=len(subsets[percent]),children=len(children) if ours else 0,total=len(rows),validation=150)
    tokenizer.save(folder/'tokenizer.json');files['tokenizer.json']=file_hash(folder/'tokenizer.json')
    write(folder/'prepared.json',dict(version='khatt_ablation_v1',seed=SEED,inputs=inputs,files=files,
        counts=summaries,child_new_lines=len(new),excluded_children=excluded,
        khatt_unannotated_train_images=sorted(set(images)-used),khatt_validation_used=False,
        khatt_terminal_csv_delimiters=delimiters,khatt_mapping_sha256=digest(CODE_MAP),
        vocabulary_origin='shared declared Arabic/Latin/digit/punctuation inventory plus literal CR/LF/TAB/≠; no validation/test-derived additions',
        image_config=dict(height=32,max_width=width_limit,patch_size=4),capacity=144,
        maximum_train_resized_width=max_width,maximum_train_characters=max(len(r['text']) for r in all_train),
        percentage_definition='floor(percent * 9497 / 100), fixed nested seeded line subsets; every selected line once per epoch',
        child_validation_count=150,historical_test_count=752,test_references_used=False,
        child_data_note='All 704 new usable lines assigned TRAIN. Existing 2103/150/752 assignments unchanged. No fresh test claim.'))
    verify(args)
    print('Prepared five AR trials; no training or OCR evaluation run.')
    for name,counts in summaries.items():print(name,counts)


def verify(args):
    prepared=read(args.suite/'prepared.json')
    if prepared['version']!='khatt_ablation_v1' or prepared['khatt_mapping_sha256']!=digest(CODE_MAP):
        raise ValueError('Preparation version/mapping mismatch')
    for p,sha in prepared['inputs'].items():
        if file_hash(Path(p))!=sha:raise ValueError('Prepared input changed: '+p)
    for name,sha in prepared['files'].items():
        if file_hash(args.suite/name)!=sha:raise ValueError('Prepared artifact changed: '+name)
    return prepared
