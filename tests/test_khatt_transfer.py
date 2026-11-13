"""Offline two-stage contract tests; no real KHATT download, OCR run or GPU."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image
import torch

from thesis.diffusion.data import file_hash
from thesis.diffusion.tokenizer import CharacterTokenizer
from thesis.khatt_transfer import data as d, training as t
from thesis.khatt_transfer.__main__ import parse_args, main


def fixture_data(root, stage="khatt", tokenizer=None):
    root.mkdir()
    parts = {}
    for i, partition in enumerate(("train", "validation") + (("test",) if stage == "children" else ())):
        image = Image.new("RGB", (24, 32), (30+i*60, 40+i*30, 100))
        parts[partition] = [d.add_image(root, f"{stage}:{partition}", image, "اب")]
    d.freeze(root, parts, stage=stage, source={}, tokenizer=tokenizer)
    return root


def args_for(prepared, output, *extra):
    return parse_args(["train", "--prepared", str(prepared), "--output", str(output),
                       "--device", "cpu", "--epochs", "1", "--patience", "3",
                       "--batch-size", "1", "--accumulation", "1", "--dim", "8",
                       "--layers", "1", "--capacity", "8", "--max-width", "64",
                       "--freeze-epochs", "0", "--warmup-epochs", "1", "--cpu-threads", "1", *extra])


def fake_parent(path):
    tokenizer = CharacterTokenizer.build(["اب"])
    value = dict(version=1, arm="mdiff", stage="khatt", model={"a": torch.tensor([1.])},
                 tokenizer=tokenizer.as_dict(), config=dict(capacity=16, dim=8, layers=1, dropout=.1, max_width=128),
                 seen_pixels=[])
    t.save_checkpoint(path, value)
    return path


def test_khatt_download_pins_revision_and_reverses_only_mirror(monkeypatch, tmp_path):
    calls = []
    def load(name, **kw):
        calls.append((name, kw))
        color = "red" if kw["split"] == "train" else "blue"
        return [dict(image=Image.new("RGB", (24, 32), color), text="با")]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load, load_from_disk=None))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=lambda *a, **kw: SimpleNamespace(sha="pinned-commit"))))
    main(["prepare-khatt", "--output", str(tmp_path/"prepared")])
    meta, tok, parts = d.load_prepared(tmp_path/"prepared")
    assert meta["source"]["revision"] == "pinned-commit"
    assert parts["train"][0]["source_text"] == "اب"
    assert parts["train"][0]["original_mirror_text"] == "با"
    assert {c[1]["split"] for c in calls} == {"train", "validation"}
    assert all(c[1]["revision"] == "pinned-commit" for c in calls)


def test_saved_hf_splits_can_be_reused_without_hub(monkeypatch, tmp_path):
    def load_disk(path):
        return [dict(image=Image.new("RGB", (24, 32), "red" if path.endswith("train") else "blue"), text="با")]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_from_disk=load_disk, load_dataset=None))
    main(["prepare-khatt", "--hf-root", str(tmp_path), "--output", str(tmp_path/"p")])
    assert d.load_prepared(tmp_path/"p")[2]["validation"][0]["text"] == "اب"


def test_manifest_and_image_tampering_are_rejected(tmp_path):
    root = fixture_data(tmp_path/"p")
    (root/"validation.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="Manifest hash"):
        d.load_prepared(root)
    root2 = fixture_data(tmp_path/"p2")
    next((root2/"images").iterdir()).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Image path/hash"):
        d.load_prepared(root2)


def test_duplicate_pixels_cannot_cross_supplied_splits(tmp_path):
    root=tmp_path/"p"; root.mkdir()
    image=Image.new("RGB", (24,32), "red")
    parts={p:[d.add_image(root,p,image,"اب")] for p in ("train","validation")}
    with pytest.raises(ValueError, match="Duplicate image"):
        d.freeze(root,parts,stage="khatt",source={})


def test_group_split_joins_duplicate_images_and_writer_ids():
    rows=[dict(sample_id=str(i),pixels_sha256=str(i),text="اب") for i in range(12)]
    rows[1]["pixels_sha256"]="0"
    writers={str(i):str(i//2) for i in range(12)}
    parts=d.split_children(rows,None,writers,42,.2,.2)
    assigned={r["sample_id"]:p for p,rs in parts.items() for r in rs}
    assert assigned["0"]==assigned["1"]
    for i in range(0,12,2): assert assigned[str(i)]==assigned[str(i+1)]
    assert all(parts.values())
    assert sum(map(len,parts.values()))==12


def test_existing_assignments_preserved_and_group_overlap_rejected():
    rows=[dict(sample_id=str(i),pixels_sha256=str(i),text="اب") for i in range(6)]
    assignments={str(i):("train","validation","test")[i//2] for i in range(6)}
    parts=d.split_children(rows,"line",None,42,.1,.2,assignments)
    assert {r["sample_id"]:p for p,rs in parts.items() for r in rs}==assignments
    rows[2]["pixels_sha256"]=rows[0]["pixels_sha256"]
    with pytest.raises(ValueError,match="Existing assignments split"):
        d.split_children(rows,"line",None,42,.1,.2,assignments)


def test_notebook_grouping_is_explicit_and_rejects_unknown_names():
    assert d.notebook_group("000001_0001_School_Notebook_5_Page_030_line_001")=="School_Notebook_5"
    assert d.notebook_group("000001_0001_0001_School_Notebook_5_Page_030_line_001")=="School_Notebook_5"
    with pytest.raises(ValueError,match="Cannot identify notebook"):
        d.notebook_group("unknown")


def test_children_prepare_inherits_tokenizer_and_preserves_source_labels(tmp_path):
    parent=fake_parent(tmp_path/"parent.pt")
    dataset=tmp_path/"children";dataset.mkdir()
    for i in range(8):
        folder=dataset/str(i);folder.mkdir()
        Image.new("RGB",(24,32),(10+i*20,30,70)).save(folder/"image.png")
        (folder/"ground_truth.txt").write_text(" اَب \n",encoding="utf-8")
    main(["prepare-children","--dataset-root",str(dataset),"--init-from",str(parent),
          "--split-strategy","line","--output",str(tmp_path/"p")])
    meta,tok,parts=d.load_prepared(tmp_path/"p",("train","validation","test"))
    assert tok.as_dict()==t.load_checkpoint(parent)["tokenizer"]
    assert sum(map(len,parts.values()))==8
    assert next(iter(parts["train"]))["text"]==" اب "
    assert (dataset/"0/ground_truth.txt").read_text(encoding="utf-8")==" اَب \n"
    assert meta["source"]["init_checkpoint_sha256"]==file_hash(parent)


def test_preflight_rejects_unknown_overflow_and_width_without_truncating():
    tok=CharacterTokenizer.build(["اب"])
    config=dict(capacity=4,max_width=64)
    base=dict(sample_id="sample",text="اب",width=24,height=32)
    for changed in (dict(text="🦒"),dict(text="اباب"),dict(width=200)):
        with pytest.raises(ValueError,match="sample"):
            t.check_records([{**base,**changed}],tok,config)


def test_checkpoint_tampering_and_vocabulary_drift_are_rejected(tmp_path):
    path=fake_parent(tmp_path/"parent.pt")
    saved=torch.load(path,weights_only=True);saved["model"]["a"][0]=2
    torch.save(saved,path)
    with pytest.raises(ValueError,match="hash mismatch"): t.load_checkpoint(path)
    parent=dict(config=dict(capacity=4,dim=8,layers=1,dropout=.1,max_width=64),tokenizer=CharacterTokenizer.build(["اب"]).as_dict())
    args=SimpleNamespace(capacity=None,dim=None,layers=None,dropout=None,max_width=None)
    with pytest.raises(ValueError,match="tokenizer differs"):
        t.build_config(args,CharacterTokenizer.build(["🦒"]),{},parent)


def test_test_reference_diagnostics_do_not_enter_generation(tmp_path):
    tokenizer=CharacterTokenizer.build(["اب"])
    image=Image.new("RGB",(24,32),"red")
    row=d.add_image(tmp_path,"test",image,"🦒"*20)
    class ImageOnly:
        def eval(self): pass
        def generate(self,pixels,widths,steps):
            assert pixels.shape==(1,3,32,24) and widths.tolist()==[24]
            return torch.tensor([[tokenizer.EOS]*4]),dict(decoder_passes=6)
    scores,predictions=t.evaluate_rows(ImageOnly(),tmp_path,[row],tokenizer,
                            dict(capacity=4,max_width=64),torch.device("cpu"),3)
    assert scores["expected"]==scores["completed"]==1
    assert scores["reference_oov_lines"]==scores["reference_over_capacity_lines"]==1
    assert scores["deletions"]==20


def test_asset_download_is_pinned_and_tampering_fails(tmp_path,monkeypatch):
    import hashlib
    from thesis.mdiff_arabic_v1 import artifacts
    from thesis.khatt_transfer.__main__ import setup_assets
    payload=b"synthetic asset"
    monkeypatch.setattr(artifacts,"ASSETS",{"best.pth":hashlib.sha256(payload).hexdigest()})
    urls=[]
    def retrieve(url,path):
        urls.append(url);path.write_bytes(payload)
    monkeypatch.setattr("urllib.request.urlretrieve",retrieve)
    setup_assets(tmp_path)
    setup_assets(tmp_path)
    assert len(urls)==1 and artifacts.HF_REVISION in urls[0]
    (tmp_path/"best.pth").write_bytes(b"changed")
    with pytest.raises(ValueError,match="asset hash mismatch"):setup_assets(tmp_path)


def test_cluster_shell_routes_stages_and_resume_without_running_training(tmp_path):
    git_bash=Path(r"C:\Program Files\Git\bin\bash.exe")
    bash=str(git_bash) if git_bash.exists() else shutil.which("bash")
    if not bash:pytest.skip("Bash is unavailable")
    script=Path(__file__).resolve().parents[1]/"thesis/scripts/run_khatt_transfer.sh"
    subprocess.run([bash,"-n",str(script)],check=True,capture_output=True)
    fake=tmp_path/"fake-python.sh"
    fake.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$CAPTURE"\n',encoding="utf-8",newline="\n")
    fake.chmod(0o755)
    env={**os.environ,"TRAIN_PYTHON":fake.as_posix(),"CAPTURE":(tmp_path/"args.txt").as_posix(),
         "KHATT_TRANSFER_ROOT":(tmp_path/"artifacts with spaces").as_posix()}
    for phase in ("pretrain","finetune"):
        result=subprocess.run([bash,str(script),phase,"--resume"],env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        argv=(tmp_path/"args.txt").read_text().splitlines()
        assert argv[:3]==["-m","thesis.khatt_transfer","train"]
        assert argv[-1]=="--resume"
        if phase=="finetune":
            assert "--init-from" in argv and any(p.endswith("khatt_run/best.pt") for p in argv)
            assert argv[argv.index("--decoder-lr")+1]=="3e-5"
    invalid=subprocess.run([bash,str(script),"prepare-khatt","--resume"],env=env,capture_output=True)
    assert invalid.returncode!=0


def test_real_mdiff_two_stages_and_explicit_test(tmp_path):
    pre=fixture_data(tmp_path/"khatt")
    first=args_for(pre,tmp_path/"pretrain","--random-init")
    t.train(first)
    parent=t.load_checkpoint(first.output/"best.pt")
    child=fixture_data(tmp_path/"children",stage="children",tokenizer=CharacterTokenizer(parent["tokenizer"]["characters"]))
    # Different images from KHATT, with a recorded parent identity.
    meta=d.read_json(child/"prepared.json")
    for partition in ("train","validation","test"):
        rows=[json.loads(x) for x in (child/(partition+".jsonl")).read_text(encoding="utf-8").splitlines()]
        for row in rows:
            image=Image.open(child/row["image_path"]).convert("RGB")
            image.putpixel((0,0),(240,240,240));image.save(child/row["image_path"])
            row.update(image_sha256=file_hash(child/row["image_path"]),pixels_sha256=d.pixel_hash(image))
        (child/(partition+".jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
        meta["hashes"][partition+".jsonl"]=file_hash(child/(partition+".jsonl"))
    meta["source"]["init_checkpoint_sha256"]=file_hash(first.output/"best.pt")
    d.write_json(child/"prepared.json",meta)
    second=args_for(child,tmp_path/"finetune","--init-from",str(first.output/"best.pt"))
    # Train must not read or validate the test manifest at all.
    test_bytes=(child/"test.jsonl").read_bytes()
    (child/"test.jsonl").write_bytes(b"test references inaccessible during training")
    t.train(second)
    run=d.read_json(second.output/"run.json")
    assert run["optimizer_reset"] and run["parent_stage"]=="khatt"
    assert run["binding"]["options"]["decoder_lr"]==3e-5
    assert t.load_checkpoint(second.output/"last.pt")["epoch"]==1
    last=t.load_checkpoint(second.output/"last.pt")
    assert last["lineage"][0]["checkpoint_sha256"]==file_hash(first.output/"best.pt")
    assert max(int(s["step"]) for s in last["optimizer"]["state"].values())==1
    assert not (second.output/"test.json").exists()
    (child/"test.jsonl").write_bytes(test_bytes)
    main(["evaluate","--prepared",str(child),"--checkpoint",str(second.output/"best.pt"),
          "--partition","test","--output",str(tmp_path/"report.json"),"--device","cpu"])
    report=d.read_json(tmp_path/"report.json")
    assert report["metrics"]["expected"]==report["metrics"]["completed"]==1
    assert report["predictions"][0]["decoder_passes"]==6
    second.resume=True
    original=file_hash(second.output/"last.pt")
    t.train(second)
    assert original==file_hash(second.output/"last.pt")
    second.epochs=2
    with pytest.raises(ValueError,match="options.epochs"):
        t.train(second)


def test_interruption_after_training_resumes_without_repeating_epoch(tmp_path,monkeypatch):
    pre=fixture_data(tmp_path/"p")
    args=args_for(pre,tmp_path/"interrupted","--random-init")
    original=t.evaluate_rows;calls=0
    def fail_validation(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError("synthetic validation interruption")
        return original(*a,**kw)
    monkeypatch.setattr(t,"evaluate_rows",fail_validation)
    with pytest.raises(RuntimeError,match="interruption"): t.train(args)
    trained=t.load_checkpoint(args.output/"trained.pt")["model"]
    monkeypatch.setattr(t,"evaluate_rows",original)
    args.resume=True
    t.train(args)
    resumed=t.load_checkpoint(args.output/"last.pt")
    assert all(torch.equal(v,resumed["model"][k]) for k,v in trained.items())
    assert not (args.output/"trained.pt").exists()
    whole=args_for(pre,tmp_path/"whole","--random-init")
    t.train(whole)
    uninterrupted=t.load_checkpoint(whole.output/"last.pt")
    assert all(torch.equal(v,uninterrupted["model"][k]) for k,v in resumed["model"].items())
    assert resumed["history"]==uninterrupted["history"]
    assert torch.equal(resumed["rng"]["torch"],uninterrupted["rng"]["torch"])


def test_stopping_epoch_cannot_extend_after_completion_write_crash(tmp_path,monkeypatch):
    pre=fixture_data(tmp_path/"p")
    args=args_for(pre,tmp_path/"run","--random-init")
    args.epochs=3;args.patience=1
    original_eval=t.evaluate_rows
    def constant_metrics(*a,**kw):
        metrics,predictions=original_eval(*a,**kw)
        metrics.update(char_distance=2,word_distance=1,exact_lines=0)
        return metrics,predictions
    monkeypatch.setattr(t,"evaluate_rows",constant_metrics)
    original_write=t.atomic_json
    def interrupt_complete(path,value):
        if path.name=="complete.json": raise RuntimeError("completion interrupted")
        return original_write(path,value)
    monkeypatch.setattr(t,"atomic_json",interrupt_complete)
    with pytest.raises(RuntimeError,match="completion interrupted"):t.train(args)
    state=t.load_checkpoint(args.output/"last.pt")
    assert state["epoch"]==1 and state["stale"]==1
    monkeypatch.setattr(t,"atomic_json",original_write)
    args.resume=True
    t.train(args)
    assert d.read_json(args.output/"complete.json")["epoch"]==1
    final=t.load_checkpoint(args.output/"last.pt")
    assert all(torch.equal(v,final["model"][k]) for k,v in state["model"].items())


@pytest.mark.parametrize("extra",[["--dim","7"],["--max-width","65"],["--dropout","1"],["--decoder-lr","0"]])
def test_cli_rejects_invalid_training_options(extra):
    with pytest.raises(SystemExit): parse_args(["train","--prepared","p","--output","o",*extra])
