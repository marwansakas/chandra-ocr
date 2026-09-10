# Synthetic Arabic line data for TrOCR

The generator runs on CPU and writes the project's existing sample layout:

~~~text
thesis/datasets/synthetic_train_lines/
  synthetic_render_000001/
    image.png
    ground_truth.txt
    metadata.json
  manifest.jsonl
  run.json
  preview.jpg
~~~

There are two modes:

- **render:** render new Arabic sentences with fonts, then apply mild image
  variation. These are synthetic font images, not learned children's handwriting.
- **augment:** vary actual handwriting images from a real training manifest,
  keeping the original transcriptions. This preserves the children's writing style
  but does not create new words or new writers.

Neither mode edits your real images, labels, or test samples. A nonempty output
directory is rejected. Use a new directory for another experiment.

## Install on the cluster

If you use the GitHub checkout, pull the code and install the generator dependencies:

~~~bash
cd /home/sakas/chandra-ocr
git pull --ff-only origin main
python -m pip install -r requirements-synthetic.txt
~~~

The GitHub version already includes the TrOCR training integration; no patch is
needed. Alternatively, install the transfer ZIP as described below.

Copy the supplied synthetic-data-program.zip into /home/sakas/chandra-ocr
using VS Code's remote file explorer, then run:

~~~bash
cd /home/sakas/chandra-ocr
unzip -n synthetic-data-program.zip
python -m pip install -r requirements-synthetic.txt
~~~

The archive includes fonts and their original SIL Open Font License files.
No font download, GPU, API key, or cloud service is required during generation.
Use your existing TrOCR environment for training.

## Generate 1,000 new font-rendered samples

~~~bash
python thesis/scripts/generate_synthetic_lines.py \
  --mode render --count 1000 --seed 42 \
  --output-root thesis/datasets/synthetic_train_lines
~~~

This uses a small built-in demonstration corpus of elementary Arabic sentences.
It is not a curriculum dataset, and texts can repeat with different appearances.
For a better match to your task, supply your own UTF-8 corpus with one complete
line per image:

~~~bash
python thesis/scripts/generate_synthetic_lines.py \
  --text-file /path/to/training_text.txt \
  --count 1000 \
  --output-root thesis/datasets/synthetic_custom_lines
~~~

Text must use normal logical Unicode order. Do not reverse Arabic strings.
The renderer handles direction and shaping; labels stay in logical order.
Pillow RAQM is preferred. When RAQM is unavailable, reshaper/bidi is used and
only fonts covering the resulting presentation characters are selected
(the bundled Amiri font supports this fallback). Aref Ruqaa and Lateef normally
require RAQM. Metadata records the actual font and shaping engine.
Custom fonts can be supplied with --fonts-dir /path/to/fonts.
Unsupported characters and overlong lines are rejected rather than silently
losing characters or clipping text.

## Generate variations of your children's training handwriting

Use an existing real train.jsonl if you have a saved experiment. Never supply
the full dataset, test.jsonl, or validation samples as the source.
If this is a new experiment, prepare a new split directory first:

~~~bash
python thesis/scripts/train_line_ocr_model.py --prepare-only \
  --split-strategy student --seed 42 \
  --split-root thesis/datasets/synthetic_experiment_split

python thesis/scripts/generate_synthetic_lines.py \
  --mode augment \
  --train-manifest thesis/datasets/synthetic_experiment_split/train.jsonl \
  --count 1000 \
  --output-root thesis/datasets/synthetic_handwriting_lines
~~~

You can also pass --train-manifest with --mode render to render the training
transcriptions in new fonts. Input records must include sample_id, image_path
and text, matching the project's manifests. Relative image paths are resolved
against the project root. Samples cycle through the manifest in a seeded,
shuffled order.

Inspect preview.jpg and some full-size samples before training. Variation is
limited to small rotations, width scaling, contrast, slight blur and noise.
There are no flips, stroke erosion, or text crops.

## Enable generated data in your TrOCR trainers

The local checkout already includes this integration. When installing the ZIP
on the cluster, apply the included patch to the existing training scripts:

~~~bash
git apply --check synthetic-training-integration.patch &&
git apply synthetic-training-integration.patch
~~~

If the check fails, the cluster scripts differ or already contain the update.
Do not force the patch. Generation still works independently; inspect the
reported difference before integrating.

All three training entry points now accept --synthetic-root and the optional
--synthetic-limit. Generated data is added only after the real train/test split.
For example, run the following **inside your cluster GPU allocation**:

~~~bash
python thesis/scripts/train_line_ocr_model.py \
  --split-root thesis/datasets/synthetic_experiment_split \
  --split-strategy student --seed 42 \
  --synthetic-root thesis/datasets/synthetic_handwriting_lines \
  --synthetic-limit 300 \
  --output-dir thesis/models/trocr_with_synthetic \
  --epochs 5 --batch-size 4 --views-per-line 1
~~~

The 300-sample limit is an initial experiment, not an optimal mixing ratio.
Increase/decrease it based on real validation results.
Use train_cnn_trocr_model.py with a different output directory for CNN-TrOCR.
Use train_trocr_augmentation_models.py with separate --model-root,
--prediction-root, and --report for the six augmentation experiments.
Dictionary-assisted decoding can use a newly trained standard TrOCR checkpoint.

The loader checks completion, labels, sample IDs, and source provenance.
If an augmented source has moved to the held-out split or its image/transcription
changed, loading fails. Synthetic samples cannot be supplied as the real
--dataset-root.

The existing trainers still recompute real splits on each run and have no
separate validation partition. Keep real data, split strategy, ratio and seed
fixed across a comparison; changed data can change membership even with the
same seed. Do not select the best augmentation/epoch using the final test set.
Use fresh output directories for each experiment because existing predictors
can reuse cached predictions.

Use the same training budget and real evaluation samples for real-only versus
real-plus-synthetic comparisons. There is no guarantee of an accuracy gain.

## Provenance and verification

The generator records seeds, font hashes, transformation parameters, source IDs,
source image/label hashes (where applicable), and dataset completion. It writes
sample counts and distinct transcription counts to run.json. Given the same
inputs, fonts, shaping engine and dependency versions, generation is repeatable.
An interrupted dataset is marked incomplete and cannot be loaded for training.

Font origins, pinned upstream revision, and SHA-256 hashes are in
thesis/assets/synthetic_fonts/sources.json. The fonts come from the
[Google Fonts repository](https://github.com/google/fonts); original licenses
are included beside each font. Arabic layout follows
[Pillow's text-rendering API](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html).

Tests:

~~~bash
python -m pytest tests/test_synthetic_lines.py tests/test_pipeline_contracts.py -q
~~~
