#!/usr/bin/env bash
# Run one named phase; all paths are initialized afresh for every invocation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$PWD"
RUN="${OCR_RUN:-round_20260913}"
[[ "$RUN" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid OCR_RUN name'; exit 1; }
DATA="$ROOT/thesis/datasets/ground_truth_lines"
SPLIT="$ROOT/thesis/datasets/ground_truth_lines_ml"
SYNTH="$ROOT/thesis/datasets/synthetic_$RUN"
MODELS="$ROOT/thesis/models/$RUN"
PRED="$ROOT/thesis/predictions/$RUN"
REPORT="$ROOT/thesis/reports/$RUN"
PY="${TRAIN_PYTHON:-python}"
COMMON=(--dataset-root "$DATA" --split-root "$SPLIT" --split-strategy student --train-ratio 0.8 --seed 42)

require_gpu() {
  "$PY" -c "import torch; assert torch.cuda.is_available(), 'Use your training environment inside a Slurm GPU allocation'"
}

check_split() {
  [[ -s "$SPLIT/train.jsonl" && -s "$SPLIT/test.jsonl" ]] || { echo 'Run the split phase first'; exit 1; }
  [[ -f "$REPORT/data.sha256" ]] || { echo 'Run the split phase first'; exit 1; }
  # Refuse additions, removals, or content changes during this experiment.
  local current
  current="$(mktemp)"
  dataset_hashes > "$current"
  if ! cmp -s "$REPORT/data.sha256" "$current"; then
    rm -f "$current"
    echo 'Dataset changed. Start a new experiment and regenerate synthetic data.'
    exit 1
  fi
  rm -f "$current"
  sha256sum --check --status "$REPORT/split.sha256" || { echo 'Split changed. Start a new experiment.'; exit 1; }
}

dataset_hashes() {
  find "$DATA" -type f \( -name image.png -o -name ground_truth.txt -o -name metadata.json \) -print0 |
    sort -z | xargs -0 -r sha256sum
}

case "${1:-help}" in
  download)
    mkdir -p "$DATA"
    listing="$(mktemp)"
    trap 'rm -f "$listing"' EXIT
    rclone lsf 'gdrive:Arabic_Children_OCR/ground_truth_lines' --dirs-only > "$listing"
    while IFS= read -r folder; do
      folder="${folder%/}"
      [[ -n "$folder" && "$folder" != '.' && "$folder" != '..' && "$folder" != */* ]] || exit 1
      [[ -d "$DATA/$folder" ]] && continue
      # Only expose the new folder after its entire copy succeeds.
      staging="$(mktemp -d "$ROOT/thesis/datasets/.line-download-XXXXXX")"
      rclone copy "gdrive:Arabic_Children_OCR/ground_truth_lines/$folder" "$staging" --progress
      mv "$staging" "$DATA/$folder"
    done < "$listing"
    ;;
  split)
    [[ ! -e "$REPORT/split.sha256" && ! -d "$MODELS" && ! -d "$PRED" ]] || {
      echo 'This experiment already exists. Reuse it, or choose a new OCR_RUN name.'; exit 1;
    }
    "$PY" thesis/scripts/train_line_ocr_model.py "${COMMON[@]}" --prepare-only
    mkdir -p "$REPORT"
    dataset_hashes > "$REPORT/data.sha256"
    sha256sum "$SPLIT/train.jsonl" "$SPLIT/test.jsonl" > "$REPORT/split.sha256"
    cp "$SPLIT/train.jsonl" "$SPLIT/test.jsonl" "$REPORT/"
    wc -l "$SPLIT/train.jsonl" "$SPLIT/test.jsonl"
    ;;
  synthetic)
    check_split
    "$PY" thesis/scripts/generate_synthetic_lines.py --mode augment --train-manifest "$SPLIT/train.jsonl" --count 1000 --output-root "$SYNTH"
    ;;
  train-real|train-synthetic)
    check_split
    require_gpu
    variant=real_only
    extra=()
    if [[ "$1" == train-synthetic ]]; then
      variant=real_plus_synthetic
      extra=(--synthetic-root "$SYNTH")
    fi
    for model in trocr cnn_trocr; do
      script=thesis/scripts/train_line_ocr_model.py
      [[ "$model" != cnn_trocr ]] || script=thesis/scripts/train_cnn_trocr_model.py
      [[ ! -e "$MODELS/${model}_${variant}/config.json" ]] || { echo 'Model already exists; choose a new experiment to retrain'; exit 1; }
      "$PY" "$script" "${COMMON[@]}" "${extra[@]}" --output-dir "$MODELS/${model}_${variant}" --epochs 5 --batch-size 4 --views-per-line 1
    done
    ;;
  train-variants|train-variants-synthetic)
    check_split
    require_gpu
    variant=real
    extra=()
    if [[ "$1" == train-variants-synthetic ]]; then
      variant=synthetic
      extra=(--synthetic-root "$SYNTH")
    fi
    [[ ! -d "$MODELS/augmentation_$variant" ]] || { echo 'Augmentation experiment already exists'; exit 1; }
    "$PY" thesis/scripts/train_trocr_augmentation_models.py "${COMMON[@]}" "${extra[@]}" \
      --profiles none,light,scan,geometry,stroke,mixed --epochs 5 --batch-size 4 --views-per-line 3 \
      --model-root "$MODELS/augmentation_$variant" --prediction-root "$PRED/augmentation_$variant" \
      --report "$REPORT/augmentation_$variant.csv" --device cuda
    ;;
  test-local)
    check_split
    require_gpu
    for variant in real_only real_plus_synthetic; do
      for model in trocr cnn_trocr; do
        script=thesis/scripts/predict_trocr_test_samples.py
        [[ "$model" != cnn_trocr ]] || script=thesis/scripts/predict_cnn_trocr_test_samples.py
        "$PY" "$script" --test-manifest "$SPLIT/test.jsonl" --model "$MODELS/${model}_${variant}" \
          --output-root "$PRED/${model}_${variant}" --device cuda --batch-size 4 --force
      done
    done
    "$PY" thesis/scripts/download_general_arabic_dictionary.py
    for variant in real_only real_plus_synthetic; do
      "$PY" pipelines/general_arabic_trocr_run.py --test-manifest "$SPLIT/test.jsonl" \
        --model "$MODELS/trocr_$variant" --output-root "$PRED/dictionary_$variant" --device cuda --force
    done
    ;;
  test-pretrained)
    check_split
    for model in got-ocr2 glm-ocr hunyuan-ocr lighton-ocr2; do
      .venv-ocr-hf/bin/python -m pretrained_ocr run --model "$model" --dataset "$DATA" \
        --test-manifest "$SPLIT/test.jsonl" --output "$PRED/pretrained/$model" --device cuda --normalization arabic-light
    done
    .venv-ocr-paddle/bin/python -m pretrained_ocr run --model paddle-arabic --dataset "$DATA" \
      --test-manifest "$SPLIT/test.jsonl" --output "$PRED/pretrained/paddle-arabic" --device cpu --normalization arabic-light
    ;;
  test-chandra)
    check_split
    "$PY" pipelines/chandra.py --input-root "$DATA" --test-manifest "$SPLIT/test.jsonl" \
      --output-root "$PRED/chandra" --chandra /home/sakas/.conda/envs/chandra/bin/chandra --method hf --force
    ;;
  setup-invizo)
    python3 -m venv .venv-ocr-invizo
    .venv-ocr-invizo/bin/python -m pip install numpy==1.26.4 pillow==10.3.0 opencv-python-headless==4.9.0.80 onnxruntime==1.20.1
    mkdir -p thesis/invizo/vendor thesis/invizo/models
    for file in OCRModule.py OCRConfig.py; do
      curl -fL "https://raw.githubusercontent.com/Hedrax/Invizo-OCR/2b6f416b6218adc87de06c941382b2f708749090/src/main/python/$file" -o "thesis/invizo/vendor/$file"
    done
    curl -fL 'https://github.com/Hedrax/Invizo-OCR/releases/download/v0.2-alpha/recognition-CNN-TR-1.0.onnx' \
      -o thesis/invizo/models/recognition-CNN-TR-1.0.onnx
    ;;
  test-invizo)
    check_split
    .venv-ocr-invizo/bin/python pipelines/invizo_onnx_run.py --test-manifest "$SPLIT/test.jsonl" \
      --model thesis/invizo/models/recognition-CNN-TR-1.0.onnx --upstream-dir thesis/invizo/vendor \
      --output-root "$PRED/invizo" --force
    ;;
  metrics)
    check_split
    predictions=()
    for name in trocr_real_only trocr_real_plus_synthetic cnn_trocr_real_only cnn_trocr_real_plus_synthetic dictionary_real_only dictionary_real_plus_synthetic chandra invizo; do
      predictions+=(--prediction "$name=$PRED/$name")
    done
    for model in got-ocr2 glm-ocr hunyuan-ocr lighton-ocr2 paddle-arabic; do
      predictions+=(--prediction "$model=$PRED/pretrained/$model/samples")
    done
    for variant in real synthetic; do
      for profile in none light scan geometry stroke mixed; do
        predictions+=(--prediction "augmentation_${variant}_$profile=$PRED/augmentation_$variant/$profile")
      done
    done
    "$PY" thesis/scripts/evaluate_all_line_models.py --test-manifest "$SPLIT/test.jsonl" \
      "${predictions[@]}" --output-dir "$REPORT/comparison"
    ;;
  *)
    echo 'Phases: download split synthetic train-real train-synthetic train-variants train-variants-synthetic test-local test-pretrained setup-invizo test-invizo test-chandra metrics'
    ;;
esac
