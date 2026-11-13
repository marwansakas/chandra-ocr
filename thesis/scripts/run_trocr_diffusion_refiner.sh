#!/usr/bin/env bash
# Run inside an existing GPU allocation. No partition/account names are assumed.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

: "${DATASET_ROOT:?Set DATASET_ROOT to the real image.png/ground_truth.txt dataset}"
: "${SPLIT_ROOT:?Set SPLIT_ROOT to an existing frozen train/validation/test split}"
: "${CACHE_ROOT:?Set CACHE_ROOT to a dedicated feature cache directory}"
: "${RUN_ROOT:?Set RUN_ROOT to a NEW model run directory}"
python_bin="${PYTHON:-python}"
device="${DEVICE:-cuda}"
base_model="${BASE_MODEL:-microsoft/trocr-base-handwritten}"
report_root="${REPORT_ROOT:-${RUN_ROOT}_test}"

if [[ -e "$RUN_ROOT" || -e "$report_root" ]]; then
    printf '%s\n' 'RUN_ROOT and REPORT_ROOT must be new directories; existing runs are preserved.' >&2
    exit 1
fi
if [[ "$device" == cuda ]]; then
    "$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: activate the GPU environment inside an allocation"; print(torch.cuda.get_device_name(0))'
fi

base_options=(--base-model "$base_model")
if [[ -n "${BASE_TRAINING_MANIFEST:-}" ]]; then
    base_options+=(--base-training-manifest "$BASE_TRAINING_MANIFEST")
elif [[ "$base_model" == microsoft/trocr-base-handwritten || "${EXTERNAL_BASE:-0}" == 1 ]]; then
    base_options+=(--external-base)
else
    printf '%s\n' 'For a project-fine-tuned base, supply its actual BASE_TRAINING_MANIFEST. Set EXTERNAL_BASE=1 only for a base never trained on project data.' >&2
    exit 1
fi
if [[ -n "${BASE_REVISION:-}" ]]; then
    base_options+=(--base-revision "$BASE_REVISION")
fi
common=(--dataset-root "$DATASET_ROOT" --split-root "$SPLIT_ROOT" --cache-root "$CACHE_ROOT" --device "$device")
cache_options=("${base_options[@]}" --max-new-tokens "${MAX_NEW_TOKENS:-128}" --batch-size "${CACHE_BATCH_SIZE:-4}" --resume)

"$python_bin" -m thesis.refiner cache "${common[@]}" "${cache_options[@]}" --partition train validation
"$python_bin" -m thesis.refiner train "${common[@]}" --output-dir "$RUN_ROOT" \
    --epochs "${EPOCHS:-30}" --batch-size "${BATCH_SIZE:-8}" --patience "${PATIENCE:-7}" \
    --sampling-steps "${SAMPLING_STEPS:-8}" --seed "${SEED:-42}"
# The checkpoint and strategy are now fixed; only now prepare and score test data.
"$python_bin" -m thesis.refiner cache "${common[@]}" "${cache_options[@]}" --partition test
"$python_bin" -m thesis.refiner evaluate "${common[@]}" --run-dir "$RUN_ROOT" --output-dir "$report_root"
printf 'Model: %s\nComparison: %s/comparison.md\n' "$RUN_ROOT" "$report_root"
