#!/usr/bin/env bash
# Explicit phases; training never automatically evaluates the reserved test set.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$PWD"
WORK="${KHATT_TRANSFER_ROOT:-$ROOT/thesis/transfer_artifacts}"
PY="${TRAIN_PYTHON:-python}"
CAPACITY="${MDIFF_CAPACITY:-256}"
MAX_WIDTH="${MDIFF_MAX_WIDTH:-4096}"
resume=()
if [[ "${2:-}" == --resume ]]; then
    resume=(--resume)
elif [[ -n "${2:-}" ]]; then
    echo 'Only --resume is accepted after the phase name' >&2
    exit 1
fi
if [[ $# -gt 2 ]]; then
    echo 'Too many arguments' >&2
    exit 1
fi
if [[ ${#resume[@]} -gt 0 && "${1:-}" != pretrain && "${1:-}" != finetune ]]; then
    echo '--resume is only valid for pretrain or finetune' >&2
    exit 1
fi
case "${1:-help}" in
    prepare-khatt)
        "$PY" -m thesis.khatt_transfer setup-assets --output "$WORK/encoder"
        if [[ ! -f "$WORK/khatt_data/prepared.json" ]]; then
            source_args=()
            [[ -z "${KHATT_HF_ROOT:-}" ]] || source_args=(--hf-root "$KHATT_HF_ROOT")
            "$PY" -m thesis.khatt_transfer prepare-khatt --output "$WORK/khatt_data" "${source_args[@]}"
        fi
        "$PY" -m thesis.khatt_transfer check --prepared "$WORK/khatt_data" \
            --capacity "$CAPACITY" --max-width "$MAX_WIDTH"
        ;;
    pretrain)
        "$PY" -m thesis.khatt_transfer train --prepared "$WORK/khatt_data" \
            --output "$WORK/khatt_run" --encoder-checkpoint "$WORK/encoder/best.pth" \
            --capacity "$CAPACITY" --max-width "$MAX_WIDTH" \
            --epochs 100 --patience 10 --batch-size 1 --accumulation 16 \
            --device cuda "${resume[@]}"
        ;;
    prepare-children)
        if [[ ! -f "$WORK/children_data/prepared.json" ]]; then
            group_args=(--split-strategy "${CHILDREN_SPLIT_STRATEGY:-notebook}")
            [[ -z "${CHILDREN_WRITER_MAP:-}" ]] || group_args=(--writer-map "$CHILDREN_WRITER_MAP")
            [[ -z "${CHILDREN_ASSIGNMENTS:-}" ]] || group_args+=(--assignments "$CHILDREN_ASSIGNMENTS")
            "$PY" -m thesis.khatt_transfer prepare-children \
                --dataset-root "${CHILDREN_DATA_ROOT:-$ROOT/thesis/datasets/ground_truth_lines}" \
                --init-from "$WORK/khatt_run/best.pt" --output "$WORK/children_data" "${group_args[@]}"
        fi
        "$PY" -m thesis.khatt_transfer check --prepared "$WORK/children_data" \
            --init-from "$WORK/khatt_run/best.pt"
        ;;
    finetune)
        "$PY" -m thesis.khatt_transfer train --prepared "$WORK/children_data" \
            --init-from "$WORK/khatt_run/best.pt" --output "$WORK/children_run" \
            --epochs 30 --patience 7 --warmup-epochs 1 --decoder-lr 3e-5 --encoder-lr 1e-6 \
            --batch-size 1 --accumulation 16 --device cuda "${resume[@]}"
        ;;
    test)
        "$PY" -m thesis.khatt_transfer evaluate --prepared "$WORK/children_data" \
            --checkpoint "$WORK/children_run/best.pt" --partition test \
            --output "$WORK/children_test.json" --device cuda
        ;;
    help)
        echo 'Phases: prepare-khatt, pretrain [--resume], prepare-children, finetune [--resume], test'
        ;;
    *) echo "Unknown phase: $1" >&2; exit 1 ;;
esac
