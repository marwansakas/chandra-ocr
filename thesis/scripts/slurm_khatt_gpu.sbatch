#!/usr/bin/env bash
#SBATCH --job-name=khatt-mdiff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=khatt-mdiff-%j.log

# Submit from the repository root; add your university GPU partition/account.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the repository root}"
source .venv-khatt-transfer/bin/activate
export TRAIN_PYTHON="$PWD/.venv-khatt-transfer/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
phase="${1:-pretrain}"
if [[ $# -gt 0 ]]; then shift; fi
case "$phase" in
    pretrain|finetune|test) exec bash thesis/scripts/run_khatt_transfer.sh "$phase" "$@" ;;
    *) echo 'GPU phases: pretrain, finetune, test' >&2; exit 1 ;;
esac
