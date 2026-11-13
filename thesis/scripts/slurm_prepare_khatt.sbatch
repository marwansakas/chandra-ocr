#!/usr/bin/env bash
#SBATCH --job-name=khatt-prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=khatt-prepare-%j.log

# Submit from the repository root; add your university partition/account as needed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the repository root}"
source .venv-khatt-transfer/bin/activate
export TRAIN_PYTHON="$PWD/.venv-khatt-transfer/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
bash thesis/scripts/run_khatt_transfer.sh prepare-khatt
