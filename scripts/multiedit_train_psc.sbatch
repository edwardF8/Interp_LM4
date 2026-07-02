#!/bin/bash
#SBATCH --job-name=multiedit-clt
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --account=cis240072p
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egfriedm@andrew.cmu.edu
#SBATCH --array=0-5
#
# GPU-hour-efficient multi-edit CLT training. Each array task = one GPU that runs a
# SHARD of train_jobs.jsonl (jobs[shard::NUM_SHARDS]) with CONCURRENCY trains PACKED
# on it (grid-L4-H6 is tiny -> a GPU fits several). Effective GPU-hours ~ sum(train
# time) / CONCURRENCY. Submit from the Interp_LM4 repo root on a Bridges-2 login node.
#
#   # calibrate first (one train, one shard) to measure per-train time + VRAM headroom:
#   JOBS=/path/train_jobs.jsonl NUM_SHARDS=108 CONCURRENCY=1 sbatch --array=0-0 scripts/multiedit_train_psc.sbatch
#   # then the full run (6 GPUs x 4 packed = 24 concurrent):
#   JOBS=/path/train_jobs.jsonl NUM_SHARDS=6   CONCURRENCY=4 sbatch --array=0-5 scripts/multiedit_train_psc.sbatch
set -euo pipefail
REMOTE_BASE="${REMOTE_BASE:-/jet/home/friedmae/data_storage/LM4_Results}"
JOBS="${JOBS:?set JOBS=/path/to/train_jobs.jsonl}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-6}}"
CONCURRENCY="${CONCURRENCY:-4}"          # trains packed per GPU (raise if VRAM allows)
TEST="${TEST:-0}"                        # 1 = smoke (first job, tiny budget) — gates the main run
CONDA_ENV="${CONDA_ENV:-lm4-ct}"
DATA_DIR="${DATA_DIR:-$REMOTE_BASE/Data/bioS_N-Bd_final_grid}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; mkdir -p logs
module load cuda 2>/dev/null || true
PY="${PY:-$HOME/.conda/envs/$CONDA_ENV/bin/python}"
[ -x "$PY" ] || { echo "ERROR: no python at $PY (env '$CONDA_ENV')" >&2; exit 1; }
[ -f "$JOBS" ] || { echo "ERROR: JOBS file not found: $JOBS" >&2; exit 1; }
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$(pwd)}${PYTHONPATH:+:$PYTHONPATH}"
export CLT_STORAGE_ROOT="$REMOTE_BASE"

echo "=== multiedit pack: shard ${SLURM_ARRAY_TASK_ID:-0}/$NUM_SHARDS conc=$CONCURRENCY on $(hostname) ==="
date; nvidia-smi || true
TEST_ARG=(); [ "$TEST" = 1 ] && TEST_ARG=(--test)
"$PY" -u scripts/multiedit_train_pack.py --jobs "$JOBS" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS" \
    --concurrency "$CONCURRENCY" --py "$PY" \
    --storage-root "$REMOTE_BASE" --data-dir "$DATA_DIR" \
    ${TEST_ARG[@]+"${TEST_ARG[@]}"}
echo "Finished shard ${SLURM_ARRAY_TASK_ID:-0}: $(date)"
