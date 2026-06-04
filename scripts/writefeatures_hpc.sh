#!/bin/bash
# writeFeatures population sweep — dual-mode runner.
#
#   * As a SLURM array:   sbatch scripts/writefeatures_hpc.sh
#       (SLURM reads the #SBATCH directives; each task runs ONE shard. Run the merge
#        after, e.g. the afterok example at the bottom.)
#   * On a single node:   bash scripts/writefeatures_hpc.sh
#       (no SLURM needed: runs all shards 0..NUM_SHARDS-1 sequentially, then merges.
#        For one GPU box set NUM_SHARDS=1 so a single process does everything.)
#
# No attribution cache is written, so total output is ~100-150 MB of aggregates+reports.
#
# CALIBRATE on a GPU node first (model is tiny, so measure the real rate):
#   clts/.venv-ct/bin/python clts/run_writefeatures_hpc.py \
#       --months August --limit-people 50 --device cuda --batch-size 1024 --progress-every 10
# per-graph time = 1 / (printed people/s * 46); size NUM_SHARDS so each shard fits --time.

#SBATCH --job-name=writefeatures
#SBATCH --output=logs/wf_%A_%a.out
#SBATCH --error=logs/wf_%A_%a.err
#SBATCH --array=0-31              # GPU shards (1 GPU each); match NUM_SHARDS below.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00

set -euo pipefail

# ----------------------------- EDIT THESE -----------------------------
REPO=/path/to/Interp_LM4                       # absolute path to the repo
NUM_SHARDS=32                                  # SLURM: match --array size. Local: set 1 for one box.
DEVICE=cuda                                    # cuda (GPU) | cpu
BATCH_SIZE=1024                                # bigger batch fills the GPU
EXTRA_ARGS=""                                  # e.g. "--templates 0,5,12" or "--limit-people 500"
# ----------------------------------------------------------------------

cd "$REPO"
mkdir -p logs
PY="$REPO/clts/.venv-ct/bin/python"

run_shard () {
    "$PY" clts/run_writefeatures_hpc.py \
        --num-shards "$NUM_SHARDS" --shard-index "$1" \
        --device "$DEVICE" --batch-size "$BATCH_SIZE" --skip-existing $EXTRA_ARGS
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # SLURM array mode: this task runs its own shard. Merge separately afterwards.
    run_shard "${SLURM_ARRAY_TASK_ID}"
else
    # Local mode: run every shard in sequence on this node, then merge.
    echo "[local] running $NUM_SHARDS shard(s) sequentially on this node..."
    for ((s = 0; s < NUM_SHARDS; s++)); do
        echo "[local] shard $s/$NUM_SHARDS"
        run_shard "$s"
    done
    "$PY" clts/merge_writefeatures_hpc.py
    echo "[local] done — reports in <out-dir>/reports/"
fi

# --- Merge after a SLURM array (run manually, OR submit with a dependency) ---
#   "$PY" clts/merge_writefeatures_hpc.py
#   AID=$(sbatch --parsable scripts/writefeatures_hpc.sh)
#   sbatch --dependency=afterok:$AID --wrap="$PY clts/merge_writefeatures_hpc.py"
