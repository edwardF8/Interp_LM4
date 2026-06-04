#!/bin/bash
# SLURM array job for the writeFeatures population sweep (GPU).
#
# Each array task runs ONE shard (work[shard_index::num_shards]) on ONE GPU; after the
# whole array finishes you run the merge ONCE. No attribution cache is written, so the
# only output is small aggregate partials (+ final reports) — ~100-150 MB total for the
# full ~50k-people x 46-template run, NOT the ~270 GB an interactive cache would cost.
#
# CALIBRATE FIRST (the model is tiny, so the GPU speedup is unknown until you measure):
#   1) clts/.venv-ct/bin/python clts/run_writefeatures_hpc.py --dry-run     # size: ~2.3M graphs
#   2) On a GPU node, time the real per-graph rate:
#        clts/.venv-ct/bin/python clts/run_writefeatures_hpc.py \
#            --months August --limit-people 50 --device cuda --progress-every 10
#      Read the printed "(N/s people)" rate; per-graph time = 1 / (rate * 46).
#   3) Pick NUM_SHARDS so each shard finishes inside --time:
#        graphs_per_shard = 2.3e6 / NUM_SHARDS ; shard_hours = graphs_per_shard * sec_per_graph / 3600
#      Set --array=0-(NUM_SHARDS-1) AND NUM_SHARDS below to the same value.
#
# Throughput tips (tiny model -> GPU is easy to under-utilize):
#   * Raise --batch-size (e.g. 1024) to keep the GPU busy.
#   * If one task barely uses a GPU, pack several shards per GPU by launching multiple
#     --shard-index values in the same task, or just use CPU (see the CPU variant note).

#SBATCH --job-name=writefeatures
#SBATCH --output=logs/wf_%A_%a.out
#SBATCH --error=logs/wf_%A_%a.err
#SBATCH --array=0-31              # <-- GPU shards (1 GPU each). Calibrate, then match NUM_SHARDS.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00

set -euo pipefail

# ----------------------------- EDIT THESE -----------------------------
REPO=/path/to/Interp_LM4                       # absolute path to the repo
NUM_SHARDS=32                                  # MUST equal the --array size above
DEVICE=cuda                                    # cuda (GPU) | cpu
BATCH_SIZE=1024                                # bigger batch fills the GPU; ignored-ish on cpu
EXTRA_ARGS=""                                  # e.g. "--templates 0,5,12" or "--limit-people 500"
# ----------------------------------------------------------------------
# CPU-only variant: set DEVICE=cpu, delete the "#SBATCH --gres=gpu:1" line, and widen
# the array a lot (e.g. --array=0-199, NUM_SHARDS=200) since CPU is ~65 CPU-days total.

cd "$REPO"
mkdir -p logs
PY="$REPO/clts/.venv-ct/bin/python"

"$PY" clts/run_writefeatures_hpc.py \
    --num-shards "$NUM_SHARDS" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --skip-existing \
    $EXTRA_ARGS

# --- Merge after the array (run manually, OR submit with a dependency) ---
# Manual:
#   "$PY" clts/merge_writefeatures_hpc.py
#
# Dependency (submit BOTH; merge waits for the whole array to succeed):
#   AID=$(sbatch --parsable scripts/writefeatures_hpc.sbatch)
#   sbatch --dependency=afterok:$AID --wrap="$PY clts/merge_writefeatures_hpc.py"
