#!/bin/bash
#
# Run the writeFeatures stage (compute all attribution-graph features over every
# person x template) for a chosen CLT per model. For each (model, wandb run):
#   1. resolve the run's CLT checkpoint dir from its wandb `storage_path`,
#   2. submit the sharded GPU array (scripts/writefeatures_psc.sbatch),
#   3. chain a merge job (afterok) that aggregates the shards once.
#
# The chosen runs (edit MODELS below):
#   grid-L1-H6 -> happy-deluge-193
#   grid-L2-H6 -> true-pond-212
#   grid-L8-H6 -> devout-morning-219
#
# Outputs land under $REMOTE_BASE/clt_feature_explorer/<scan-name>/hpc/.
# Workers default to --skip-existing, so a shard that hits the walltime can just
# be re-submitted and it resumes.
#
# Run from the repo root on a Bridges-2 (PSC) login node:
#     bash scripts/submit_writefeatures_grid.sh
#     NUM_SHARDS=32 TIME=12:00:00 bash scripts/submit_writefeatures_grid.sh   # deeper/wider
#
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-lm4-ct}"
REMOTE_BASE="${REMOTE_BASE:-/jet/home/friedmae/data_storage/LM4_Results}"
GRID="$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid"
DATA_DIR="$REMOTE_BASE/Data/bioS_N-Bd_final_grid"
NUM_SHARDS="${NUM_SHARDS:-16}"     # must match the --array size below
TIME="${TIME:-08:00:00}"          # per-shard walltime
ACCOUNT="${ACCOUNT:-cis240072p}"

# model : wandb run name  (the chosen CLT for each model)
MODELS=(
    "grid-L1-H6:happy-deluge-193"
    "grid-L2-H6:true-pond-212"
    "grid-L8-H6:devout-morning-219"
)

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

# IMPORTANT: do NOT `conda activate` in this (submitting) shell -- with sbatch
# --export=ALL the CONDA_* vars leak into the job and break its own conda setup.
# Resolve CLT dirs inside the env via a subshell instead.
module load anaconda3 2>/dev/null || true
resolve_clt() {  # $1 = wandb run name -> prints storage_path (CLT dir) on stdout
    ( eval "$(conda shell.bash hook 2>/dev/null)" \
        || { _b=$(conda info --base 2>/dev/null); [ -n "$_b" ] && . "$_b/etc/profile.d/conda.sh"; }
      conda activate "$CONDA_ENV" 1>&2
      python clts/resolve_clt_run.py "$1" )
}

for pair in "${MODELS[@]}"; do
    model="${pair%%:*}"
    run="${pair#*:}"
    scan="$model"
    model_dir="$GRID/$model/final"

    echo "=== $model ($run): resolving CLT dir from wandb ==="
    if ! clt_dir="$(resolve_clt "$run")"; then
        echo "  !! could not resolve $run -- skipping $model" >&2
        continue
    fi
    echo "  CLT_DIR=$clt_dir"
    if [ ! -e "$clt_dir/config.yaml" ] || [ ! -e "$model_dir/config.json" ]; then
        echo "  !! missing $clt_dir/config.yaml or $model_dir/config.json -- skipping" >&2
        continue
    fi

    echo "  submitting writeFeatures array ($NUM_SHARDS shards, ${TIME} each)"
    aid=$(sbatch --parsable \
        --export=ALL,REMOTE_BASE="$REMOTE_BASE",MODEL_DIR="$model_dir",CLT_DIR="$clt_dir",DATA_DIR="$DATA_DIR",SCAN_NAME="$scan",NUM_SHARDS="$NUM_SHARDS",CONDA_ENV="$CONDA_ENV" \
        --job-name="wf-$model" \
        --array="0-$((NUM_SHARDS - 1))" \
        --time="$TIME" \
        scripts/writefeatures_psc.sbatch)
    echo "  array job: $aid"

    # merge once, after all shards succeed (tiny CPU job on RM-shared)
    mid=$(sbatch --parsable \
        --dependency="afterok:$aid" \
        --account="$ACCOUNT" --partition=RM-shared --time=00:30:00 \
        --job-name="wf-merge-$model" \
        --output="logs/%x-%j.out" --error="logs/%x-%j.err" \
        --export=ALL,CONDA_ENV="$CONDA_ENV",SCAN_NAME="$scan" \
        --wrap="cd \"\$SLURM_SUBMIT_DIR\"; module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate $CONDA_ENV; export PYTHONPATH=\$SLURM_SUBMIT_DIR; python -u clts/merge_writefeatures_hpc.py --scan-name $scan")
    echo "  merge job: $mid (runs after $aid)"
done

echo
echo "submitted. watch with:  squeue -u \$USER"
echo "results: \$REMOTE_BASE/clt_feature_explorer/<model>/hpc/"
