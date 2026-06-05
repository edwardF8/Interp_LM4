#!/bin/bash
# Source this on PSC to populate the writeFeatures env vars for interactive runs
# (calibration, manual single-shard runs, the merge). These are the SAME defaults the
# sbatch (scripts/writefeatures_psc.sbatch) already bakes in, so the array job does not
# need them -- this file is for when you run commands by hand on a login/GPU node.
#
#   source scripts/env_writefeatures_psc.sh
#   # then e.g. calibrate the real GPU rate:
#   python -u clts/run_writefeatures_hpc.py --model-dir "$MODEL_DIR" --clt-dir "$CLT_DIR" \
#       --data-dir "$DATA_DIR" --scan-name "$SCAN_NAME" \
#       --months August --limit-people 50 --device cuda --progress-every 10
#
# Override any of these before sourcing (e.g. EXPORT REMOTE_BASE=... ) or after.

export REMOTE_BASE="${REMOTE_BASE:-/jet/home/friedmae/data_storage/LM4_Results}"
export MODEL_DIR="${MODEL_DIR:-$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid/grid-L4-H6/final}"
export CLT_DIR="${CLT_DIR:-$REMOTE_BASE/clt_runs/grid-L4-H6/sweep-cfbp6man/mult16_l02_lr0.0001_ep50_n10000/final}"
export DATA_DIR="${DATA_DIR:-$REMOTE_BASE/Data/bioS_N-Bd_final_grid}"
export SCAN_NAME="${SCAN_NAME:-grid-L4-H6}"
export CONDA_ENV="${CONDA_ENV:-lm4-ct}"

echo "writeFeatures env set:"
echo "  REMOTE_BASE = $REMOTE_BASE"
echo "  MODEL_DIR   = $MODEL_DIR"
echo "  CLT_DIR     = $CLT_DIR"
echo "  DATA_DIR    = $DATA_DIR"
echo "  SCAN_NAME   = $SCAN_NAME"
echo "  CONDA_ENV   = $CONDA_ENV"

# Quick existence check (prints MISSING lines if a path is wrong on this machine):
for _p in "$MODEL_DIR/config.json" "$CLT_DIR/config.yaml" "$DATA_DIR/people.json"; do
    [ -e "$_p" ] || echo "  MISSING: $_p"
done