#!/bin/bash
#
# Circuit-tracer pipeline, STAGE 1 (feature dashboards) for a chosen CLT per
# model. For each (model, wandb run) it resolves the run's CLT dir and submits
# scripts/gen_dashboards_psc.sbatch (GPU), writing per-feature dashboards to
# $REMOTE_BASE/clt_features/<model>/<idx>.json -- the data the attribution-graph
# viewer loads. (Stage 2, the graphs themselves, is a separate driver.)
#
# Chosen runs:
#   grid-L1-H6 -> happy-deluge-193
#   grid-L2-H6 -> true-pond-212
#   grid-L8-H6 -> devout-morning-219
#
# Run from the repo root on a Bridges-2 (PSC) login node:
#     bash scripts/submit_dashboards_grid.sh              # MODE=moderate (1000 people)
#     MODE=full bash scripts/submit_dashboards_grid.sh    # all people (more GPU time)
#     MODE=validate bash scripts/submit_dashboards_grid.sh# smoke (100 people)
#
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-lm4}"        # training+dashboards env; circuit_tracer NOT needed here
REMOTE_BASE="${REMOTE_BASE:-/jet/home/friedmae/data_storage/LM4_Results}"
GRID="$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid"
DATA_DIR="$REMOTE_BASE/Data/bioS_N-Bd_final_grid"
FEATURES_ROOT="${FEATURES_ROOT:-$REMOTE_BASE/clt_features}"
MODE="${MODE:-moderate}"             # validate | moderate | full
TIME="${TIME:-04:00:00}"            # per-model walltime; raise for MODE=full

MODELS=(
    "grid-L1-H6:happy-deluge-193"
    "grid-L2-H6:true-pond-212"
    "grid-L8-H6:devout-morning-219"
)

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

# Resolve CLT dirs with the env's Python DIRECTLY -- no conda activation on the
# login node (which is what kept leaking into / breaking the jobs).
PY="${PY:-$HOME/.conda/envs/$CONDA_ENV/bin/python}"
[ -x "$PY" ] || { echo "ERROR: no python at $PY (is conda env '$CONDA_ENV' created?)" >&2; exit 1; }
resolve_clt() { "$PY" clts/resolve_clt_run.py "$1"; }

for pair in "${MODELS[@]}"; do
    model="${pair%%:*}"
    run="${pair#*:}"
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

    echo "  submitting dashboards job (mode=$MODE)"
    aid=$(sbatch --parsable \
        --export=ALL,REMOTE_BASE="$REMOTE_BASE",MODEL_DIR="$model_dir",CLT_DIR="$clt_dir",DATA_DIR="$DATA_DIR",SCAN_NAME="$model",FEATURES_ROOT="$FEATURES_ROOT",CONDA_ENV="$CONDA_ENV" \
        --job-name="dash-$model" \
        --time="$TIME" \
        scripts/gen_dashboards_psc.sbatch "$MODE")
    echo "  dashboards job: $aid"
done

echo
echo "submitted. watch with:  squeue -u \$USER"
echo "dashboards: $FEATURES_ROOT/<model>/<idx>.json"
echo "next: stage 2 (attribution graphs) once the prompt set is decided"
