#!/bin/bash
#
# Submit CLT hyperparameter-sweep jobs for selected models in the
# bioS_N-Bd_final_grid. Each job runs the baked-in CLT sweep via
# scripts/train_clt_psc.sh:
#
#     expansion {4,8,16,32}  x  l0 {1,2,5}  (lr 1e-4, n_examples 50000, epochs 10)
#     = 12 wandb trials, run sequentially on one GPU-shared GPU.
#
# This is a thin driver: it just sets MODEL_NAME per model (inherited by the job)
# and an appropriate walltime, then submits train_clt_psc.sh once per model.
# train_clt_psc.sh runs SWEEP=1 / OVERRIDE_SWEEP=0 by default, so the baked-in
# grid is used and the values below are NOT overridden.
#
# Run from the repo root on a Bridges-2 (PSC) login node:
#     bash scripts/submit_clt_sweeps_grid.sh
#
set -euo pipefail

# --- which models to sweep (subdirs of the final grid on PSC) ---------------
MODELS=("grid-L2-H6" "grid-L8-H6")

# --- per-model walltime (GPU-shared). ESTIMATES -- validate after first run. -
# 12 trials run sequentially; hyperband (min_iter=5, eta=3) kills weak trials
# early. Deeper models cost more per trial, so the 8-layer gets more headroom.
declare -A TIME=(
    ["grid-L2-H6"]="04:00:00"
    ["grid-L8-H6"]="08:00:00"
)
DEFAULT_TIME="04:00:00"

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs

for m in "${MODELS[@]}"; do
    t="${TIME[$m]:-$DEFAULT_TIME}"
    echo "submitting CLT sweep: model=$m  time=$t"
    MODEL_NAME="$m" sbatch \
        --job-name="train-clt-$m" \
        --time="$t" \
        scripts/train_clt_psc.sh
done

echo
echo "submitted ${#MODELS[@]} job(s). watch with:  squeue -u \$USER"
echo "outputs land under: \$CLT_STORAGE_ROOT/clt_runs/<model>/sweep-<id>/"
