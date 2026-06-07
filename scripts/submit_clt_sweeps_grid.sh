#!/bin/bash
#
# Submit CLT grid sweeps for the bioS_N-Bd_final_grid models as REAL wandb
# sweeps run with PARALLEL AGENTS:
#   1. register one wandb sweep per model (a proper Sweep object in the UI), then
#   2. launch a SLURM array of agents that all pull trials from that sweep.
#
# Each agent runs with --count 1 (exactly one trial, then exits), and the array
# is capped at %MAX_CONCURRENT, so: real wandb Sweep grouping/plots, trials run
# in parallel, and each task's walltime is bounded by ONE trial (fits the
# 24h-no-resume limit even for L8's exp32).
#
#   grid-L1-H6 : exp{8,16} x l0{1,2,5}      = 6 trials  (drop exp4)
#   grid-L2-H6 : exp{4,8,16,32} x l0{1,2,5} = 12 trials
#   grid-L8-H6 : exp{4,8,16,32} x l0{1,2,5} = 12 trials
#
# Sweeps appear in wandb project interpLM4; outputs land under
# $CLT_STORAGE_ROOT/clt_runs/<model>/sweep-<id>/<trial>/final/.
#
# Run from the repo root on a Bridges-2 (PSC) login node:
#     bash scripts/submit_clt_sweeps_grid.sh
#
set -euo pipefail

# --- environment ------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-lm4-ct}"     # lm4 was deleted; lm4-ct has the training stack
REMOTE_BASE="/jet/home/friedmae/data_storage/LM4_Results"
GRID="$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid"
DATA_DIR="$REMOTE_BASE/Data/bioS_N-Bd_final_grid"

# --- shared sweep hyperparameters (fixed; the grid varies expansion x l0) ----
N_EXAMPLES=50000
EPOCHS=15
LR=1e-4
MAX_CONCURRENT=6     # cap simultaneous agents PER model array

# --- per-model: expansion grid | l0 grid | walltime PER TRIAL ---------------
MODELS=("grid-L1-H6" "grid-L2-H6" "grid-L8-H6")
declare -A EXPANSION=(
    ["grid-L1-H6"]="8 16"        # tuning: drop exp4 (underperformed in the L1 sweep)
    ["grid-L2-H6"]="4 8 16 32"
    ["grid-L8-H6"]="4 8 16 32"
)
declare -A L0=(
    ["grid-L1-H6"]="1 2 5"
    ["grid-L2-H6"]="1 2 5"
    ["grid-L8-H6"]="1 2 5"
)
declare -A TIME=(   # one trial; L8 exp32 is the long pole (~15h measured)
    ["grid-L1-H6"]="08:00:00"
    ["grid-L2-H6"]="12:00:00"
    ["grid-L8-H6"]="24:00:00"
)

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs
export CLT_STORAGE_ROOT="$REMOTE_BASE"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

# conda is needed on THIS login node to register the sweeps (no GPU required).
module load anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || {
    _base="$(conda info --base 2>/dev/null)"
    [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ] && source "$_base/etc/profile.d/conda.sh"
}
conda activate "$CONDA_ENV"

for m in "${MODELS[@]}"; do
    read -r -a _e <<< "${EXPANSION[$m]}"
    read -r -a _l <<< "${L0[$m]}"
    n=$(( ${#_e[@]} * ${#_l[@]} ))
    last=$(( n - 1 ))

    echo "=== $m: registering wandb sweep (expansion=[${EXPANSION[$m]}] x l0=[${L0[$m]}] -> $n trials) ==="
    SWEEP_ID=$(CLT_SWEEP_EXPANSION="${EXPANSION[$m]}" \
               CLT_SWEEP_L0="${L0[$m]}" \
               CLT_SWEEP_LR="$LR" \
               CLT_SWEEP_N_EXAMPLES="$N_EXAMPLES" \
               CLT_SWEEP_EPOCHS="$EPOCHS" \
               python clts/trainCLT.py --create-sweep \
                   --model-dir "$GRID/$m/final" --data-dir "$DATA_DIR" --model-name "$m" \
               | sed -n 's/^SWEEP_ID=//p')
    if [ -z "${SWEEP_ID:-}" ]; then
        echo "  !! failed to register sweep for $m -- skipping" >&2
        continue
    fi
    echo "  sweep id: $SWEEP_ID"
    echo "  launching $n agents (count=1 each, $MAX_CONCURRENT concurrent, walltime ${TIME[$m]})"

    MODEL_NAME="$m" \
    AGENT_SWEEP_ID="$SWEEP_ID" \
    AGENT_COUNT=1 \
    CONDA_ENV="$CONDA_ENV" \
    sbatch \
        --export=ALL \
        --job-name="clt-sweep-$m" \
        --array="0-$last%$MAX_CONCURRENT" \
        --time="${TIME[$m]}" \
        scripts/train_clt_psc.sh
done

echo
echo "submitted. watch with:  squeue -u \$USER"
echo "sweeps: wandb project interpLM4 -> Sweeps tab (clt_sweep_<model>)"
