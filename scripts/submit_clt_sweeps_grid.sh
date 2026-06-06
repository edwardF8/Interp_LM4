#!/bin/bash
#
# Submit CLT grid sweeps for the bioS_N-Bd_final_grid models as SLURM ARRAY
# jobs: one array task per (expansion, l0) point, each a single training run
# (scripts/train_clt_psc.sh in SWEEP=0 mode). Tasks run concurrently, so a
# "large" grid finishes in ~one slowest-single-run instead of the sum -- and
# each task fits its own walltime, sidestepping the 24h-no-resume problem that
# a single sequential wandb sweep hit on the deep models.
#
#   grid-L1-H6 : tuning sweep, exp{8,16} x l0{1,2,5}      = 6 runs  (drop exp4)
#   grid-L2-H6 : full grid,    exp{4,8,16,32} x l0{1,2,5} = 12 runs
#   grid-L8-H6 : full grid,    exp{4,8,16,32} x l0{1,2,5} = 12 runs
#
# Each run logs to wandb (project interpLM4) and is grouped there under
# "grid-array-<model>" for easy comparison; outputs land under
# $CLT_STORAGE_ROOT/clt_runs/<model>/standalone/<trial>/final/.
#
# Run from the repo root on a Bridges-2 (PSC) login node:
#     bash scripts/submit_clt_sweeps_grid.sh
#
set -euo pipefail

# --- shared single-run hyperparameters (every array task uses these) --------
N_EXAMPLES=50000
EPOCHS=15            # modest bump from 10 (let sparsity converge a bit more)
LR=1e-4
MAX_CONCURRENT=6     # cap simultaneous array tasks (politeness to the allocation)

# --- per-model: expansion grid | l0 grid | walltime PER TASK ----------------
MODELS=("grid-L1-H6" "grid-L2-H6" "grid-L8-H6")
declare -A EXPANSION=(
    ["grid-L1-H6"]="8 16"        # tuning: drop exp4 (it underperformed in the L1 sweep)
    ["grid-L2-H6"]="4 8 16 32"
    ["grid-L8-H6"]="4 8 16 32"
)
declare -A L0=(
    ["grid-L1-H6"]="1 2 5"
    ["grid-L2-H6"]="1 2 5"
    ["grid-L8-H6"]="1 2 5"
)
declare -A TIME=(   # per single run; L8 exp32 is the long pole (~17h est)
    ["grid-L1-H6"]="08:00:00"
    ["grid-L2-H6"]="12:00:00"
    ["grid-L8-H6"]="24:00:00"
)

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs

for m in "${MODELS[@]}"; do
    read -r -a _e <<< "${EXPANSION[$m]}"
    read -r -a _l <<< "${L0[$m]}"
    n=$(( ${#_e[@]} * ${#_l[@]} ))
    last=$(( n - 1 ))
    echo "submitting CLT grid array: model=$m  (${#_e[@]} x ${#_l[@]} = $n runs)  array=0-$last%$MAX_CONCURRENT  time=${TIME[$m]}"
    echo "    expansion=[${EXPANSION[$m]}] x l0=[${L0[$m]}]  n_examples=$N_EXAMPLES epochs=$EPOCHS lr=$LR"
    MODEL_NAME="$m" \
    SWEEP=0 \
    GRID_EXPANSION="${EXPANSION[$m]}" \
    GRID_L0="${L0[$m]}" \
    LR="$LR" \
    EPOCHS="$EPOCHS" \
    N_EXAMPLES="$N_EXAMPLES" \
    sbatch \
        --job-name="clt-grid-$m" \
        --array="0-$last%$MAX_CONCURRENT" \
        --time="${TIME[$m]}" \
        scripts/train_clt_psc.sh
done

echo
echo "submitted ${#MODELS[@]} array job(s). watch with:  squeue -u \$USER"
echo "compare in wandb: project interpLM4, group 'grid-array-<model>'"
