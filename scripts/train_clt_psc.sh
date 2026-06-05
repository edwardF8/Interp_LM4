#!/bin/bash
#SBATCH --job-name=train-clt
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --account=cis240072p
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egfriedm@andrew.cmu.edu
#
# Trains a cross-layer transcoder (CLT) on a base Llama checkpoint via
# clts/trainCLT.py. Self-contained: env setup + fail-fast preflight + training.
# Submit from the repo root on a Bridges-2 login node:   sbatch scripts/train_clt_psc.sh
#
set -euo pipefail

# ############################################################################
# ###  EDIT YOUR HYPERPARAMETERS HERE  #######################################
# ############################################################################

# --- what to train on -------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-grid-L1-H6}"   # base model identifier (and output subdir); override via env (see submit_clt_sweeps_grid.sh)
REMOTE_BASE="/jet/home/friedmae/data_storage/LM4_Results"
GRID="$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid"
MODEL_DIR="$GRID/$MODEL_NAME/final"               # HF checkpoint dir
DATA_DIR="$REMOTE_BASE/Data/bioS_N-Bd_final_grid" # has people.json + old_to_new.json

# --- CLT activation sites ({layer} is filled in for EVERY block) ------------
# A CLT spans all layers at once; there is no single layer to pick. Change the
# hook SITE here (e.g. hook_resid_pre / hook_resid_post) to move where it reads/writes.
ENC_HOOK="blocks.{layer}.hook_resid_mid"   # encoder input
DEC_HOOK="blocks.{layer}.hook_mlp_out"     # decoder target

# --- mode: pick ONE of the two sections below -------------------------------
SWEEP=1       # 0 = single run (SECTION A) ; 1 = wandb grid sweep (SECTION B)
CONTEXT_SIZE=512    # (applies to both)

# === SECTION A: single-run hyperparameters (used when SWEEP=0) ===============
N_EXAMPLES=1000     # 1000 = quick test; 10000 = full
EPOCHS=20           # 3 = quick test; 30 = full
EXPANSION=16        # d_transcoder = EXPANSION * d_model
L0=5.0              # sparsity (L0) coefficient
LR=5e-5             # learning rate

# === SECTION B: sweep grid (used when SWEEP=1) ==============================
# OVERRIDE_SWEEP=0 -> run the baked-in grid (expansion {4,8,16,32} x l0 {1,2,5},
# lr 1e-4, n_examples 50000, epochs 10) and IGNORE the values below.
# OVERRIDE_SWEEP=1 -> use the values below. Lists are space-separated.
OVERRIDE_SWEEP=0
SWEEP_EXPANSION="4 8 16 32"
SWEEP_L0="1 2 5"
SWEEP_LR=1e-4
SWEEP_N_EXAMPLES=20000
SWEEP_EPOCHS=20

# --- environment ------------------------------------------------------------
CONDA_ENV="lm4"

# ############################################################################
# ###  machinery below — you shouldn't need to touch this  ###################
# ############################################################################

# CLT trainer resolves its output root from this env var (see trainCLT.py).
export CLT_STORAGE_ROOT="$REMOTE_BASE"

# If overriding the sweep grid, hand the values to build_sweep_config() via env.
# When OVERRIDE_SWEEP=0 these stay unset -> trainCLT.py uses its baked-in grid.
if [ "$SWEEP" = 1 ] && [ "$OVERRIDE_SWEEP" = 1 ]; then
    export CLT_SWEEP_EXPANSION="$SWEEP_EXPANSION"
    export CLT_SWEEP_L0="$SWEEP_L0"
    export CLT_SWEEP_LR="$SWEEP_LR"
    export CLT_SWEEP_N_EXAMPLES="$SWEEP_N_EXAMPLES"
    export CLT_SWEEP_EPOCHS="$SWEEP_EPOCHS"
    echo "sweep grid OVERRIDDEN: expansion=[$SWEEP_EXPANSION] l0=[$SWEEP_L0] lr=$SWEEP_LR n_examples=$SWEEP_N_EXAMPLES epochs=$SWEEP_EPOCHS"
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

module purge
module load cuda
module load anaconda3
eval "$(conda shell.bash hook 2>/dev/null)" || {
    _base="$(conda info --base 2>/dev/null)"
    [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ] && source "$_base/etc/profile.d/conda.sh"
}
conda deactivate 2>/dev/null || true
conda activate "$CONDA_ENV"
echo "env: ${CONDA_DEFAULT_ENV:-none}   python: $(which python)"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$(pwd)}${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${LOCAL:-}" ]; then
    export HF_HOME="$LOCAL/hf_cache"; export WANDB_CACHE_DIR="$LOCAL/wandb_cache"
    mkdir -p "$HF_HOME" "$WANDB_CACHE_DIR"
fi

# ---- preflight: fail fast before booking GPU time --------------------------
echo "=== preflight ==="
fail=0
for p in "$MODEL_DIR/config.json" "$DATA_DIR/people.json" "$DATA_DIR/old_to_new.json" \
         clts/trainCLT.py; do
    [ -e "$p" ] || { echo "  MISSING: $p" >&2; fail=1; }
done
if [ "$SWEEP" = 1 ]; then
    if ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null && [ -z "${WANDB_API_KEY:-}" ]; then
        echo "  wandb not authenticated: run \`wandb login\`, or set WANDB_API_KEY" >&2
        fail=1
    fi
fi
python - "$MODEL_DIR" <<'PY' || fail=1
import sys, json
for m in ("torch", "transformer_lens", "safetensors", "wandb"):
    __import__(m)
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
print(f"  model: {cfg.get('num_hidden_layers')} layers, d_model={cfg.get('hidden_size')}")
PY
[ "$fail" = 0 ] || { echo "PREFLIGHT FAILED -- not launching." >&2; exit 1; }
echo "preflight OK"

# ---- run -------------------------------------------------------------------
mode="single run"; [ "$SWEEP" = 1 ] && mode="sweep (expansion x l0)"
echo "=== CLT $mode on $(hostname) ==="
echo "    enc=$ENC_HOOK  dec=$DEC_HOOK  n_examples=$N_EXAMPLES  epochs=$EPOCHS"
date
nvidia-smi || true

cmd=(python -u clts/trainCLT.py
    --model-dir "$MODEL_DIR"
    --data-dir  "$DATA_DIR"
    --model-name "$MODEL_NAME"
    --enc-hook-template "$ENC_HOOK"
    --dec-hook-template "$DEC_HOOK"
    --expansion "$EXPANSION"
    --l0 "$L0"
    --lr "$LR"
    --epochs "$EPOCHS"
    --context-size "$CONTEXT_SIZE"
    --n-examples "$N_EXAMPLES")
[ "$SWEEP" = 1 ] && cmd+=(--sweep)

"${cmd[@]}"

echo "Finished: $(date)"
echo "Runs under: $CLT_STORAGE_ROOT/clt_runs/$MODEL_NAME/"
