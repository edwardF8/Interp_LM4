#!/bin/bash
#SBATCH --job-name=sweep-crl
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=12:00:00
#SBATCH --account=cis240072p
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egfriedm@andrew.cmu.edu
#
# SELF-CONTAINED GPU job: wandb sweep over SAE-CRL "interpretations" on PSC.
# Sweep axes = graph-sparsity beta (sets l_spB = l_spM, paper Eq.9) x latent L0 (topk):
#   beta in {0.001, 0.01, 0.1}  x  topk in {25, 100}  = 6 trials (grid, all run to completion).
# Small beta -> denser M / B_tau; large beta -> sparser causal graph. See build_sweep_config().
#
# Run from the repo root on a Bridges-2 login node (after `wandb login` once):
#   sbatch scripts/sweep_crl_psc.sbatch                 # layer 2 (default), full defaults
#   LAYER=3 sbatch scripts/sweep_crl_psc.sbatch         # sweep a different resid layer
#   N_BIOS=10000 EPOCHS=5 sbatch scripts/sweep_crl_psc.sbatch   # quicker pass
#   sbatch --export=ALL,MODEL_NAME=grid-L8-H6,MODEL_DIR=/path/final scripts/sweep_crl_psc.sbatch
#
# Outputs (storage_root() auto-resolves to REMOTE_BASE on PSC):
#   $REMOTE_BASE/sae_CRL_runs/<model_name>/sweep-<id>/<trial>/final/
set -euo pipefail

# ---- config (edit here, or override via env / --export) --------------------
REMOTE_BASE="${REMOTE_BASE:-/jet/home/friedmae/data_storage/LM4_Results}"
GRID="${GRID:-$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid}"
MODEL_NAME="${MODEL_NAME:-grid-L4-H6}"
MODEL_DIR="${MODEL_DIR:-$GRID/$MODEL_NAME/final}"     # ends in /final -> model_name auto = parent dir
DATA_DIR="${DATA_DIR:-$REMOTE_BASE/Data/bioS_N-Bd_final_grid}"
CONDA_ENV="${CONDA_ENV:-lm4}"
# fixed (non-swept) knobs — beta & topk come from the sweep, everything else from here:
LAYER="${LAYER:-2}"
Z_DIM="${Z_DIM:-3072}"        # 8x d_model(=384); held fixed so beta/topk are the only axes
EPOCHS="${EPOCHS:-10}"
N_BIOS="${N_BIOS:-50000}"
LR="${LR:-0.01}"
NOISE_MODE="${NOISE_MODE:-lap}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# ---- environment (mirrors submit_job_psc.sh) ------------------------------
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
# project root on import path so sae_CRL/ can import top-level clts/ & util/
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$(pwd)}${PYTHONPATH:+:$PYTHONPATH}"
# keep HF + wandb caches on node-local scratch if available
if [ -n "${LOCAL:-}" ]; then
    export HF_HOME="$LOCAL/hf_cache"; export WANDB_CACHE_DIR="$LOCAL/wandb_cache"
    mkdir -p "$HF_HOME" "$WANDB_CACHE_DIR"
fi

# ---- preflight: fail fast before booking GPU time --------------------------
echo "=== preflight ==="
fail=0
for p in "$MODEL_DIR/config.json" "$DATA_DIR/old_to_new.json" "$DATA_DIR/people.json" \
         sae_CRL/trainSAE_CRL.py; do
    [ -e "$p" ] || { echo "  MISSING: $p" >&2; fail=1; }
done
# sweeps need ONLINE wandb (offline mode can't orchestrate a sweep agent):
if ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "  wandb not authenticated: run \`wandb login\` on the login node, or set WANDB_API_KEY" >&2
    fail=1
fi
python - "$MODEL_DIR" <<'PY' || fail=1
import sys
for m in ("torch", "transformer_lens", "safetensors", "yaml", "wandb"):
    __import__(m)
import torch, json
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
print(f"  model: {cfg.get('num_hidden_layers')} layers, d_model={cfg.get('hidden_size')}")
PY
[ "$fail" = 0 ] || { echo "PREFLIGHT FAILED -- not launching sweep." >&2; exit 1; }
echo "preflight OK"

# ---- run sweep -------------------------------------------------------------
echo "=== SAE-CRL sweep (beta x topk, 6 trials) on $(hostname) ==="
date
nvidia-smi || true
python -u sae_CRL/trainSAE_CRL.py --sweep \
    --model-dir "$MODEL_DIR" \
    --data-dir  "$DATA_DIR" \
    --model-name "$MODEL_NAME" \
    --layer "$LAYER" \
    --z-dim "$Z_DIM" \
    --epochs "$EPOCHS" \
    --n-bios "$N_BIOS" \
    --lr "$LR" \
    --noise-mode "$NOISE_MODE"

echo "Finished: $(date)"
echo "Runs under: $REMOTE_BASE/sae_CRL_runs/$MODEL_NAME/sweep-<id>/"
