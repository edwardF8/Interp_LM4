#!/bin/bash
# Recreate the CLT *training* conda env on PSC (the deleted `lm4`), but LEAN:
# only what clts/trainCLT.py actually needs. Installed with uv (fast + cached),
# and deliberately WITHOUT torchvision/torchaudio, sae-dashboard, or decode-clt
# (those bloated the old `lm4` and are only used by other tools). This is the
# training-side analogue of scripts/setup_ct_env_psc.sh.
#
# Pins match the last known-good run (wandb freeze 2026-06-03).
#
# Run once on PSC (login node is fine — no GPU needed to install):
#   bash scripts/setup_train_env_psc.sh
#   ENV_NAME=lm4 PYVER=3.11 bash scripts/setup_train_env_psc.sh      # override defaults
#
# Then launch training:  bash scripts/submit_clt_sweeps_grid.sh
# (or CONDA_ENV=<name> bash scripts/submit_clt_sweeps_grid.sh if you renamed it)
set -euo pipefail

ENV_NAME="${ENV_NAME:-lm4}"
PYVER="${PYVER:-3.11}"

# ---- conda + modules -------------------------------------------------------
module purge 2>/dev/null || true
module load anaconda3 2>/dev/null || true
module load cuda 2>/dev/null || true             # CUDA libs for the runtime torch
eval "$(conda shell.bash hook 2>/dev/null)" || {
    _base="$(conda info --base 2>/dev/null)"
    [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ] && source "$_base/etc/profile.d/conda.sh"
}

echo "=== creating conda env '$ENV_NAME' (python $PYVER) ==="
conda create -y -n "$ENV_NAME" "python=$PYVER"
conda activate "$ENV_NAME"
PYBIN="$(which python)"
echo "env: ${CONDA_DEFAULT_ENV:-none}   python: $PYBIN"

# ---- uv (fast installer, reuses ~/.cache/uv across envs) --------------------
python -m pip install -q -U pip uv
echo "uv: $(uv --version)"

# ---- lean training deps ----------------------------------------------------
# torch is the big one; uv will reuse a cached wheel if lm4-ct already pulled it.
# Add --index-url https://download.pytorch.org/whl/cu124 if the default resolve
# grabs the wrong CUDA build for the PSC driver.
#
# Only the top-level frameworks are pinned. numpy and the HuggingFace transitive
# pkgs are intentionally LEFT UNPINNED: on Python <3.12, transformer-lens caps
# numpy<2, so pinning numpy==2.4.4 (which only the py3.12 freeze used) makes the
# resolve unsatisfiable. Let the resolver pick compatible versions.
echo "=== uv pip install training deps ==="
uv pip install --python "$PYBIN" \
    "torch==2.12.0" \
    "transformer-lens==2.17.0" \
    "transformers==4.56.2" \
    "safetensors==0.4.5" \
    "wandb==0.27.0" \
    "datasets==3.6.0" \
    "einops==0.8.2"

# ---- import gate (mirrors train_clt_psc.sh preflight) ----------------------
echo "=== import gate ==="
python - <<'PY'
import torch, transformer_lens, safetensors, wandb, numpy, datasets, einops
print(f"OK  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
PY

echo
echo "Done -> conda env '$ENV_NAME' ready."
echo "Launch:  bash scripts/submit_clt_sweeps_grid.sh"
echo "(cuda_available prints False on a login node — that's expected; True inside the GPU job.)"
