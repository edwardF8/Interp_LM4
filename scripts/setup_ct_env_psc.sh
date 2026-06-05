#!/bin/bash
# Create a clean conda env for the CLT-attribution / writeFeatures pipeline on PSC,
# installed FAST with uv. This is the conda equivalent of clts/.venv-ct.
#
# Why a separate env: circuit-tracer pins transformers<=4.57.3 and safetensors>=0.5.0,
# which conflict with the SAE/training env's sae-dashboard (safetensors<0.5) and
# decode-clt (torchvision) — that's the pip resolver error you hit. So attribution
# lives in its OWN env, never installed into `lm4`. Mirrors clts/circuit_env/README.md.
#
# Run once on PSC (login node is fine — no GPU needed to install):
#   bash scripts/setup_ct_env_psc.sh
#   ENV_NAME=lm4-ct PYVER=3.11 bash scripts/setup_ct_env_psc.sh     # override defaults
#
# Then point jobs at it:  CONDA_ENV=lm4-ct sbatch scripts/writefeatures_psc.sbatch
# (lm4-ct is already the sbatch default after this change.)
set -euo pipefail

ENV_NAME="${ENV_NAME:-lm4-ct}"
PYVER="${PYVER:-3.11}"
CT_SRC="${CT_SRC:-/tmp/circuit-tracer}"          # local clone of the decoderesearch fork
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

# ---- uv (fast installer) ---------------------------------------------------
python -m pip install -q -U pip uv
echo "uv: $(uv --version)"

# ---- deps via uv (the same pins as clts/circuit_env/requirements.txt) ------
# On PSC/Linux, torch>=2.0.0 resolves to a CUDA wheel. If your driver needs a
# specific CUDA build, add e.g. --index-url https://download.pytorch.org/whl/cu124
echo "=== uv pip install requirements ==="
uv pip install --python "$PYBIN" -r "$REPO/clts/circuit_env/requirements.txt"

# ---- circuit-tracer from the decoderesearch fork ---------------------------
if [ ! -d "$CT_SRC/.git" ]; then
    echo "=== cloning circuit-tracer -> $CT_SRC ==="
    git clone https://github.com/decoderesearch/circuit-tracer.git "$CT_SRC"
fi
echo "=== uv pip install circuit-tracer ($CT_SRC) ==="
uv pip install --python "$PYBIN" "$CT_SRC"

# ---- import gate (must print OK) -------------------------------------------
echo "=== import gate ==="
python - <<'PY'
from transformer_lens.loading_from_pretrained import convert_llama_weights
from circuit_tracer.replacement_model.replacement_model_transformerlens import TransformerLensReplacementModel
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt
from circuit_tracer import attribute
from circuit_tracer.utils.create_graph_files import create_graph_files
import torch
print(f"OK  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
PY

echo
echo "Done -> conda env '$ENV_NAME' ready."
echo "Run the sweep with:  CONDA_ENV=$ENV_NAME sbatch scripts/writefeatures_psc.sbatch"
echo "(cuda_available prints False on a login node with no GPU — that's expected; it's True inside the GPU job.)"
