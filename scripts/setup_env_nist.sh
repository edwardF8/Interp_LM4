#!/bin/bash
# ============================================================================
# RUN ONCE ON THE NIST GPU MACHINE (from inside the unpacked bundle's
# Interp_LM4/ dir) to build the python env for the sweep.
#
# The two things that bit us on PSC, handled here in the right order:
#   * circuit-tracer pinned to the validated release @v0.4.1, and
#   * torch installed LAST from a CUDA build that matches the NIST driver
#     (installing circuit-tracer can otherwise drag in a too-new torch wheel).
#
# Steps:
#   1) nvidia-smi            -> read "CUDA Version" (top-right) = the driver's CUDA
#   2) pick a wheel tag <= that:  cu121 / cu124 / cu126 / cu128 ...
#   3) TORCH_CUDA=cu126 bash scripts/setup_env_nist.sh
#
# Uses a plain venv (no conda needed). Override ENV_DIR to relocate it.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TORCH_CUDA="${TORCH_CUDA:-cu126}"           # MATCH the NIST driver (see nvidia-smi)
ENV_DIR="${ENV_DIR:-$HOME/lm4-ct-venv}"
PYBIN_BASE="${PYBIN_BASE:-python3}"

echo "=== creating venv at $ENV_DIR ==="
"$PYBIN_BASE" -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python -m pip install -q -U pip uv
PYBIN="$(which python)"
echo "python: $PYBIN"

echo "=== deps (transformer-lens, transformers, etc.) ==="
uv pip install --python "$PYBIN" -r "$REPO/clts/circuit_env/requirements.txt"

echo "=== circuit-tracer @v0.4.1 (validated release) ==="
uv pip install --python "$PYBIN" \
  "git+https://github.com/decoderesearch/circuit-tracer.git@v0.4.1"

echo "=== torch LAST, build=${TORCH_CUDA} (must match NIST driver) ==="
uv pip install --python "$PYBIN" --reinstall --no-cache \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" torch

echo "=== verify ==="
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
python -c "from circuit_tracer.transcoder.cross_layer_transcoder import load_clt; import inspect; print('load_clt has scan:', 'scan' in str(inspect.signature(load_clt)))"
python -c "from circuit_tracer import attribute; print('circuit_tracer import OK')"
echo
echo "env ready: $ENV_DIR"
echo "activate with:  source $ENV_DIR/bin/activate"
echo "if cuda_available is False, re-run with TORCH_CUDA set to a build <= your nvidia-smi CUDA Version."