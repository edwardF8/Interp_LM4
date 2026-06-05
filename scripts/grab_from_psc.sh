#!/bin/bash
# ============================================================================
# RUN ON THE NIST GPU MACHINE. scp's the writeFeatures bundle from PSC (built by
# scripts/pack_for_nist.sh) and untars it. Prompts for your PSC password/2FA.
#
# Or just do it by hand -- this script is only those two lines:
#   scp friedmae@data.bridges2.psc.edu:/jet/home/friedmae/lm4_writefeatures_bundle.tar.gz .
#   tar -xzf lm4_writefeatures_bundle.tar.gz
#
# Override paths if you changed OUT on the pack side:
#   PSC_BUNDLE=/ocean/projects/cis240072p/bundle.tgz bash grab_from_psc.sh
# ============================================================================
set -euo pipefail
PSC_USER="${PSC_USER:-friedmae}"
PSC_HOST="${PSC_HOST:-data.bridges2.psc.edu}"          # PSC transfer node
PSC_BUNDLE="${PSC_BUNDLE:-/jet/home/friedmae/lm4_writefeatures_bundle.tar.gz}"
DEST="${DEST:-$HOME/lm4_writefeatures}"

mkdir -p "$DEST"
echo "scp $PSC_USER@$PSC_HOST:$PSC_BUNDLE -> $DEST/"
scp "$PSC_USER@$PSC_HOST:$PSC_BUNDLE" "$DEST/"
tar -xzf "$DEST/$(basename "$PSC_BUNDLE")" -C "$DEST"

echo
echo "Unpacked -> $DEST/bundle"
echo "Next:  cd $DEST/bundle/Interp_LM4"
echo "       nvidia-smi   # note CUDA Version"
echo "       TORCH_CUDA=cuXXX bash scripts/setup_env_nist.sh"
echo "       source \$HOME/lm4-ct-venv/bin/activate"
echo "       bash scripts/run_on_nist.sh --months August --limit-people 50"