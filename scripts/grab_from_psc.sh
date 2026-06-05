#!/bin/bash
# ============================================================================
# RUN ON THE NIST GPU MACHINE. Pulls the writeFeatures bundle from PSC (built by
# scripts/pack_for_nist.sh) and unpacks it. Uses rsync over SSH to PSC's
# dedicated transfer host -- it will prompt for your PSC password / 2FA.
#
# Copy THIS one small file to NIST first (paste or scp), then:
#   bash grab_from_psc.sh
#   # override any of these as needed:
#   PSC_USER=friedmae PSC_BUNDLE=/ocean/projects/cis240072p/bundle.tgz \
#     DEST=$HOME/lm4 bash grab_from_psc.sh
#
# If NIST cannot SSH to PSC at all, use Globus instead (PSC endpoint
# "PSC Bridges-2") to move the same tarball, then just run the tar -xzf line.
# ============================================================================
set -euo pipefail
PSC_USER="${PSC_USER:-friedmae}"
PSC_HOST="${PSC_HOST:-data.bridges2.psc.edu}"          # PSC's transfer node
PSC_BUNDLE="${PSC_BUNDLE:-/jet/home/friedmae/lm4_writefeatures_bundle.tar.gz}"
DEST="${DEST:-$HOME/lm4_writefeatures}"

mkdir -p "$DEST"
echo "pulling $PSC_USER@$PSC_HOST:$PSC_BUNDLE"
echo "    -> $DEST/"
rsync -avP "$PSC_USER@$PSC_HOST:$PSC_BUNDLE" "$DEST/"

echo "unpacking..."
tar -xzf "$DEST/$(basename "$PSC_BUNDLE")" -C "$DEST"
BUNDLE="$DEST/bundle"

echo
echo "================ UNPACKED -> $BUNDLE ================"
echo "Next, on NIST:"
echo "  cd $BUNDLE/Interp_LM4"
echo "  nvidia-smi                                  # note 'CUDA Version' (the driver)"
echo "  TORCH_CUDA=cuXXX bash scripts/setup_env_nist.sh   # XXX <= that CUDA version"
echo "  source \$HOME/lm4-ct-venv/bin/activate"
echo "  bash scripts/run_on_nist.sh --months August --limit-people 50   # smoke test"
echo "  bash scripts/run_on_nist.sh --num-shards 1 --shard-index 0       # full run"