#!/bin/bash
# ============================================================================
# RUN ON PSC (login node). Bundles EVERYTHING needed to run the writeFeatures
# sweep on another GPU machine (e.g. NIST) into ONE tarball:
#   * the repo code (clts/, util/, data/*.py, scripts/, tests/ ...)
#   * the model / CLT / dataset artifacts (which live outside the repo, under
#     $REMOTE_BASE/...), copied into artifacts/{model,clt,data}
#   * helper scripts that ride along inside the bundle:
#       scripts/setup_env_nist.sh   (build the python env on NIST)
#       scripts/run_on_nist.sh      (run the sweep on NIST)
#
# Pairs with scripts/grab_from_psc.sh, which you run ON THE NIST SIDE to pull
# this tarball over and unpack it.
#
#   source scripts/env_writefeatures_psc.sh   # sets MODEL_DIR/CLT_DIR/DATA_DIR
#   bash   scripts/pack_for_nist.sh           # -> $HOME/lm4_writefeatures_bundle.tar.gz
#   OUT=/ocean/projects/cis240072p/bundle.tgz bash scripts/pack_for_nist.sh   # if $HOME is tight
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# Pull MODEL_DIR / CLT_DIR / DATA_DIR / SCAN_NAME (same defaults as the sbatch).
# shellcheck disable=SC1091
source "$HERE/env_writefeatures_psc.sh"

OUT="${OUT:-$HOME/lm4_writefeatures_bundle.tar.gz}"
# Extra rsync excludes for the CODE copy, e.g. if your PSC repo also holds big
# in-tree artifact dirs:  EXTRA_EXCLUDES="--exclude model --exclude data/bioS_*"
EXTRA_EXCLUDES="${EXTRA_EXCLUDES:-}"

# Stage on the SAME filesystem as OUT (login-node /tmp is tiny).
STAGE="$(mktemp -d -p "$(dirname "$OUT")" lm4pack.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT
echo "staging in $STAGE"

# 1) code -- repo working tree minus heavy / machine-specific dirs.
# tar-pipe instead of rsync (login nodes lack rsync); GNU tar --exclude is
# no-anchored, so bare names like '.git'/'__pycache__' match at any depth.
mkdir -p "$STAGE/bundle/Interp_LM4"
echo "copying code   <- $REPO"
# shellcheck disable=SC2086
( cd "$REPO" && tar cf - \
    --exclude='.git' --exclude='.venv*' --exclude='*.venv*' \
    --exclude='clt_storage' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints' --exclude='logs' \
    $EXTRA_EXCLUDES . ) | ( cd "$STAGE/bundle/Interp_LM4" && tar xf - )

# 2) artifacts -- live outside the repo, copied with a clean relative layout.
# cp -aL dereferences any symlinks so the bundle is self-contained.
mkdir -p "$STAGE/bundle/artifacts/model" "$STAGE/bundle/artifacts/clt" "$STAGE/bundle/artifacts/data"
echo "copying model  <- $MODEL_DIR"; cp -aL "$MODEL_DIR/." "$STAGE/bundle/artifacts/model/"
echo "copying clt    <- $CLT_DIR";   cp -aL "$CLT_DIR/."   "$STAGE/bundle/artifacts/clt/"
echo "copying data   <- $DATA_DIR";  cp -aL "$DATA_DIR/."  "$STAGE/bundle/artifacts/data/"
echo "${SCAN_NAME:-grid-L4-H6}" > "$STAGE/bundle/artifacts/SCAN_NAME"

# 3) one tarball
echo "compressing -> $OUT"
tar -czf "$OUT" -C "$STAGE" bundle
echo
echo "================ BUNDLE READY ================"
du -sh "$OUT"
echo "path: $OUT"
echo
echo "Now ON THE NIST MACHINE run scripts/grab_from_psc.sh (copy that one small"
echo "script over first). If you changed OUT, pass PSC_BUNDLE=$OUT to it."