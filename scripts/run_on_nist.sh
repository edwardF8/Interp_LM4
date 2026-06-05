#!/bin/bash
# ============================================================================
# RUN ON THE NIST GPU MACHINE, from inside the unpacked bundle's Interp_LM4/ dir,
# AFTER scripts/setup_env_nist.sh and `source $HOME/lm4-ct-venv/bin/activate`.
#
# Resolves the bundled model/CLT/data (../artifacts/*) and runs the writeFeatures
# sweep on GPU. Any extra args pass straight through to the worker, e.g.:
#   bash scripts/run_on_nist.sh --months August --limit-people 50   # smoke / calibrate
#   bash scripts/run_on_nist.sh --num-shards 1 --shard-index 0       # full run, 1 GPU
#   bash scripts/run_on_nist.sh --num-shards 8 --shard-index 3       # one shard of 8
#
# Results land under the worker's storage root (clt_feature_explorer/<scan>/hpc).
# Merge after all shards finish:  python clts/merge_writefeatures_hpc.py --scan-name "$SCAN_NAME"
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ART="$(cd "$REPO/../artifacts" && pwd)"

export MODEL_DIR="$ART/model"
export CLT_DIR="$ART/clt"
export DATA_DIR="$ART/data"
export SCAN_NAME="$(cat "$ART/SCAN_NAME" 2>/dev/null || echo grid-L4-H6)"

echo "MODEL_DIR=$MODEL_DIR"
echo "CLT_DIR=$CLT_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "SCAN_NAME=$SCAN_NAME"
for p in "$MODEL_DIR/config.json" "$CLT_DIR/config.yaml" "$DATA_DIR/people.json"; do
    [ -e "$p" ] || { echo "MISSING: $p" >&2; exit 1; }
done

cd "$REPO"
python -u clts/run_writefeatures_hpc.py \
    --model-dir "$MODEL_DIR" --clt-dir "$CLT_DIR" --data-dir "$DATA_DIR" \
    --scan-name "$SCAN_NAME" --device cuda --progress-every 10 "$@"