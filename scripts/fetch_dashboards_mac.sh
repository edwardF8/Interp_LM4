#!/usr/bin/env bash
# Pull the CLT dashboard tarball from PSC and extract it under the Mac's
# STORAGE_ROOT (clt_storage/clt_features/<scan>/). Run this ON THE MAC after
# scripts/tar_dashboards_psc.sh has built the tarball on PSC.
#
# Usage (on the Mac, from anywhere):
#   scripts/fetch_dashboards_mac.sh                 # scan = grid-L4-H6
#   scripts/fetch_dashboards_mac.sh <scan-name>
set -euo pipefail

SCAN_NAME="${1:-grid-L4-H6}"
DATA_REMOTE="${DATA_REMOTE:-friedmae@data.bridges2.psc.edu}"   # PSC data-transfer node
REMOTE_BASE="${REMOTE_BASE:-/jet/home/friedmae/data_storage/LM4_Results}"
TARBALL="clt_features_${SCAN_NAME}.tar.gz"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Pulling $TARBALL from PSC ..."
rsync -avhP "$DATA_REMOTE:$REMOTE_BASE/transfer/$TARBALL" .

mkdir -p clt_storage
echo "Extracting into clt_storage/ ..."
tar xzf "$TARBALL" -C clt_storage/
rm -f "$TARBALL"

dest="clt_storage/clt_features/$SCAN_NAME"
n="$( { ls "$dest"/*.json 2>/dev/null || true; } | wc -l | tr -d ' ')"
echo "Done: $n dashboards at $dest/"
echo
echo "Build a graph + serve (dashboards now resolve locally):"
echo "  clts/.venv-ct/bin/python clts/build_attribution_graph.py \\"
echo "      --model-dir model/$SCAN_NAME \\"
echo "      --clt-dir   clts/clt_runs/$SCAN_NAME/<run>/final \\"
echo "      --data-dir  data/bioS_N-Bd_final_grid --slug bday-recall"
echo "  clts/.venv-ct/bin/python clts/serve_ui.py \\"
echo "      --graph-dir clt_storage/clt_graphs/$SCAN_NAME/bday-recall \\"
echo "      --features-dir $dest --scan-name $SCAN_NAME"
