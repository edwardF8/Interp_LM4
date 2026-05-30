#!/usr/bin/env bash
# Tar the generated CLT feature dashboards on PSC for transfer to the Mac.
# Run this ON PSC (login node is fine -- tarring ~24k small JSONs into one
# gzipped file is far faster over the link than per-file rsync). Then pull the
# tarball from the Mac with scripts/fetch_dashboards_mac.sh.
#
# Usage (on PSC, from anywhere):
#   scripts/tar_dashboards_psc.sh                 # scan = grid-L4-H6
#   scripts/tar_dashboards_psc.sh <scan-name>
#   REMOTE_BASE=/other/root scripts/tar_dashboards_psc.sh <scan-name>
set -euo pipefail

REMOTE_BASE="${REMOTE_BASE:-/jet/home/friedmae/data_storage/LM4_Results}"
SCAN_NAME="${1:-grid-L4-H6}"
SRC="$REMOTE_BASE/clt_features/$SCAN_NAME"
OUT_DIR="$REMOTE_BASE/transfer"
TARBALL="$OUT_DIR/clt_features_${SCAN_NAME}.tar.gz"

[ -d "$SRC" ] || { echo "ERROR: no dashboards dir at $SRC -- generate them first." >&2; exit 1; }
n="$( { ls "$SRC"/*.json 2>/dev/null || true; } | wc -l | tr -d ' ')"
[ "$n" -gt 0 ] || { echo "ERROR: $SRC has no .json dashboards (generation incomplete?)." >&2; exit 1; }
mkdir -p "$OUT_DIR"

echo "Taring $n dashboards from $SRC ..."
# -C so the archive holds  clt_features/<scan>/...  -> extracts cleanly under
# the Mac's STORAGE_ROOT (clt_storage/).
tar czf "$TARBALL" -C "$REMOTE_BASE" "clt_features/$SCAN_NAME"
echo "Wrote $(du -h "$TARBALL" | cut -f1)  ->  $TARBALL"
echo
echo "Now pull it to the Mac:"
echo "    scripts/fetch_dashboards_mac.sh $SCAN_NAME"
echo "  (or manually:"
echo "    rsync -avhP friedmae@data.bridges2.psc.edu:$TARBALL ."
echo "    tar xzf clt_features_${SCAN_NAME}.tar.gz -C clt_storage/ )"
