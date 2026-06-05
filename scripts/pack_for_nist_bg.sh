#!/bin/bash
# ============================================================================
# RUN ON PSC. Launches scripts/pack_for_nist.sh DETACHED so it keeps running
# after you disconnect / close your laptop. Uses nohup (ignores the logout
# SIGHUP) and logs to $HOME, which is shared storage -- so you can reconnect to
# ANY login node later and check progress, and the finished tarball will be
# waiting regardless of which node you land on.
#
#   source scripts/env_writefeatures_psc.sh    # MUST do this first (exports paths)
#   bash   scripts/pack_for_nist_bg.sh          # launches, then you can close the laptop
#   # ... later, from any login node:
#   tail -f ~/pack_for_nist.log                 # progress; look for "BUNDLE READY"
#
# Honors the same knobs as pack_for_nist.sh (OUT=..., EXTRA_EXCLUDES=...), since
# nohup inherits your exported environment.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${LOG:-$HOME/pack_for_nist.log}"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${DATA_DIR:-}" ]; then
    echo "MODEL_DIR/DATA_DIR not set -- run 'source scripts/env_writefeatures_psc.sh' first." >&2
    exit 1
fi

: > "$LOG"
nohup bash "$HERE/pack_for_nist.sh" >"$LOG" 2>&1 &
PID=$!
disown "$PID" 2>/dev/null || true

echo "pack_for_nist running DETACHED (PID $PID)."
echo "  -> you can close your laptop now; it survives logout."
echo "  log:  $LOG"
echo "  watch:  tail -f $LOG"
echo "  done when the log prints:  ====== BUNDLE READY ======"