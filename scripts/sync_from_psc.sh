#!/usr/bin/env bash
# Bundle ONLY the artifacts we picked (5 chosen SAEs + their inference outputs
# + the 3 base models' final/ + tokenizer data) into a single gzipped tarball
# on PSC, transfer it, extract on this Mac.
#
# Why tar first: dashboards are ~30k small HTML files per SAE. Rsyncing those
# individually is dominated by per-file overhead and balloons over a slow link.
# Tar+gzip turns it into one big sequential download, and HTML compresses
# ~3-4x. Net: typically 5-10x faster wall-clock than per-file rsync.
#
# Layout produced on Mac (extracted at the repo root):
#   model/grid-L1-H6/...                 ← PSC's grid/grid-L1-H6/final/* (the
#   model/grid-L4-H6/...                   "final/" segment is dropped to match
#   model/grid-L8-H6/...                   your existing model/<name>/ convention)
#   saes/sae_runs/<model>/<sweep>/<trial>/final/...   (kept as-is)
#   saes/sae_inference/<model>/<trial>/...            (kept as-is)
#   saes/sae_inference/<model>/index_corpus.pt        (shared per model)
#   data/bioS_N-Bd_final_grid/old_to_new.json
#   data/bioS_N-Bd_final_grid/people.json
#
# Usage:
#   ./scripts/sync_from_psc.sh            # bundle + transfer + extract
#   ./scripts/sync_from_psc.sh bundle     # build the .tar.gz on PSC
#   ./scripts/sync_from_psc.sh transfer   # download an already-built tarball
#   ./scripts/sync_from_psc.sh extract    # extract a locally-present tarball
#   ./scripts/sync_from_psc.sh clean      # delete local + remote tarballs

set -euo pipefail

# PSC has two relevant hosts:
#   bridges2.psc.edu       — login node, full shell, run tar here
#   data.bridges2.psc.edu  — transfer-only, no arbitrary commands, rsync here
SSH_REMOTE=friedmae@bridges2.psc.edu
DATA_REMOTE=friedmae@data.bridges2.psc.edu
REMOTE_BASE=/jet/home/friedmae/data_storage/LM4_Results
REMOTE_TAR_DIR=$REMOTE_BASE/transfer
TARBALL=interp_lm4_bundle.tar.gz
LOCAL_REPO="$HOME/Code/Project Code/CRL-Interp/Interp_LM4"
LOCAL_TAR="$LOCAL_REPO/$TARBALL"

# The 5 SAEs picked from the sweep grades.
# Format: model_name|sweep_id|trial_name
SAES=(
  "grid-L4-H6|sweep-io37yc23|L0_mult8_l02_lr3e-05_ep50_n10000"
  "grid-L4-H6|sweep-gie2mbe3|L1_mult16_l02_lr3e-05_ep50_n10000"
  "grid-L4-H6|sweep-7t8t55s5|L2_mult16_l010_lr3e-05_ep50_n10000"
  "grid-L8-H6|sweep-6r4rakfq|L3_mult16_l02_lr3e-05_ep50_n10000"
  "grid-L1-H6|sweep-8sx9ovd5|L0_mult16_l02_lr3e-05_ep50_n10000"
)

# Base model checkpoint dirs to ship.
MODELS=(grid-L1-H6 grid-L4-H6 grid-L8-H6)

# Build the list of paths to feed to tar. All paths are relative to
# $REMOTE_BASE because the remote `tar` invocation cd's there first.
build_path_list() {
    local seen_models=""
    for entry in "${SAES[@]}"; do
        IFS='|' read -r model sweep trial <<< "$entry"
        echo "saes/sae_runs/$model/$sweep/$trial/final"
        echo "saes/sae_inference/$model/$trial"
        # Include the shared index_corpus.pt once per model
        if [[ "$seen_models" != *"|$model|"* ]]; then
            echo "saes/sae_inference/$model/index_corpus.pt"
            seen_models="$seen_models|$model|"
        fi
    done
    for m in "${MODELS[@]}"; do
        echo "runResults/bioS_N-Bd_final_grid/20260520-134455/grid/$m/final"
    done
    echo "Data/bioS_N-Bd_final_grid/old_to_new.json"
    echo "Data/bioS_N-Bd_final_grid/people.json"
    # CLT attribution feature dashboards: gen_feature_dashboards.py writes them on
    # PSC under $REMOTE_BASE/clt_features/<scan>/<idx>.json (PSC STORAGE_ROOT ==
    # REMOTE_BASE). They extract on the Mac to clt_storage/clt_features/ (see the
    # --transform rule below) so they match the Mac's STORAGE_ROOT (clts/storage.py).
    # Generate dashboards on PSC before syncing, or this path won't exist.
    echo "clt_features"
}

bundle_remote() {
    echo "=== Launching tarball build on PSC (background, survives disconnect) ==="
    local paths
    # Newline-joined list -> space-joined string for the tar command line
    paths=$(build_path_list | tr '\n' ' ')

    # nohup + redirect-everything + & detaches tar from the SSH session, so
    # closing your laptop / losing the connection won't SIGHUP it.
    # The two --transform rules rewrite paths inside the tar so that when
    # we extract at the repo root on Mac, files land in the right place:
    #   runResults/.../grid/grid-LX-HY/final/foo -> model/grid-LX-HY/foo
    #   Data/...                                 -> data/...
    ssh "$SSH_REMOTE" "set -e
        mkdir -p $REMOTE_TAR_DIR
        cd $REMOTE_BASE
        # Kill any prior bundle PID still hanging around
        if [[ -f $REMOTE_TAR_DIR/bundle.pid ]]; then
            prev=\$(cat $REMOTE_TAR_DIR/bundle.pid)
            kill -0 \$prev 2>/dev/null && { echo \"Killing prior tar PID \$prev\"; kill \$prev; sleep 1; }
            rm -f $REMOTE_TAR_DIR/bundle.pid
        fi
        nohup tar czhf $REMOTE_TAR_DIR/$TARBALL \
            --transform='s|^runResults/bioS_N-Bd_final_grid/20260520-134455/grid/\([^/]*\)/final|model/\1|' \
            --transform='s|^Data/|data/|' \
            --transform='s|^clt_features/|clt_storage/clt_features/|' \
            $paths \
            > $REMOTE_TAR_DIR/bundle.log 2>&1 </dev/null &
        echo \$! > $REMOTE_TAR_DIR/bundle.pid
        echo \"Tar started in background. PID=\$(cat $REMOTE_TAR_DIR/bundle.pid)\"
        echo \"Log:   $REMOTE_TAR_DIR/bundle.log\"
        echo \"Check: ./scripts/sync_from_psc.sh status\""
}

status_remote() {
    echo "=== Bundle status on PSC ==="
    ssh "$SSH_REMOTE" "
        cd $REMOTE_TAR_DIR 2>/dev/null || { echo 'no bundle dir yet'; exit 0; }
        if [[ -f bundle.pid ]] && kill -0 \$(cat bundle.pid) 2>/dev/null; then
            echo 'tar is RUNNING (pid '\$(cat bundle.pid)')'
        else
            echo 'tar is NOT running'
        fi
        echo
        echo 'Tarball:'
        ls -lh $TARBALL 2>/dev/null || echo '  (not yet)'
        echo
        echo 'Last 10 log lines:'
        tail -10 bundle.log 2>/dev/null || echo '  (no log)'
    "
}

wait_remote() {
    echo "=== Waiting for tarball to finish on PSC (polls every 30s) ==="
    while true; do
        local running
        running=$(ssh "$SSH_REMOTE" "[[ -f $REMOTE_TAR_DIR/bundle.pid ]] && kill -0 \$(cat $REMOTE_TAR_DIR/bundle.pid) 2>/dev/null && echo 1 || echo 0")
        if [[ "$running" == "0" ]]; then
            echo "Done."
            ssh "$SSH_REMOTE" "ls -lh $REMOTE_TAR_DIR/$TARBALL"
            return
        fi
        echo "  still tarring... ($(date '+%H:%M:%S'))"
        sleep 30
    done
}

transfer_to_local() {
    echo "=== Downloading tarball to Mac ==="
    rsync -avhL --progress \
        "$DATA_REMOTE:$REMOTE_TAR_DIR/$TARBALL" \
        "$LOCAL_TAR"
}

extract_local() {
    echo "=== Extracting into $LOCAL_REPO ==="
    [[ -f "$LOCAL_TAR" ]] || { echo "ERROR: $LOCAL_TAR not found. Run 'transfer' first."; exit 1; }
    tar xzf "$LOCAL_TAR" -C "$LOCAL_REPO"
    echo "Done."
}

clean_all() {
    echo "Deleting remote tarball ($SSH_REMOTE:$REMOTE_TAR_DIR/$TARBALL)..."
    ssh "$SSH_REMOTE" "rm -f $REMOTE_TAR_DIR/$TARBALL"
    echo "Deleting local tarball ($LOCAL_TAR)..."
    rm -f "$LOCAL_TAR"
}

case "${1:-all}" in
    bundle)    bundle_remote ;;
    status)    status_remote ;;
    wait)      wait_remote ;;
    transfer)  transfer_to_local ;;
    extract)   extract_local ;;
    clean)     clean_all ;;
    all)       bundle_remote; wait_remote; transfer_to_local; extract_local ;;
    *)
        echo "Usage: $0 {all|bundle|status|wait|transfer|extract|clean}"
        exit 1
        ;;
esac
