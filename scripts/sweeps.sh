#!/usr/bin/env bash
# Submit and monitor the six SAE training sweeps on PSC.
#
# Run from the repo root (~/Interp_LM4) on a Bridges-2 login node:
#
#   ./scripts/sweeps.sh submit    # queue all 6 sweeps
#   ./scripts/sweeps.sh status    # one-shot status snapshot
#   ./scripts/sweeps.sh watch     # status, refreshing every 60s, until all jobs end
#   ./scripts/sweeps.sh cancel    # scancel anything still in the queue
#
# Layout: each sbatch runs trainSAE.py with --layers <N> --sweep, registering one
# wandb sweep (12 trials = 3 l0 x 2 sae_mult x 2 lr). Outputs land under
# /jet/.../data_storage/LM4_Results/saes/sae_runs/<model_name>/sweep-<id>/L<N>_*/final/.

set -euo pipefail

DATA=/jet/home/friedmae/data_storage/LM4_Results/Data/bioS_N-Bd_final_grid
GRID=/jet/home/friedmae/data_storage/LM4_Results/runResults/bioS_N-Bd_final_grid/20260520-134455/grid
STORAGE=/jet/home/friedmae/data_storage/LM4_Results/saes

# job_name | model_dir | layer | model_name (for trial counting under sae_runs/)
SWEEPS=(
  "L4-L0|$GRID/grid-L4-H6/final|0|grid-L4-H6"
  "L4-L1|$GRID/grid-L4-H6/final|1|grid-L4-H6"
  "L4-L2|$GRID/grid-L4-H6/final|2|grid-L4-H6"
  "L4-L3|$GRID/grid-L4-H6/final|3|grid-L4-H6"
  "L8-L3|$GRID/grid-L8-H6/final|3|grid-L8-H6"
  "L1-L0|$GRID/grid-L1-H6/final|0|grid-L1-H6"
)
EXPECTED_TRIALS=12   # 3 l0 * 2 sae_mult * 2 lr per layer sweep

# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------
submit_all() {
    echo "Submitting ${#SWEEPS[@]} sweeps..."
    for entry in "${SWEEPS[@]}"; do
        IFS='|' read -r jname mdir layer _ <<< "$entry"
        out=$(sbatch -J "$jname" -t 12:00:00 submit_job_psc.sh saes/trainSAE.py \
            --model-dir "$mdir" --data-dir "$DATA" --layers "$layer" --sweep)
        # sbatch prints "Submitted batch job 12345"
        jid=${out##* }
        printf "  %-8s -> job %s\n" "$jname" "$jid"
    done
    echo
    echo "Run \`$0 watch\` to monitor."
}

# ---------------------------------------------------------------------------
# status snapshot
# ---------------------------------------------------------------------------
show_status() {
    echo "============================================================"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    printf "%-8s  %-12s  %-10s  %-6s  %s\n" "JOB" "STATE" "ELAPSED" "TRIALS" "LATEST-JID"
    local any_running=0
    for entry in "${SWEEPS[@]}"; do
        IFS='|' read -r jname mdir layer mname <<< "$entry"

        # Most recent job matching this name (sacct gives newest last)
        local jid state elapsed
        jid=$(sacct -X --noheader --name="$jname" --format=JobID 2>/dev/null | tail -1 | tr -d ' ')
        if [[ -z "$jid" ]]; then
            state="never-submitted"; elapsed="-"
        else
            state=$(sacct -j "$jid" -X --noheader --format=State 2>/dev/null | head -1 | awk '{print $1}')
            elapsed=$(sacct -j "$jid" -X --noheader --format=Elapsed 2>/dev/null | head -1 | tr -d ' ')
        fi

        # Count completed trial dirs for this (model, layer) — across any sweep folders
        local trials
        trials=$(find "$STORAGE/sae_runs/$mname" -maxdepth 5 -type d -name final \
                      -path "*/L${layer}_*" 2>/dev/null | wc -l | tr -d ' ')

        # Mark running/pending so we know whether to keep watching
        case "$state" in
            RUNNING|PENDING|REQUEUED|CONFIGURING) any_running=1 ;;
        esac

        printf "%-8s  %-12s  %-10s  %-6s  %s\n" \
            "$jname" "$state" "$elapsed" "${trials}/${EXPECTED_TRIALS}" "$jid"
    done
    return $any_running
}

# ---------------------------------------------------------------------------
# watch loop — refreshes until every job has hit a terminal state
# ---------------------------------------------------------------------------
watch_loop() {
    while true; do
        clear || true
        if show_status; then
            echo
            echo "All jobs in terminal state."
            break
        fi
        echo
        echo "Refresh in 60s, Ctrl-C to exit watch (jobs keep running)."
        sleep 60
    done
}

# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------
cancel_all() {
    echo "Cancelling running/pending jobs for: ${SWEEPS[*]%%|*}"
    for entry in "${SWEEPS[@]}"; do
        IFS='|' read -r jname _ <<< "$entry"
        # Cancel any active jobs (RUNNING/PENDING) with this name; ignore terminal jobs.
        local jids
        jids=$(squeue -h -u "$USER" -n "$jname" -o %i 2>/dev/null | tr '\n' ' ')
        if [[ -n "$jids" ]]; then
            echo "  $jname: scancel $jids"
            # shellcheck disable=SC2086
            scancel $jids
        else
            echo "  $jname: nothing to cancel"
        fi
    done
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    submit) submit_all ;;
    status) show_status || true ;;
    watch)  watch_loop ;;
    cancel) cancel_all ;;
    *)
        echo "Usage: $0 {submit|status|watch|cancel}"
        exit 1
        ;;
esac
