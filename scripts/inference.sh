#!/usr/bin/env bash
# Run the post-sweep inference pass (feature_stats + buckets + dashboards) for
# the five selected best-of-sweep SAEs.
#
# Each sbatch runs saes/runInference.py on ONE trial dir — runInference's
# find_saes() rglob's for `final/` under whatever you pass as --sweep-dir, so
# pointing it at a single trial dir restricts the work to just that SAE.
#
# Usage:
#   ./scripts/inference.sh submit   # queue all 5 inference jobs
#   ./scripts/inference.sh status   # snapshot status
#   ./scripts/inference.sh cancel   # scancel anything still queued/running
#
# Outputs land in:
#   /jet/.../data_storage/LM4_Results/saes/sae_inference/<model>/<trial>/
#     ├── feature_stats.pt
#     ├── buckets.pt
#     ├── dashboard_feature_*.html
#     └── index.html

set -euo pipefail

DATA=/jet/home/friedmae/data_storage/LM4_Results/Data/bioS_N-Bd_final_grid
GRID=/jet/home/friedmae/data_storage/LM4_Results/runResults/bioS_N-Bd_final_grid/20260520-134455/grid
ROOT=/jet/home/friedmae/data_storage/LM4_Results/saes/sae_runs

# job_name | trial_dir | model_dir
TRIALS=(
  "inf-L4-L0|$ROOT/grid-L4-H6/sweep-io37yc23/L0_mult8_l02_lr3e-05_ep50_n10000|$GRID/grid-L4-H6/final"
  "inf-L4-L1|$ROOT/grid-L4-H6/sweep-gie2mbe3/L1_mult16_l02_lr3e-05_ep50_n10000|$GRID/grid-L4-H6/final"
  "inf-L4-L2|$ROOT/grid-L4-H6/sweep-7t8t55s5/L2_mult16_l010_lr3e-05_ep50_n10000|$GRID/grid-L4-H6/final"
  "inf-L8-L3|$ROOT/grid-L8-H6/sweep-6r4rakfq/L3_mult16_l02_lr3e-05_ep50_n10000|$GRID/grid-L8-H6/final"
  "inf-L1-L0|$ROOT/grid-L1-H6/sweep-8sx9ovd5/L0_mult16_l02_lr3e-05_ep50_n10000|$GRID/grid-L1-H6/final"
)

submit_all() {
    echo "Submitting ${#TRIALS[@]} inference jobs..."
    for entry in "${TRIALS[@]}"; do
        IFS='|' read -r jname trial mdir <<< "$entry"
        if [[ ! -d "$trial/final" ]]; then
            echo "  $jname: SKIPPING — no final/ under $trial"
            continue
        fi
        out=$(sbatch -J "$jname" -t 6:00:00 submit_job_psc.sh saes/runInference.py \
            --sweep-dir "$trial" \
            --model-dir "$mdir" \
            --data-dir "$DATA")
        jid=${out##* }
        printf "  %-10s -> job %s\n" "$jname" "$jid"
    done
    echo
    echo "Monitor with: $0 status  (or squeue -u \$USER)"
}

show_status() {
    echo "============================================================"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    printf "%-10s  %-12s  %-10s  %s\n" "JOB" "STATE" "ELAPSED" "LATEST-JID"
    for entry in "${TRIALS[@]}"; do
        IFS='|' read -r jname _ _ <<< "$entry"
        local jid state elapsed
        jid=$(sacct -X --noheader --name="$jname" --format=JobID 2>/dev/null | tail -1 | tr -d ' ')
        if [[ -z "$jid" ]]; then
            state="never-submitted"; elapsed="-"
        else
            state=$(sacct -j "$jid" -X --noheader --format=State 2>/dev/null | head -1 | awk '{print $1}')
            elapsed=$(sacct -j "$jid" -X --noheader --format=Elapsed 2>/dev/null | head -1 | tr -d ' ')
        fi
        printf "%-10s  %-12s  %-10s  %s\n" "$jname" "$state" "$elapsed" "$jid"
    done
}

cancel_all() {
    for entry in "${TRIALS[@]}"; do
        IFS='|' read -r jname _ _ <<< "$entry"
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

case "${1:-}" in
    submit) submit_all ;;
    status) show_status ;;
    cancel) cancel_all ;;
    *)
        echo "Usage: $0 {submit|status|cancel}"
        exit 1
        ;;
esac
