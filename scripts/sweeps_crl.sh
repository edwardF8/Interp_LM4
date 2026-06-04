#!/usr/bin/env bash
# Submit and monitor per-layer SAE-CRL sweeps on PSC (one parallel job per layer).
#
# Run from the repo root (~/Interp_LM4) on a Bridges-2 login node (after `wandb login`):
#
#   ./scripts/sweeps_crl.sh submit    # queue one job per layer in $LAYERS
#   ./scripts/sweeps_crl.sh status    # one-shot status snapshot
#   ./scripts/sweeps_crl.sh watch     # status, refreshing every 60s, until all jobs end
#   ./scripts/sweeps_crl.sh cancel    # scancel anything still queued
#
# Each job runs scripts/sweep_crl_psc.sbatch with LAYER=<n>, registering one wandb sweep
# (sae_CRL_sweep_<model>_L<n>) of 6 trials = 3 beta x 2 topk. Outputs land under
#   $STORAGE/sae_CRL_runs/<model_name>/sweep-<id>/L<n>_<trial>/final/
#
# Which layers: 0-indexed TransformerLens resid_post hooks (blocks.<n>.hook_resid_post).
# Default = "0 1" (blocks.0 and blocks.1, the first two layers). Override:  LAYERS="2 3" ./scripts/sweeps_crl.sh submit
set -euo pipefail

# ============================ EDIT HERE ============================
MODEL_NAME="${MODEL_NAME:-grid-L4-H6}"      # model dir name (resolved under $GRID in the .sbatch)
DATASET="${DATASET:-bioS_N-Bd_final_grid}"  # MUST be the dataset $MODEL_NAME was trained on (tokenizer + bios)
LAYERS="${LAYERS:-0 1}"                     # space list; 0-indexed resid_post hooks (blocks.<n>)

# --- sweep grid: the "different CRL interps" (comma lists; one job per layer runs this whole grid) ---
BETAS="${BETAS:-0.001,0.01}" #,0.1}"            # l_spB = l_spM  (graph sparsity; small -> denser M/B_tau)
TOPKS="${TOPKS:-25,100}"                    # latent L0 (active concepts per token)

# --- fixed (non-swept) training knobs ---
Z_DIM="${Z_DIM:-3072}"                      # dictionary size (8x d_model=384)
EPOCHS="${EPOCHS:-15}"
N_BIOS="${N_BIOS:-10000}"
LR="${LR:-0.01}"
NOISE_MODE="${NOISE_MODE:-lap}"             # lap | gau

# --- cluster ---
TIME="${TIME:-12:00:00}"                    # walltime per layer-job
STORAGE="${STORAGE:-/jet/home/friedmae/data_storage/LM4_Results}"   # where runs land (status counting)
# ===================================================================

SBATCH="scripts/sweep_crl_psc.sbatch"
# trials per layer = |BETAS| x |TOPKS|, derived from the lists above (no need to hand-edit)
EXPECTED_TRIALS=$(( $(awk -F, '{print NF}' <<<"$BETAS") * $(awk -F, '{print NF}' <<<"$TOPKS") ))

# ---------------------------------------------------------------------------
submit_all() {
    echo "Submitting layers [$LAYERS] for $MODEL_NAME: beta=[$BETAS] x topk=[$TOPKS] = $EXPECTED_TRIALS trials/layer"
    local exp="ALL,MODEL_NAME=$MODEL_NAME,BETAS=$BETAS,TOPKS=$TOPKS,Z_DIM=$Z_DIM,EPOCHS=$EPOCHS,N_BIOS=$N_BIOS,LR=$LR,NOISE_MODE=$NOISE_MODE"
    for L in $LAYERS; do
        out=$(sbatch -J "crl-L$L" -t "$TIME" --export="$exp,LAYER=$L" "$SBATCH")
        jid=${out##* }                  # sbatch prints "Submitted batch job 12345"
        printf "  crl-L%-3s -> job %s\n" "$L" "$jid"
    done
    echo
    echo "Run \`$0 watch\` to monitor."
}

# ---------------------------------------------------------------------------
show_status() {
    echo "============================================================"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')   model=$MODEL_NAME"
    echo "============================================================"
    printf "%-8s  %-12s  %-10s  %-7s  %s\n" "JOB" "STATE" "ELAPSED" "TRIALS" "LATEST-JID"
    local any_running=0
    for L in $LAYERS; do
        local jname="crl-L$L" jid state elapsed trials
        jid=$(sacct -X --noheader --name="$jname" --format=JobID 2>/dev/null | tail -1 | tr -d ' ')
        if [[ -z "$jid" ]]; then
            state="never-submitted"; elapsed="-"
        else
            state=$(sacct -j "$jid" -X --noheader --format=State 2>/dev/null | head -1 | awk '{print $1}')
            elapsed=$(sacct -j "$jid" -X --noheader --format=Elapsed 2>/dev/null | head -1 | tr -d ' ')
        fi
        # Count completed trial dirs for this layer (L<n>_ prefix), across any sweep folders
        trials=$(find "$STORAGE/sae_CRL_runs/$MODEL_NAME" -maxdepth 4 -type d -name final \
                      -path "*/L${L}_*" 2>/dev/null | wc -l | tr -d ' ')
        case "$state" in RUNNING|PENDING|REQUEUED|CONFIGURING) any_running=1 ;; esac
        printf "%-8s  %-12s  %-10s  %-7s  %s\n" \
            "$jname" "$state" "$elapsed" "${trials}/${EXPECTED_TRIALS}" "$jid"
    done
    return $any_running
}

# ---------------------------------------------------------------------------
watch_loop() {
    while true; do
        clear || true
        if show_status; then
            echo; echo "All jobs in terminal state."; break
        fi
        echo; echo "Refresh in 60s, Ctrl-C to exit watch (jobs keep running)."
        sleep 60
    done
}

# ---------------------------------------------------------------------------
cancel_all() {
    echo "Cancelling running/pending crl-L* jobs for layers [$LAYERS]..."
    for L in $LAYERS; do
        local jname="crl-L$L" jids
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
    *) echo "Usage: $0 {submit|status|watch|cancel}   (LAYERS='$LAYERS', MODEL_NAME='$MODEL_NAME')"; exit 1 ;;
esac
