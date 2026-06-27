#!/bin/bash
# Submit the edit-CLT training jobs (Method 1 + Method 2 variants) on PSC by
# setting env vars and sbatch-ing the shared scripts/train_clt_psc.sh.
#   bash clts/edit_clt/submit_edit_clt.sh            # full runs
#   bash clts/edit_clt/submit_edit_clt.sh --test     # cheap GPU smoke (N=1000, ep=3)
# Run from the Interp_LM4 repo root on a Bridges-2 login node.
set -euo pipefail

TEST=0; [ "${1:-}" = "--test" ] && TEST=1
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# ---- experiment paths (edit to match your manifest) ------------------------
REMOTE_BASE="/jet/home/friedmae/data_storage/LM4_Results"
EDITED_MODEL_NAME="${EDITED_MODEL_NAME:-grid-L4-H6-edit-p0-month-jul}"
EDITED_MODEL_DIR="${EDITED_MODEL_DIR:-$REMOTE_BASE/runResults/edited/$EDITED_MODEL_NAME}"
DATA_DIR="${DATA_DIR:-$REMOTE_BASE/Data/bioS_N-Bd_final_grid}"
BASE_CLT_DIR="${BASE_CLT_DIR:-$REMOTE_BASE/clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final}"
TARGET_CE="${TARGET_CE_RECOVERED:-}"   # from manifest target_stats; optional

export CLT_STORAGE_ROOT="$REMOTE_BASE"
COMMON=(--export=ALL --account=cis240072p --partition=GPU-shared --gres=gpu:1)

# Cheap-test overrides.
if [ "$TEST" = 1 ]; then N=1000; EP1=3; EP2=1; WALL="00:30:00"; TAG="-test";
else N=10000; EP1=50; EP2=5; WALL="04:00:00"; TAG=""; fi

submit() {  # name extra_env...
    local name="$1"; shift
    echo "submitting $name ..."
    env MODEL_NAME="$EDITED_MODEL_NAME" MODEL_DIR="$EDITED_MODEL_DIR" \
        DATA_DIR="$DATA_DIR" CONDA_ENV=lm4-ct SWEEP=0 \
        EXPANSION=16 L0=2 N_EXAMPLES="$N" CONTEXT_SIZE=512 "$@" \
        sbatch "${COMMON[@]}" --job-name="$name" --time="$WALL" \
               scripts/train_clt_psc.sh
}

# Method 1: from scratch, apricot config.
submit "clt-edit-m1${TAG}" LR=1e-4 EPOCHS="$EP1"

# Method 2 v2-basic: fine-tune, plateau-or-parity.
submit "clt-edit-m2basic${TAG}" LR=2e-5 EPOCHS="$EP2" \
    RESUME_FROM="$BASE_CLT_DIR" OUT_TAG="method2-v2-basic${TAG}" \
    PLATEAU_PATIENCE=3 PLATEAU_MIN_DELTA=0.01 EVAL_EVERY=100 \
    ${TARGET_CE:+TARGET_CE_RECOVERED="$TARGET_CE"}

# Method 2 v2-fixed: fine-tune, fixed short budget.
submit "clt-edit-m2fixed${TAG}" LR=2e-5 EPOCHS=2 \
    RESUME_FROM="$BASE_CLT_DIR" OUT_TAG="method2-v2-fixed${TAG}" EVAL_EVERY=100

echo "submitted. watch:  squeue -u \$USER"
