#!/bin/bash
# Self-data fine-tune CONTROL for the edit-CLT experiment: continue training the
# "apricot" CLT (clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final)
# on its OWN original model (grid-L4-H6) + original data (bioS_N-Bd_final_grid),
# with the exact recipe the Method-2 edit fine-tunes used on the *edited* model
# (FactEditingLM4/edit_clt/submit_edit_clt.sh). Any drift measured here is due to
# continued training alone — the baseline against which Method-2's edit-induced
# drift is read. Saves to a NEW OUT_TAG; the apricot CLT is never overwritten.
#
#   bash finetuning/submit_finetune_clt.sh              # basic: plateau-stopped, <=5 epochs
#   bash finetuning/submit_finetune_clt.sh --fixed      # fixed short budget: 2 epochs
#   bash finetuning/submit_finetune_clt.sh --test       # cheap GPU smoke (N=1000, ep=1, -test suffix)
#   bash finetuning/submit_finetune_clt.sh --dry-run    # print the sbatch cmd; execute nothing
#   ANCHOR_LAMBDA=1e-4 bash finetuning/submit_finetune_clt.sh  # + L2 pull toward apricot weights
#   WRITE_DASHBOARDS=0 bash finetuning/submit_finetune_clt.sh  # train only, no dashboards
#
# Flags compose (e.g. --dry-run --test --fixed). Runs from anywhere (cd's to the
# repo root); submit on a Bridges-2 login node. See finetuning/README.md.
set -euo pipefail

# ---- flags ------------------------------------------------------------------
VARIANT=basic; TEST=0; DRY=0
for arg in "$@"; do
    case "$arg" in
        --fixed)   VARIANT=fixed ;;
        --test)    TEST=1 ;;
        --dry-run) DRY=1 ;;
        *) echo "unknown flag: $arg (known: --fixed --test --dry-run)" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# ---- experiment paths (each env-overridable) ---------------------------------
REMOTE_BASE="${REMOTE_BASE:-/jet/home/friedmae/data_storage/LM4_Results}"
MODEL_NAME="${MODEL_NAME:-grid-L4-H6}"   # the ORIGINAL model — that is the point of the control
MODEL_DIR="${MODEL_DIR:-$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid/grid-L4-H6/final}"
DATA_DIR="${DATA_DIR:-$REMOTE_BASE/Data/bioS_N-Bd_final_grid}"
# The apricot CLT: resumed from (RESUME_FROM), written elsewhere (OUT_TAG).
BASE_CLT_DIR="${BASE_CLT_DIR:-$REMOTE_BASE/clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final}"

export CLT_STORAGE_ROOT="$REMOTE_BASE"
COMMON=(--export=ALL --account=cis240072p --partition=GPU-shared --gres=gpu:1)

# ---- variant knobs (Method-2 recipe, pointed at the ORIGINAL model) ----------
# basic = procedure-matched to Method-2 v2-basic (plateau-or-parity early stop);
# fixed = step-matched to Method-2 v2-fixed (fixed 2-epoch budget, no early stop).
if [ "$VARIANT" = fixed ]; then
    OUT_TAG=apricot-finetune-fixed; JOB_NAME=clt-apricot-ft-fixed; EPOCHS=2
    PLATEAU=()
else
    OUT_TAG=apricot-finetune-basic; JOB_NAME=clt-apricot-ft-basic; EPOCHS=5
    PLATEAU=(PLATEAU_PATIENCE=3 PLATEAU_MIN_DELTA=0.01)
fi

# ---- cheap-test overrides (-test suffix keeps the real artifacts safe) -------
N_EXAMPLES=10000; WALL="06:00:00"; DASH_NP_DEFAULT=1000
if [ "$TEST" = 1 ]; then
    N_EXAMPLES=1000; EPOCHS=1; WALL="01:00:00"; DASH_NP_DEFAULT=100
    OUT_TAG="${OUT_TAG}-test"; JOB_NAME="${JOB_NAME}-test"
fi

# ---- feature dashboards (built post-training in the same job; opt out with
# WRITE_DASHBOARDS=0). DASH_SCAN default inherits OUT_TAG's -test suffix. -------
DASH="${WRITE_DASHBOARDS:-1}"
FEATURES_ROOT="${FEATURES_ROOT:-$REMOTE_BASE/clt_features}"
DASH_SCAN="${DASH_SCAN:-$MODEL_NAME-$OUT_TAG}"
DASH_NP="${DASH_N_PEOPLE:-$DASH_NP_DEFAULT}"
DASH_DEV="${DASH_DEVICE:-cuda}"

# ---- compose the job ----------------------------------------------------------
# CONDA_ENV=lm4-ct ALWAYS: the rebuilt `lm4` env's cu130 torch reports cuda=False.
# WANDB_NAME / WANDB_RUN_GROUP: trainCLT.py passes no name and group=None for
# non-robust standalone runs, so these env vars take effect (same mechanism as
# train_clt_psc.sh's SLURM-array WANDB_RUN_GROUP).
# ANCHOR_LAMBDA is threaded ONLY when set in the caller's environment: default
# unset = off, matching the Method-2 recipe; e.g. ANCHOR_LAMBDA=1e-4 adds an L2
# pull toward the apricot weights for even less drift.
CMD=(env
    MODEL_NAME="$MODEL_NAME" MODEL_DIR="$MODEL_DIR" DATA_DIR="$DATA_DIR"
    CONDA_ENV=lm4-ct SWEEP=0
    EXPANSION=16 L0=2 N_EXAMPLES="$N_EXAMPLES" CONTEXT_SIZE=512
    LR=2e-5 EPOCHS="$EPOCHS" EVAL_EVERY=100
    RESUME_FROM="$BASE_CLT_DIR" OUT_TAG="$OUT_TAG"
    ${PLATEAU[@]+"${PLATEAU[@]}"}
    ${ANCHOR_LAMBDA:+ANCHOR_LAMBDA="$ANCHOR_LAMBDA"}
    WANDB_NAME="$MODEL_NAME/$OUT_TAG" WANDB_RUN_GROUP=clt-finetune-control
    WRITE_DASHBOARDS="$DASH" FEATURES_ROOT="$FEATURES_ROOT"
    DASH_SCAN="$DASH_SCAN" DASH_N_PEOPLE="$DASH_NP" DASH_DEVICE="$DASH_DEV"
    sbatch "${COMMON[@]}" --job-name="$JOB_NAME" --time="$WALL"
    scripts/train_clt_psc.sh)

# Where trainCLT.py will land the result: trial dir from trial_name() =
# mult{expansion}_l0{l0:g}_lr{lr:g}_ep{epochs}_n{n}; lr 2e-5 renders as lr2e-05.
OUT_DIR="$REMOTE_BASE/clt_runs/$MODEL_NAME/$OUT_TAG/mult16_l02_lr2e-05_ep${EPOCHS}_n${N_EXAMPLES}/final"
echo "CLT will land at: $OUT_DIR"

if [ "$DRY" = 1 ]; then
    echo "[dry-run] would execute:"
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

echo "submitting $JOB_NAME ..."
"${CMD[@]}"    # sbatch prints:  Submitted batch job <jobid>
echo "watch:   squeue -u \$USER    (output: logs/${JOB_NAME}-<jobid>.out)"
echo "dashboards (WRITE_DASHBOARDS=$DASH): $FEATURES_ROOT/$DASH_SCAN"
