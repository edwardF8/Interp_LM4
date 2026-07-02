#!/bin/bash
#SBATCH --job-name=train-clt
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --account=cis240072p
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egfriedm@andrew.cmu.edu
#
# Trains a cross-layer transcoder (CLT) on a base Llama checkpoint via
# clts/trainCLT.py. Self-contained: env setup + fail-fast preflight + training.
# Submit from the repo root on a Bridges-2 login node:   sbatch scripts/train_clt_psc.sh
#
set -euo pipefail

# ############################################################################
# ###  EDIT YOUR HYPERPARAMETERS HERE  #######################################
# ############################################################################

# --- what to train on -------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-grid-L1-H6}"   # base model identifier (and output subdir); override via env (see submit_clt_sweeps_grid.sh)
REMOTE_BASE="/jet/home/friedmae/data_storage/LM4_Results"
GRID="${GRID:-$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid}"
MODEL_DIR="${MODEL_DIR:-$GRID/$MODEL_NAME/final}"               # HF checkpoint dir
DATA_DIR="${DATA_DIR:-$REMOTE_BASE/Data/bioS_N-Bd_final_grid}" # has people.json + old_to_new.json

# Optional: RobustnessTest manifest. When set, limited people's CLT training
# bios render only from their allowed templates -- matching the robust corpus
# the base model was trained on (see submit_clt_robust_grid.sh).
ROBUSTNESS_MANIFEST="${ROBUSTNESS_MANIFEST:-}"
ROBUST_ARGS=()
[ -n "$ROBUSTNESS_MANIFEST" ] && ROBUST_ARGS=(--robustness-manifest "$ROBUSTNESS_MANIFEST")

# Optional: resolve MODEL_DIR at RUNTIME from a glob (newest match wins).
# Lets a driver submit dependency-held jobs before the checkpoint exists --
# e.g. CLTs on the robust grid, whose runs/<INVOCATION>/ dir isn't known
# until the training job actually starts (see submit_clt_robust_grid.sh).
if [ -n "${MODEL_DIR_GLOB:-}" ]; then
    MODEL_DIR=$(ls -1dt $MODEL_DIR_GLOB 2>/dev/null | head -1 || true)
    [ -n "$MODEL_DIR" ] || { echo "ERROR: MODEL_DIR_GLOB matched nothing: $MODEL_DIR_GLOB" >&2; exit 1; }
    echo "MODEL_DIR (newest glob match) = $MODEL_DIR"
fi

# --- CLT activation sites ({layer} is filled in for EVERY block) ------------
# A CLT spans all layers at once; there is no single layer to pick. Change the
# hook SITE here (e.g. hook_resid_pre / hook_resid_post) to move where it reads/writes.
ENC_HOOK="blocks.{layer}.hook_resid_mid"   # encoder input
DEC_HOOK="blocks.{layer}.hook_mlp_out"     # decoder target

# --- mode: pick ONE of the two sections below -------------------------------
SWEEP="${SWEEP:-1}"   # 0 = single run (SECTION A) ; 1 = wandb grid sweep (SECTION B)
CONTEXT_SIZE="${CONTEXT_SIZE:-512}"    # (applies to both)

# === SECTION A: single-run hyperparameters (used when SWEEP=0) ===============
# Env-overridable so a driver can submit single runs, incl. SLURM-array grids.
N_EXAMPLES="${N_EXAMPLES:-1000}"     # 1000 = quick test; 10000 = full
EPOCHS="${EPOCHS:-20}"               # 3 = quick test; 30 = full
EXPANSION="${EXPANSION:-16}"         # d_transcoder = EXPANSION * d_model
L0="${L0:-5.0}"                      # sparsity (L0) coefficient
LR="${LR:-5e-5}"                     # learning rate

# --- edit-CLT add-on (opt-in; unset -> legacy behavior) ---------------------
RESUME_FROM="${RESUME_FROM:-}"            # final/ dir of a CLT to fine-tune from
OUT_TAG="${OUT_TAG:-}"                     # run-path folder (variant disambiguator)
TARGET_CE_RECOVERED="${TARGET_CE_RECOVERED:-}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-}"
PLATEAU_MIN_DELTA="${PLATEAU_MIN_DELTA:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
ANCHOR_LAMBDA="${ANCHOR_LAMBDA:-}"
EVAL_PERSON="${EVAL_PERSON:-}"            # 0-based people.json index: eval ce_recovered on THIS person only (local-parity stop)
ADDON_ARGS=()
[ -n "$RESUME_FROM" ]         && ADDON_ARGS+=(--resume-from "$RESUME_FROM")
[ -n "$OUT_TAG" ]             && ADDON_ARGS+=(--out-tag "$OUT_TAG")
[ -n "$TARGET_CE_RECOVERED" ] && ADDON_ARGS+=(--target-ce-recovered "$TARGET_CE_RECOVERED")
[ -n "$PLATEAU_PATIENCE" ]    && ADDON_ARGS+=(--plateau-patience "$PLATEAU_PATIENCE")
[ -n "$PLATEAU_MIN_DELTA" ]   && ADDON_ARGS+=(--plateau-min-delta "$PLATEAU_MIN_DELTA")
[ -n "$EVAL_EVERY" ]          && ADDON_ARGS+=(--eval-every "$EVAL_EVERY")
[ -n "$ANCHOR_LAMBDA" ]       && ADDON_ARGS+=(--anchor-lambda "$ANCHOR_LAMBDA")
[ -n "$EVAL_PERSON" ]         && ADDON_ARGS+=(--eval-person "$EVAL_PERSON")

# === SECTION B: sweep grid (used when SWEEP=1) ==============================
# OVERRIDE_SWEEP=0 -> run the baked-in grid (expansion {4,8,16,32} x l0 {1,2,5},
# lr 1e-4, n_examples 50000, epochs 10) and IGNORE the values below.
# OVERRIDE_SWEEP=1 -> use the values below. Lists are space-separated.
# Each is env-overridable so a driver (submit_clt_sweeps_grid.sh) can set a
# per-model grid without editing this file.
OVERRIDE_SWEEP="${OVERRIDE_SWEEP:-0}"
SWEEP_EXPANSION="${SWEEP_EXPANSION:-4 8 16 32}"
SWEEP_L0="${SWEEP_L0:-1 2 5}"
SWEEP_LR="${SWEEP_LR:-1e-4}"
SWEEP_N_EXAMPLES="${SWEEP_N_EXAMPLES:-20000}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-20}"

# === SLURM array mode: one task per (expansion, l0) grid point ==============
# A driver can submit `sbatch --array=0-N ... GRID_EXPANSION=".." GRID_L0=".."`
# to run a grid as concurrent SINGLE runs (no wandb sweep). Each task derives its
# own (expansion, l0) from SLURM_ARRAY_TASK_ID, iterating expansion-major
# (i -> EXP[i / nL0], L0[i % nL0]). Forces SWEEP=0 and groups the runs in wandb.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${GRID_EXPANSION:-}" ]; then
    read -r -a _EXPS <<< "$GRID_EXPANSION"
    read -r -a _L0S  <<< "${GRID_L0:?GRID_L0 required alongside GRID_EXPANSION}"
    _nL0=${#_L0S[@]}
    _i=$SLURM_ARRAY_TASK_ID
    EXPANSION="${_EXPS[$(( _i / _nL0 ))]}"
    L0="${_L0S[$(( _i % _nL0 ))]}"
    SWEEP=0
    export WANDB_RUN_GROUP="grid-array-$MODEL_NAME"
    echo "array task $_i -> expansion=$EXPANSION l0=$L0 (wandb group=$WANDB_RUN_GROUP)"
fi

# --- environment ------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-lm4}"   # overridable, like the other *_psc scripts

# ############################################################################
# ###  machinery below — you shouldn't need to touch this  ###################
# ############################################################################

# CLT trainer resolves its output root from this env var (see trainCLT.py).
export CLT_STORAGE_ROOT="$REMOTE_BASE"

# If overriding the sweep grid, hand the values to build_sweep_config() via env.
# When OVERRIDE_SWEEP=0 these stay unset -> trainCLT.py uses its baked-in grid.
if [ "$SWEEP" = 1 ] && [ "$OVERRIDE_SWEEP" = 1 ]; then
    export CLT_SWEEP_EXPANSION="$SWEEP_EXPANSION"
    export CLT_SWEEP_L0="$SWEEP_L0"
    export CLT_SWEEP_LR="$SWEEP_LR"
    export CLT_SWEEP_N_EXAMPLES="$SWEEP_N_EXAMPLES"
    export CLT_SWEEP_EPOCHS="$SWEEP_EPOCHS"
    echo "sweep grid OVERRIDDEN: expansion=[$SWEEP_EXPANSION] l0=[$SWEEP_L0] lr=$SWEEP_LR n_examples=$SWEEP_N_EXAMPLES epochs=$SWEEP_EPOCHS"
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Use the env's Python DIRECTLY -- no `conda activate` (it breaks under SLURM:
# "Run 'conda init' before 'conda deactivate'"). Override PY if your env is
# elsewhere than ~/.conda/envs/$CONDA_ENV.
# NO `module purge`: on a job from an active conda shell it unloads the inherited
# anaconda module and fires a `conda deactivate` -> error. Just load cuda.
module load cuda 2>/dev/null || true
PY="${PY:-$HOME/.conda/envs/$CONDA_ENV/bin/python}"
[ -x "$PY" ] || { echo "ERROR: no python at $PY (is conda env '$CONDA_ENV' created?)" >&2; exit 1; }
echo "python: $PY"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$(pwd)}${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${LOCAL:-}" ]; then
    export HF_HOME="$LOCAL/hf_cache"; export WANDB_CACHE_DIR="$LOCAL/wandb_cache"
    mkdir -p "$HF_HOME" "$WANDB_CACHE_DIR"
fi

# ---- preflight: fail fast before booking GPU time --------------------------
echo "=== preflight ==="
fail=0
for p in "$MODEL_DIR/config.json" "$DATA_DIR/people.json" "$DATA_DIR/old_to_new.json" \
         clts/trainCLT.py; do
    [ -e "$p" ] || { echo "  MISSING: $p" >&2; fail=1; }
done
if [ -n "$ROBUSTNESS_MANIFEST" ] && [ ! -e "$ROBUSTNESS_MANIFEST" ]; then
    echo "  MISSING: $ROBUSTNESS_MANIFEST" >&2; fail=1
fi
if [ -n "$RESUME_FROM" ] && [ ! -e "$RESUME_FROM/config.yaml" ]; then
    echo "  MISSING (resume): $RESUME_FROM/config.yaml" >&2; fail=1
fi
if [ -n "$EVAL_PERSON" ] && ! [[ "$EVAL_PERSON" =~ ^[0-9]+$ ]]; then
    echo "  EVAL_PERSON must be a non-negative integer (people.json index): $EVAL_PERSON" >&2; fail=1
fi
# trainCLT.py always calls wandb.init (sweeps AND single runs), so always check.
if ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "  wandb not authenticated: run \`wandb login\`, or set WANDB_API_KEY" >&2
    fail=1
fi
"$PY" - "$MODEL_DIR" <<'PY' || fail=1
import sys, json
for m in ("torch", "transformer_lens", "safetensors", "wandb"):
    __import__(m)
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
print(f"  model: {cfg.get('num_hidden_layers')} layers, d_model={cfg.get('hidden_size')}")
PY
[ "$fail" = 0 ] || { echo "PREFLIGHT FAILED -- not launching." >&2; exit 1; }
echo "preflight OK"

# ---- run: parallel-sweep AGENT mode ----------------------------------------
# If AGENT_SWEEP_ID is set (by submit_clt_sweeps_grid.sh), this task is a wandb
# agent pulling trials from an already-registered sweep. AGENT_COUNT trials, then
# exit. Hyperparameters come from the sweep, not from EXPANSION/L0 here.
if [ -n "${AGENT_SWEEP_ID:-}" ]; then
    echo "=== CLT sweep AGENT on $(hostname): sweep=$AGENT_SWEEP_ID count=${AGENT_COUNT:-1} ==="
    date; nvidia-smi || true
    "$PY" -u clts/trainCLT.py \
        --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --model-name "$MODEL_NAME" \
        --enc-hook-template "$ENC_HOOK" --dec-hook-template "$DEC_HOOK" \
        ${ROBUST_ARGS[@]+"${ROBUST_ARGS[@]}"} \
        --agent "$AGENT_SWEEP_ID" --count "${AGENT_COUNT:-1}"
    echo "Finished agent: $(date)"
    echo "Sweep runs under: $CLT_STORAGE_ROOT/clt_runs/$MODEL_NAME/sweep-$AGENT_SWEEP_ID/"
    exit 0
fi

# ---- run -------------------------------------------------------------------
mode="single run"; [ "$SWEEP" = 1 ] && mode="sweep (expansion x l0)"
echo "=== CLT $mode on $(hostname) ==="
echo "    enc=$ENC_HOOK  dec=$DEC_HOOK  n_examples=$N_EXAMPLES  epochs=$EPOCHS"
date
nvidia-smi || true

cmd=("$PY" -u clts/trainCLT.py
    --model-dir "$MODEL_DIR"
    --data-dir  "$DATA_DIR"
    --model-name "$MODEL_NAME"
    --enc-hook-template "$ENC_HOOK"
    --dec-hook-template "$DEC_HOOK"
    --expansion "$EXPANSION"
    --l0 "$L0"
    --lr "$LR"
    --epochs "$EPOCHS"
    --context-size "$CONTEXT_SIZE"
    --n-examples "$N_EXAMPLES"
    ${ADDON_ARGS[@]+"${ADDON_ARGS[@]}"})
[ "$SWEEP" = 1 ] && cmd+=(--sweep)
[ -n "$ROBUSTNESS_MANIFEST" ] && cmd+=(--robustness-manifest "$ROBUSTNESS_MANIFEST")

"${cmd[@]}"

# ---- optional: feature dashboards for the CLT we just trained (opt-in) ------
# WRITE_DASHBOARDS=1 (single runs only) -> right after training, build the
# circuit-tracer viewer's per-feature dashboards on THIS model + the CLT just
# written, into $FEATURES_ROOT/$DASH_SCAN. gen_feature_dashboards.py needs only
# the training env (no circuit_tracer), so it runs in this same job/env. Used by
# the edit-CLT pipeline (submit_edit_clt.sh) so the m1/m2 CLTs get the bottom-panel
# dashboards Notebook 2's viewer shows. A failure here does NOT fail the job --
# the CLT is already saved; rerun gen_feature_dashboards.py to retry.
if [ "${WRITE_DASHBOARDS:-0}" = 1 ] && [ "$SWEEP" = 0 ]; then
    DASH_SCAN="${DASH_SCAN:-$MODEL_NAME}"
    FEATURES_ROOT="${FEATURES_ROOT:-$CLT_STORAGE_ROOT/clt_features}"
    DASH_N_PEOPLE="${DASH_N_PEOPLE:-1000}"   # corpus size (cost knob); empty = all people
    DASH_DEVICE="${DASH_DEVICE:-cuda}"
    # The just-written CLT: clt_runs/<model>/<out_tag|standalone>/<trial>/final (newest).
    _clt_final=$(ls -1dt "$CLT_STORAGE_ROOT/clt_runs/$MODEL_NAME/${OUT_TAG:-standalone}"/*/final 2>/dev/null | head -1 || true)
    if [ -z "$_clt_final" ]; then
        echo "WARNING: WRITE_DASHBOARDS=1 but no trained CLT under" \
             "$CLT_STORAGE_ROOT/clt_runs/$MODEL_NAME/${OUT_TAG:-standalone}/*/final -- skipping dashboards" >&2
    else
        echo "=== feature dashboards: $_clt_final -> $FEATURES_ROOT/$DASH_SCAN (device=$DASH_DEVICE) ==="
        _people_args=(); [ -n "$DASH_N_PEOPLE" ] && _people_args=(--n-people "$DASH_N_PEOPLE")
        "$PY" -u clts/gen_feature_dashboards.py \
            --model-dir "$MODEL_DIR" --clt-dir "$_clt_final" --data-dir "$DATA_DIR" \
            --scan-name "$DASH_SCAN" --features-root "$FEATURES_ROOT" \
            --device "$DASH_DEVICE" "${_people_args[@]}" \
            && echo "dashboards under: $FEATURES_ROOT/$DASH_SCAN" \
            || echo "WARNING: dashboard generation failed (CLT is saved; rerun gen_feature_dashboards.py)" >&2
    fi
fi

echo "Finished: $(date)"
echo "Runs under: $CLT_STORAGE_ROOT/clt_runs/$MODEL_NAME/"
