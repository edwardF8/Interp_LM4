#!/bin/bash
#SBATCH --job-name=writefeatures
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --account=cis240072p
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egfriedm@andrew.cmu.edu
#SBATCH --array=0-15
#
# SELF-CONTAINED GPU array: writeFeatures feature-in-node rank sweep over EVERY person
# on PSC. Each array task runs ONE shard (work[shard_index::num_shards]); after the
# whole array finishes you run the merge ONCE. No attribution cache is written, so the
# output is ~100-150 MB of aggregates + reports (NOT the ~270 GB an interactive cache
# would cost). storage_root() resolves to $REMOTE_BASE on PSC, so partials/reports land
# under $REMOTE_BASE/clt_feature_explorer/<scan>/hpc/.
#
# Run from the repo root on PSC:
#   sbatch scripts/writefeatures_psc.sbatch
#   NUM_SHARDS=32 sbatch --array=0-31 scripts/writefeatures_psc.sbatch        # wider
#   EXTRA_ARGS="--templates 0,5,12" sbatch scripts/writefeatures_psc.sbatch   # fewer prompts
#   sbatch --export=ALL,CLT_DIR=/path/final scripts/writefeatures_psc.sbatch  # other CLT
#
# CALIBRATE FIRST (model is tiny -> GPU speedup unknown until measured): on a GPU node,
#   python -u clts/run_writefeatures_hpc.py --model-dir "$MODEL_DIR" --clt-dir "$CLT_DIR" \
#       --data-dir "$DATA_DIR" --scan-name "$SCAN_NAME" \
#       --months August --limit-people 50 --device cuda --batch-size 1024 --progress-every 10
#   per-graph time = 1 / (printed people/s * 46); pick NUM_SHARDS so a shard fits --time.
set -euo pipefail

# ---- config (edit here, or override via env / --export) --------------------
REMOTE_BASE="${REMOTE_BASE:-/jet/home/friedmae/data_storage/LM4_Results}"
MODEL_DIR="${MODEL_DIR:-$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid/grid-L4-H6/final}"
CLT_DIR="${CLT_DIR:-$REMOTE_BASE/clt_runs/grid-L4-H6/sweep-cfbp6man/mult16_l02_lr0.0001_ep50_n10000/final}"
DATA_DIR="${DATA_DIR:-$REMOTE_BASE/Data/bioS_N-Bd_final_grid}"
SCAN_NAME="${SCAN_NAME:-grid-L4-H6}"
CONDA_ENV="${CONDA_ENV:-lm4-ct}"   # dedicated circuit-tracer env (see scripts/setup_ct_env_psc.sh)
NUM_SHARDS="${NUM_SHARDS:-16}"     # MUST equal the --array size (default --array=0-15)
BATCH_SIZE="${BATCH_SIZE:-1024}"   # bigger batch fills the GPU
EXTRA_ARGS="${EXTRA_ARGS:-}"       # e.g. "--templates 0,5,12" or "--limit-people 500" or "--months August"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# ---- environment (mirrors gen_dashboards_psc.sbatch) -----------------------
# Use the env's Python DIRECTLY -- no `conda activate` (it breaks under SLURM:
# "Run 'conda init' before 'conda deactivate'"). Override PY if your env is
# elsewhere than ~/.conda/envs/$CONDA_ENV.
module purge
module load cuda
PY="${PY:-$HOME/.conda/envs/$CONDA_ENV/bin/python}"
[ -x "$PY" ] || { echo "ERROR: no python at $PY (is conda env '$CONDA_ENV' created?)" >&2; exit 1; }
echo "python: $PY"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$(pwd)}${PYTHONPATH:+:$PYTHONPATH}"

# ---- preflight: fail fast before booking GPU time --------------------------
echo "=== preflight ==="
fail=0
for p in "$MODEL_DIR/config.json" "$CLT_DIR/config.yaml" "$CLT_DIR/W_enc_0.safetensors" \
         "$DATA_DIR/old_to_new.json" "$DATA_DIR/people.json" clts/run_writefeatures_hpc.py; do
    [ -e "$p" ] || { echo "  MISSING: $p" >&2; fail=1; }
done
echo "  model:   $MODEL_DIR"
echo "  clt:     $CLT_DIR"
echo "  dataset: $DATA_DIR"
"$PY" - <<'PY' || fail=1
import torch
import clts.writefeatures as wf
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  writefeatures OK ({len([f for f in dir(wf) if f=='run_hypothesis'])} entrypoint)")
PY
[ "$fail" = 0 ] || { echo "PREFLIGHT FAILED -- not launching." >&2; exit 1; }
echo "preflight OK"

# ---- run this shard --------------------------------------------------------
SHARD="${SLURM_ARRAY_TASK_ID:-${SHARD:-0}}"
echo "=== writeFeatures sweep: shard $SHARD/$NUM_SHARDS on $(hostname) ==="
date
nvidia-smi || true
"$PY" -u clts/run_writefeatures_hpc.py \
    --model-dir "$MODEL_DIR" \
    --clt-dir   "$CLT_DIR" \
    --data-dir  "$DATA_DIR" \
    --scan-name "$SCAN_NAME" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD" \
    --device cuda \
    --batch-size "$BATCH_SIZE" \
    --skip-existing \
    $EXTRA_ARGS

echo "Finished shard $SHARD: $(date)"

# ---- merge (run ONCE after the whole array finishes) -----------------------
# On a login node (conda activate $CONDA_ENV) or a tiny CPU job:
#   python -u clts/merge_writefeatures_hpc.py --scan-name "$SCAN_NAME"
# Or chain it as a dependency:
#   AID=$(sbatch --parsable scripts/writefeatures_psc.sbatch)
#   sbatch --dependency=afterok:$AID --account=cis240072p --partition=RM-shared --time=00:30:00 \
#          --wrap="module load anaconda3; conda activate $CONDA_ENV; \
#                  python -u clts/merge_writefeatures_hpc.py --scan-name $SCAN_NAME"
