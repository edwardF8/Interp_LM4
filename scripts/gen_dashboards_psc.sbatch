#!/bin/bash
#SBATCH --job-name=gen-dashboards
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
# SELF-CONTAINED GPU job: generates CLT attribution feature dashboards on PSC.
# Does env setup + a fail-fast path/CLT preflight + the generation, all in the
# one job shell (so `conda activate` works -- no subshell / `conda init` traps).
#
# Run from the repo root on PSC:
#   sbatch scripts/gen_dashboards_psc.sbatch            # moderate (--n-people 1000, recommended)
#   sbatch scripts/gen_dashboards_psc.sbatch validate   # smoke   (--n-people 100)
#   sbatch scripts/gen_dashboards_psc.sbatch full        # all people (bump --time if needed)
#
# Override any path by editing the config block below, or at submit time:
#   sbatch --export=ALL,CLT_DIR=/path/to/clt/final scripts/gen_dashboards_psc.sbatch moderate
set -euo pipefail

# ---- config (edit here, or override via --export) --------------------------
REMOTE_BASE="${REMOTE_BASE:-/jet/home/friedmae/data_storage/LM4_Results}"
MODEL_DIR="${MODEL_DIR:-$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid/grid-L4-H6/final}"
CLT_DIR="${CLT_DIR:-$REMOTE_BASE/clt_runs/grid-L4-H6/sweep-cfbp6man/mult16_l02_lr0.0001_ep50_n10000/final}"
DATA_DIR="${DATA_DIR:-$REMOTE_BASE/Data/bioS_N-Bd_final_grid}"
SCAN_NAME="${SCAN_NAME:-grid-L4-H6}"
FEATURES_ROOT="${FEATURES_ROOT:-$REMOTE_BASE/clt_features}"
CONDA_ENV="${CONDA_ENV:-lm4}"
N_PER_PERSON="${N_PER_PERSON:-2}"
CONTEXT_SIZE="${CONTEXT_SIZE:-64}"

# Mode -> how many people in the corpus.
MODE="${1:-moderate}"
case "$MODE" in
    validate) PEOPLE=(--n-people 100) ;;
    moderate) PEOPLE=(--n-people 1000) ;;
    full)     PEOPLE=() ;;
    *) echo "usage: sbatch $0 {validate|moderate|full}" >&2; exit 1 ;;
esac

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# ---- environment (mirrors Training_On_LM4/setup_job_psc.sh) ----------------
# Don't trust conda's active-env state inherited from the submitting shell: an
# active env leaks CONDA_SHLVL/CONDA_PREFIX via --export=ALL, and the conda
# deactivate below then chokes ("Run 'conda init' before 'conda deactivate'").
unset CONDA_SHLVL CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
module purge
module load cuda
module load anaconda3
# module load puts conda on PATH but does NOT define the `conda` shell function:
eval "$(conda shell.bash hook 2>/dev/null)" || {
    _base="$(conda info --base 2>/dev/null)"
    [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ] && source "$_base/etc/profile.d/conda.sh"
}
conda deactivate 2>/dev/null || true     # drop any env inherited from the submit shell
conda activate "$CONDA_ENV"
echo "env: ${CONDA_DEFAULT_ENV:-none}   python: $(which python)"
export PYTHONUNBUFFERED=1

# ---- preflight: fail fast before the GPU-heavy pass ------------------------
echo "=== preflight ==="
fail=0
for p in "$MODEL_DIR/config.json" "$CLT_DIR/config.yaml" "$CLT_DIR/W_enc_0.safetensors" \
         "$DATA_DIR/old_to_new.json" "$DATA_DIR/people.json" clts/gen_feature_dashboards.py; do
    [ -e "$p" ] || { echo "  MISSING: $p" >&2; fail=1; }
done
mkdir -p "$FEATURES_ROOT/$SCAN_NAME" || { echo "  FEATURES_ROOT not writable: $FEATURES_ROOT" >&2; fail=1; }
python - "$CLT_DIR" "$MODEL_DIR" <<'PY' || fail=1
import sys
for m in ("torch", "transformer_lens", "transformers", "safetensors", "numpy", "yaml"):
    __import__(m)
import torch, json
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
sys.path.insert(0, ".")
from clts.clt import CrossLayerTranscoder
clt = CrossLayerTranscoder.load_from_dir(sys.argv[1])
cfg = json.load(open(f"{sys.argv[2]}/config.json"))
print(f"  clt: {clt.n_layers} layers, d_transcoder={clt.d_transcoder}; "
      f"model: {cfg.get('num_hidden_layers')} layers, vocab={cfg.get('vocab_size')}")
assert cfg.get("num_hidden_layers") == clt.n_layers, "model/clt layer mismatch"
PY
[ "$fail" = 0 ] || { echo "PREFLIGHT FAILED -- not generating." >&2; exit 1; }
echo "preflight OK"

# ---- generate --------------------------------------------------------------
echo "=== generating dashboards (mode=$MODE) on $(hostname) ==="
date
nvidia-smi || true
python -u clts/gen_feature_dashboards.py \
    --model-dir "$MODEL_DIR" \
    --clt-dir   "$CLT_DIR" \
    --data-dir  "$DATA_DIR" \
    --scan-name "$SCAN_NAME" \
    --features-root "$FEATURES_ROOT" \
    --device cuda \
    --n-per-person "$N_PER_PERSON" \
    --context-size "$CONTEXT_SIZE" \
    "${PEOPLE[@]}"

n="$( { ls "$FEATURES_ROOT/$SCAN_NAME"/*.json 2>/dev/null || true; } | wc -l | tr -d ' ')"
echo "Done: wrote $n dashboards to $FEATURES_ROOT/$SCAN_NAME"
echo "Next (Mac): ./scripts/sync_from_psc.sh  then build + serve (clts/README_attribution.md)"
echo "Finished: $(date)"
