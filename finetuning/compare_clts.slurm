#!/bin/bash
#SBATCH --job-name=clt-compare
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:20:00
#SBATCH --account=cis240072p
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egfriedm@andrew.cmu.edu
#
# Compare the apricot base CLT against its fine-tuned (RESUME_FROM) descendant.
# Read-only analysis (finetuning/compare_clts.py) — imports only first-party
# clts/ + util/, does no model editing. Submit from the repo root:
#
#     sbatch finetuning/compare_clts.slurm
#
# Overridable env: CONDA_ENV, N_ROWS, N_EVAL_ROWS, CHUNK_ROWS, SAMPLE_SEED,
# CLT_A, CLT_B, MODEL_DIR, DATA_DIR, OUT, plus CLT_STORAGE_ROOT (used by the
# python to resolve the default paths).
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-lm4-ct}"        # NEVER the cu130 `lm4` (reports cuda=False)
N_ROWS="${N_ROWS:-256}"
N_EVAL_ROWS="${N_EVAL_ROWS:-64}"
CHUNK_ROWS="${CHUNK_ROWS:-16}"
SAMPLE_SEED="${SAMPLE_SEED:-2}"
OUT="${OUT:-finetuning/compare_apricot_vs_finetune.json}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
module load cuda 2>/dev/null || true
PY="${PY:-$HOME/.conda/envs/$CONDA_ENV/bin/python}"
[ -x "$PY" ] || { echo "ERROR: no python at $PY (conda env '$CONDA_ENV'?)" >&2; exit 1; }
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$(pwd)}${PYTHONPATH:+:$PYTHONPATH}"
echo "python: $PY   conda_env: $CONDA_ENV"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

exec "$PY" finetuning/compare_clts.py \
    --n-rows "$N_ROWS" --n-eval-rows "$N_EVAL_ROWS" --chunk-rows "$CHUNK_ROWS" \
    --sample-seed "$SAMPLE_SEED" --out "$OUT" \
    ${MODEL_DIR:+--model-dir "$MODEL_DIR"} \
    ${DATA_DIR:+--data-dir "$DATA_DIR"} \
    ${CLT_A:+--clt-a "$CLT_A"} \
    ${CLT_B:+--clt-b "$CLT_B"}
