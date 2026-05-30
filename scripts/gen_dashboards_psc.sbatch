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
# Submit CLT feature-dashboard generation as a GPU batch job. Mirrors
# Training_On_LM4/setup_job_psc.sh (GPU-shared, conda env `lm4`). The actual
# work + the path/env PREFLIGHT live in scripts/gen_dashboards_psc.sh, which
# this just runs on the allocated GPU node.
#
# Usage (from the repo root on PSC):
#   sbatch scripts/gen_dashboards_psc.sbatch            # default: moderate (--n-people 1000)
#   sbatch scripts/gen_dashboards_psc.sbatch validate   # smoke (--n-people 100)
#   sbatch scripts/gen_dashboards_psc.sbatch full        # all people (bump --time if needed)
#
# Override paths/settings by exporting them before submit (sbatch --export=ALL is
# the default, so exported vars propagate), e.g.:
#   export CLT_DIR=/jet/home/friedmae/data_storage/LM4_Results/clt_runs/grid-L4-H6/sweep-cfbp6man/mult16_l02_lr0.0001_ep50_n10000/final
#   sbatch scripts/gen_dashboards_psc.sbatch moderate
#
# RECOMMENDED: run the fast preflight on the login node first (no GPU needed):
#   conda activate lm4 && CLT_DIR=$CLT_DIR scripts/gen_dashboards_psc.sh preflight
set -e
mkdir -p logs

# --- Environment (mirrors setup_job_psc.sh) --------------------------------
module purge
module load cuda
module load anaconda3

# Initialize conda for this non-interactive shell (module load puts conda on
# PATH but doesn't define the `conda` shell function).
set +e
_conda_initialized=0
eval "$(conda shell.bash hook 2>/dev/null)" && _conda_initialized=1
if [ "$_conda_initialized" = 0 ]; then
    _base="$(conda info --base 2>/dev/null)"
    if [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ]; then
        source "$_base/etc/profile.d/conda.sh" && _conda_initialized=1
    fi
fi
set -e
if [ "$_conda_initialized" = 0 ]; then
    echo "ERROR: could not initialize conda" >&2
    exit 1
fi
conda activate lm4
echo "Active env: ${CONDA_DEFAULT_ENV:-none}  ($(which python))"

# Keep HF cache on fast node-local scratch if available.
if [ -n "${LOCAL:-}" ]; then
    export HF_HOME="$LOCAL/hf_cache"
    export TRANSFORMERS_CACHE="$LOCAL/hf_cache"
    mkdir -p "$HF_HOME"
fi
export PYTHONUNBUFFERED=1

# --- Run --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
echo "=== Job ${SLURM_JOB_ID:-local} on $(hostname) ==="
date
nvidia-smi || true
echo

MODE="${1:-moderate}"
# gen_dashboards_psc.sh runs the preflight first, then the generation (DEVICE
# defaults to cuda; PYTHON defaults to the active `lm4` python).
scripts/gen_dashboards_psc.sh "$MODE"

echo
echo "Finished: $(date)"
