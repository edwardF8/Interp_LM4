#!/bin/bash
#SBATCH --job-name=runningSAEInference
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --account=cis240072p
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egfriedm@andrew.cmu.edu

# Notes on the conservative defaults above:
#   - GPU-shared lets you book 1 GPU instead of a whole 8-GPU node.
#   - gpu:1 picks one A100 on the shared partition; switch to gpu:v100-32:1
#     if you ever want a cheaper V100 node.
#   - 5 CPUs covers HF Trainer's 2 dataloader workers + the main process.
#   - No explicit --mem: GPU-shared auto-allocates ~125G per A100 which is
#     plenty for these small models.

set -e   # bail if `conda activate` (or any earlier command) fails

# --- Environment ------------------------------------------------------------
mkdir -p logs

# Clean module state so the job is reproducible regardless of login shell.
module purge
module load cuda
module load anaconda3
# module load anaconda3                # uncomment if your env needs it

# Activate the project conda env (fill in the name).
conda activate lm4
# Keep HuggingFace + wandb caches on $LOCAL (fast node-local scratch) if set,
# otherwise they default to $HOME and chew up your home quota.
if [ -n "${LOCAL:-}" ]; then
    export HF_HOME="$LOCAL/hf_cache"
    export TRANSFORMERS_CACHE="$LOCAL/hf_cache"
    export WANDB_CACHE_DIR="$LOCAL/wandb_cache"
    mkdir -p "$HF_HOME" "$WANDB_CACHE_DIR"
fi

# wandb auth: prefer `wandb login` once on the login node (writes ~/.netrc).
# If you can't do that, uncomment and set:
# export WANDB_API_KEY=<your_api_key>

# Make Python output unbuffered so the SLURM log streams live.
export PYTHONUNBUFFERED=1

# --- Run --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Started: $(date)"
nvidia-smi || true
echo

# If the first arg is a .py file, run it with `python -u`. Otherwise (e.g.
# `wandb agent <id>`), exec the command as-is.
if [[ "$1" == *.py ]]; then
    python -u "$@"
else
    "$@"
fi

echo
echo "Finished: $(date)"
