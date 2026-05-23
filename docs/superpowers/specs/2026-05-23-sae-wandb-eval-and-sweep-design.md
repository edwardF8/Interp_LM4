# SAE Eval Logging + Hyperparameter Sweep — Design

**Date:** 2026-05-23
**Files touched:** `trainSAE.py`, `sweeps/sae_grid.yaml` (new)

## Goal

Two related additions to the SAE training pipeline:

1. **Log final eval results to wandb** so reconstruction quality (explained variance, CE recovered, dead-feature count) is visible alongside training curves and shows up as sweep summary columns.
2. **Run a wandb grid sweep** over the highest-impact SAE hyperparameters with Hyperband early stopping.

## Non-goals

- No new eval logic — reuse `sae_eval()` from `evalSAE.py` as-is.
- No changes to the SAE model or training loop itself.
- No sweep over `hook_name` — stays fixed at `blocks.1.hook_mlp_out` for this round.
- No mid-training eval logging beyond what sae_lens already does.

## Design

### Part 1: Eval logging in `trainSAE.py`

At the end of `trainSAE.py`, after `runner.run()` returns the trained SAE, run held-out eval and log to the existing wandb run.

```python
from evalSAE import sae_eval, print_report
import wandb

# Held-out subset uses different seed than training
eval_subset = DiverseBioSubset(
    sampler, tokenizer, context_size=context_size, seed=SAE_seed + 1
)
eval_rows = eval_subset.to_hf_dataset(64, verbose=False)["input_ids"]
eval_tokens = torch.tensor(np.array(eval_rows), dtype=torch.long, device=device)

metrics = sae_eval(model, sae, eval_tokens, cfg.hook_name)
print_report(metrics)

if wandb.run is not None:
    payload = {f"final_eval/{k}": v for k, v in metrics.items()}
    wandb.log(payload)
    wandb.run.summary.update(payload)
```

Logging to **both** `wandb.log` (timeseries) and `wandb.run.summary` (final-value column) — the summary write is what makes eval metrics appear as columns in the sweep parallel-coords plot.

### Part 2: Sweep parameter overrides in `trainSAE.py`

At the top of `trainSAE.py`, after imports, read overrides from `wandb.config` with defaults for non-sweep runs:

```python
import wandb
wandb.init(project="interpLM4", config={})  # idempotent under sweep agent
sweep_cfg = wandb.config

l0_coefficient = sweep_cfg.get("l0_coefficient", 5.0)
sae_mult       = sweep_cfg.get("sae_mult", 8)
lr             = sweep_cfg.get("lr", 5e-5)
epochs         = sweep_cfg.get("epochs", 30)
```

Then replace the literals later in the config dataclasses with these names. sae_lens detects the existing wandb run.

### Part 3: Sweep config `sweeps/sae_grid.yaml`

```yaml
program: trainSAE.py
method: grid
metric:
  name: final_eval/ce_recovered
  goal: maximize
parameters:
  l0_coefficient:
    values: [2.0, 5.0, 10.0]
  sae_mult:
    values: [8, 16]
  lr:
    values: [3e-5, 1e-4]
  epochs:
    value: 50
early_terminate:
  type: hyperband
  min_iter: 5
  eta: 3
```

12 unique trials × ~50 min each = ~10 GPU-hours, minus Hyperband savings (typically 30-50%).

### Launch

```bash
wandb sweep sweeps/sae_grid.yaml          # prints sweep ID
wandb agent <user>/interpLM4/<sweep_id>   # one agent runs trials sequentially
```

On PSC, an agent can be wrapped in `submit_job_psc.sh` so each trial runs on a separate GPU allocation.

## Cost

| Item | Value |
|---|---|
| Trials | 12 |
| Epochs/trial | 50 (~50 min observed at 30 epochs / 30 min) |
| Naive total | ~10 GPU-hours |
| With Hyperband | ~5-7 GPU-hours expected |

## Verification

- Single non-sweep run of `trainSAE.py` still works (uses defaults), prints `print_report` output, and creates a wandb run with `final_eval/*` keys in both timeseries and summary.
- `wandb sweep sweeps/sae_grid.yaml` returns a sweep ID without error.
- One sweep trial run shows the overridden hyperparameters in `wandb.config` and uses them in the SAE training.

## Out of scope (future)

- Sweeping `hook_name` across layers/sites.
- Logging per-feature stats (activation density histograms) to wandb.
- Two-stage sweep (coarse scan + zoom-in on best region).
