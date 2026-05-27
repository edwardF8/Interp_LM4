# SAE Training and Feature Finding Framework

End-to-end pipeline for training Sparse Autoencoders on the bioS-NM-BD language models, running per-feature inference, and probing features at the per-token level. Designed to scale across base models and layers from a single command set, with a consistent file layout that survives an rsync from PSC to your laptop.

## At a glance

```
┌─────────────────────────┐    ┌───────────────────────────┐    ┌──────────────────────────┐
│  saes/trainSAE.py       │───▶│  saes/sae_runs/...        │───▶│ saes/runInference.py     │
│  • CLI flags            │    │  trained SAE checkpoints  │    │  • feature_stats.pt      │
│  • Wandb sweep grid     │    │  per (model × layer × hp) │    │  • buckets.pt            │
│  • Layer-aware trial    │    │                           │    │  • dashboard_*.html      │
│    naming               │    │                           │    │  per SAE                 │
└─────────────────────────┘    └───────────────────────────┘    └─────────┬────────────────┘
                                                                          │
                                                                          ▼
                                                                ┌───────────────────────────┐
                                                                │  saes/sae_inference/...   │
                                                                │  precomputed inference    │
                                                                └─────────┬─────────────────┘
                                                                          │
                                                  ┌───────────────────────┴────────────────────┐
                                                  ▼                                            ▼
                                       ┌──────────────────────┐                  ┌───────────────────────────┐
                                       │ analyzingSAE.ipynb   │                  │ featureBucketing.ipynb    │
                                       │ • dashboards, DLA,   │                  │ • general per-token       │
                                       │   ablation, steer    │                  │   bucket queries          │
                                       └──────────────────────┘                  └───────────────────────────┘
```

## File layout

The whole pipeline depends on one invariant: the per-trial directory name is identical in `sae_runs/` and `sae_inference/`. That's how the notebooks find the inference dir from a SAE path.

```
Interp_LM4/saes/
├── sae_runs/                                              # output of trainSAE.py
│   └── <model_name>/                                      # e.g. grid-L4-H6
│       └── <sweep_folder>/                                # sweep-<wandb-id> or "standalone"
│           └── <trial_name>/                              # L<n>_mult<m>_l0<x>_lr<r>_ep<e>_n<k>
│               ├── checkpoints/                           # mid-training snapshots
│               └── final/                                 # ← SAE_PATH points here
│                   ├── cfg.json
│                   └── sae_weights.safetensors
└── sae_inference/                                         # output of runInference.py
    └── <model_name>/
        ├── index_corpus.pt                                # shared across SAEs of this model
        └── <trial_name>/                                  # SAME name as in sae_runs/
            ├── feature_stats.pt
            ├── buckets.pt
            ├── dashboard_feature_*.html
            └── index.html
```

### Naming conventions

- **`<model_name>`** — identifier for the base model. Defaults to `parent(--model-dir).name`, so `/jet/.../grid/grid-L4-H6/final` → `grid-L4-H6`. Overridable with `--model-name`.
- **`<sweep_folder>`** — `sweep-<wandb-id>` for sweep trials, `standalone` for one-off runs. With per-layer sweeps you get one `sweep-<id>/` per layer (so `--layers 0,1,2,3` produces four sibling sweep folders under `<model_name>/`).
- **`<trial_name>`** — `L<n>_mult<m>_l0<x>_lr<r>_ep<e>_n<k>`. The `L<n>_` prefix is parsed from the hook (`blocks.<n>.hook_mlp_out`) so different-layer trials in the same sweep never collide. `:g` float formatting keeps names tidy (`l05` not `l05.0`).

### From a notebook's perspective

Given `SAE_PATH = REPO_ROOT/"saes"/"sae_runs"/"grid-L4-H6"/"sweep-abc123"/"L2_mult8_l02_lr3e-05_ep50_n10000"/"final"`:

```python
TRIAL_NAME    = SAE_PATH.parent.name           # 'L2_mult8_l02_lr3e-05_ep50_n10000'
MODEL_NAME    = SAE_PATH.parents[2].name       # 'grid-L4-H6'  (skips <sweep_folder>)
INFERENCE_DIR = REPO_ROOT/"saes"/"sae_inference"/MODEL_NAME/TRIAL_NAME
```

Everything else in the notebook keys off `INFERENCE_DIR` and `HOOK = sae.cfg.hook_name`.

## 1. Training — `saes/trainSAE.py`

Trains one or more SAEs on a base model. All paths are CLI flags; nothing is hardcoded.

### Single run

```bash
python saes/trainSAE.py \
  --model-dir /jet/.../grid-L4-H6/final \
  --data-dir  /jet/.../bioS_N-Bd_final_grid \
  --hook "blocks.2.hook_mlp_out"
```

Output: `saes/sae_runs/grid-L4-H6/standalone/L2_mult8_l05_lr5e-05_ep30_n10000/final/`.

### Sweep (recommended)

```bash
python saes/trainSAE.py \
  --model-dir /jet/.../grid-L4-H6/final \
  --data-dir  /jet/.../bioS_N-Bd_final_grid \
  --layers 0,1,2,3 \
  --hook-template "blocks.{layer}.hook_mlp_out" \
  --sweep
```

Registers **one wandb sweep per layer**, run sequentially within the same sbatch job. Each per-layer sweep grids over `(l0 × sae_mult × lr)`:

| axis            | values                       |
|-----------------|------------------------------|
| `hook_name`     | singleton — baked into each sweep's config |
| `l0_coefficient`| `[2.0, 5.0, 10.0]`           |
| `sae_mult`      | `[8, 16]`                    |
| `lr`            | `[3e-5, 1e-4]`               |
| `epochs`        | `50`                         |

So `--layers 0,1,2,3` becomes 4 wandb sweeps × 3 l0 × 2 mult × 2 lr = **48 trials total**, with the layer dimension *between* sweeps rather than inside one. Sweep names: `sae_sweep_<model>_L0`, `..._L1`, etc. Hyperband early-terminate (`min_iter=5, eta=3`) prunes within each layer's grid independently.

Per-layer (rather than combined) sweeps because:
- Cleaner rankings on the wandb UI — each sweep page is one layer's hyperparam grid
- Hyperband prunes within a layer instead of across layers (avoids cutting trials early just because a different layer is converging faster)
- Easier to rerun or extend one layer without re-running the rest

For *parallelism* (independent SLURM jobs per layer) submit separate sbatches:
```bash
for L in 0 1 2 3; do
  sbatch -J L4-L$L submit_job_psc.sh saes/trainSAE.py \
    --model-dir .../grid-L4-H6/final --data-dir ... --layers $L --sweep
done
```

### Storage strategy

One path, hardcoded at the top of `saes/trainSAE.py`:

```python
STORAGE_ROOT = Path("/jet/home/friedmae/data_storage/LM4_Results/saes")
```

Trained SAEs and all inference outputs land permanently here. On PSC this must be on Ocean (data_storage) because `$HOME` has a ~25 GB quota that fills fast. Edit the constant if you move workspaces.

**Why no `$LOCAL` staging?** With `n_checkpoints=0` sae_lens writes exactly one ~20 MB safetensors file per trial at the end of `runner.run()`. That's a single large sequential write — Lustre/Ocean's strong case per PSC's storage docs ("If you write one big checkpoint blob, Ocean handles that fine"). Going through `$LOCAL` first only earns its keep when there are *many small writes during training* — e.g. if you bump `n_checkpoints` back up to save mid-training snapshots. In that case, switch `output_path` to a `$LOCAL` staging dir and copy to `STORAGE_ROOT` after eval.

`setup()` runs a write-probe on `STORAGE_ROOT` at the start of every job so permission/quota issues surface in <1s instead of after a 30-min training run. The wandb run summary gets a `storage_path` field so you can find the on-disk SAE from the wandb UI.

### CLI reference

| flag                  | required | default                          | meaning                                  |
|-----------------------|----------|----------------------------------|------------------------------------------|
| `--model-dir`         | yes      | —                                | HF Llama checkpoint dir                  |
| `--data-dir`          | yes      | —                                | dataset dir (people.json + remap)        |
| `--model-name`        | no       | parent dir of `--model-dir`      | output path identifier                   |
| `--hook`              | no       | `blocks.1.hook_mlp_out`          | hook for single runs                     |
| `--layers`            | sweep    | —                                | comma-separated layer indices            |
| `--hook-template`     | no       | `blocks.{layer}.hook_mlp_out`    | template for `--layers` expansion        |
| `--sweep`             | no       | off                              | launch a wandb sweep                     |
| `--n-examples`        | no       | 10000                            | bios per epoch                           |
| `--epochs`            | no       | 30                               | training epochs (single run only)        |
| `--context-size`      | no       | 512                              | token budget per bio                     |
| `--sae-mult`          | no       | 8                                | `d_sae = d_model × sae_mult`             |
| `--l0`                | no       | 5.0                              | L0 sparsity coefficient                  |
| `--lr`                | no       | 5e-5                             | learning rate                            |

No `--output-root` / `--output-dir` — output location is the hardcoded `STORAGE_ROOT`. Change one place, every script picks it up.

### Internals

- `setup(args)` loads model, tokenizer, sampler, and held-out eval tokens once at module level. wandb.agent reuses the process across trials, so the model loads only once per sbatch.
- `train_one_run()` reads from `wandb.config` during a sweep, falling back to `ARGS` otherwise. So the same function powers both standalone and swept runs.
- `_patch_signal_for_worker_threads()` neutralizes the SIGINT handler sae_lens installs on the main thread (wandb.agent runs trials in worker threads). Without this, trials crash on entry.

## 2. Post-sweep inference — `saes/runInference.py`

Scans a sweep dir and runs three artifacts per SAE: feature statistics, bucket statistics, and per-feature HTML dashboards.

```bash
python saes/runInference.py \
  --sweep-dir saes/sae_runs/grid-L4-H6/sweep-<id> \
  --model-dir /jet/.../grid-L4-H6/final \
  --data-dir  /jet/.../bioS_N-Bd_final_grid
```

The hook for each SAE is read from `sae.cfg.hook_name`, so a mixed-layer sweep dir works without extra flags.

### Artifacts per SAE

| file                      | size  | purpose                                                              |
|---------------------------|-------|----------------------------------------------------------------------|
| `feature_stats.pt`        | ~50 KB | per-feature activation density, max activation, fire count           |
| `buckets.pt`              | ~45 MB | `by_input_token` bucketing framework output (see §4)                |
| `dashboard_feature_*.html`| ~500 KB ea | Neuronpedia-style per-feature panels                            |
| `index.html`              | small | navigator for the dashboards                                         |

Toggle the heavy steps with `--no-stats`, `--no-buckets`, `--no-dashboards`.

### Shared index corpus

`saes/sae_inference/<model_name>/index_corpus.pt` is built once per model. Every SAE in that model's sweep dir uses the same token tensor, which guarantees feature numbers line up between dashboards, stats, and buckets — and between any two SAEs trained on the same model.

The corpus shape: 50,000 people × 2 templates × 64 tokens = 6.4M positions. Override via `--n-per-person`, `--context-size`.

## 3. Quick SAE evaluation — `saes/evalSweep.py`

Scans a directory and prints L0 / explained variance / CE-recovered for every SAE, without going through wandb.

```bash
python saes/evalSweep.py saes/sae_runs/grid-L4-H6/sweep-<id> \
  --model-dir /jet/.../grid-L4-H6/final \
  --data-dir  /jet/.../bioS_N-Bd_final_grid
```

The hook for each SAE is read from `sae.cfg.hook_name`. Pass any directory containing `*/final/` subdirs.

## 4. Bucketing framework

The per-token analyses at the end of `analyzingSAE.ipynb` all had the same shape:

> *for each position, encode SAE features, L1-normalize the `[d_sae]` vector so it reads as a distribution over features, then average it conditional on some property of that position.*

The framework in [`saes/sae_explorer.py`](../saes/sae_explorer.py) makes that explicit. One forward+SAE pass updates every registered axis in parallel; the result is a small tensor you save once and slice many ways on CPU.

### Concepts

A **Bucketer** maps `[B, T]` token batches to `[B, T]` bucket ids. Bucket id `-1` skips a position (e.g. `next_token` at the last position of a sequence has no defined key).

```python
@dataclass
class Bucketer:
    name: str
    n_buckets: int
    fn: Callable[[torch.Tensor], torch.Tensor]
```

### Built-in bucketers

| function                       | bucket key                              |
|--------------------------------|------------------------------------------|
| `by_input_token(vocab_size)`   | token id at this position                |
| `by_next_token(vocab_size)`    | token id at the next position            |
| `by_prev_token(vocab_size)`    | token id at the previous position        |
| `by_position(context_size)`    | absolute index in the sequence           |

Need something else? Write `f([B, T] tokens) → [B, T] long`, wrap it in `Bucketer(name=..., n_buckets=..., fn=f)`, register it.

### Compute

```python
from saes.sae_explorer import compute_bucketed_stats, by_input_token

bucket_stats = compute_bucketed_stats(
    model, sae,
    tokens=tokens,
    hook_name="blocks.2.hook_mlp_out",
    bucketers=[by_input_token(tokenizer.vocab_size)],
    ignore_token_ids={tokenizer.pad_token_id},
)
torch.save(bucket_stats, "buckets.pt")
```

Returns:
```python
{
  "buckets": {
    "input_token": {
      "sum":   tensor([vocab_size, d_sae]),   # Σ L1-normalized feature vectors per bucket
      "count": tensor([vocab_size]),          # # positions per bucket
    }
  }
}
```

Adding more bucketers is nearly free — the model forward pass dominates.

### Query

```python
from saes.sae_explorer import query_bucket

result = query_bucket(
    bucket_stats["buckets"]["input_token"],
    target_keys=[tokenizer.encode(" November")[0]],
    ignore_keys=[tokenizer.pad_token_id],
)
# result keys: mean_target, mean_other, specificity,
#              n_target_positions, n_other_positions
```

`mean_target` is the average normalized feature distribution at target positions. `specificity = mean_target / (mean_other + eps)` ranks features by how selectively they fire on the target class.

### Storage tradeoff

`by_input_token` for our model: 1836 × 6144 × 4 B = **~45 MB**. Small enough to cache to disk so notebooks load instantly. The pre-aggregation locks in the *axes* you can slice on — input-token, position, etc. — but each axis is one extra accumulator inside the same forward loop, so it's cheap to add new ones at compute time.

## 5. Notebooks

Both notebooks live under `saes/`, derive `MODEL_NAME` + `TRIAL_NAME` + `INFERENCE_DIR` from `SAE_PATH`, and read `HOOK` from `sae.cfg.hook_name` after loading the SAE. To switch SAEs, edit only `SAE_PATH`.

### `analyzingSAE.ipynb` — full per-SAE exploration

- Discover available SAEs (list under `saes/sae_runs/`)
- Load model + SAE; auto-derive hook
- Feature dashboards (`show_feature(idx)` — embeds the precomputed HTML inline)
- Feature activity histogram + activations-per-latent bar plot (reads `feature_stats.pt`)
- Per-bio inspection (`top_features_for_text` — which features fire on one bio)
- DLA — direct logit attribution per feature
- `steer()` — causal probe (boost a feature's decoder direction)
- `ablate()` — causal complement (zero out a feature in the reconstruction)
- Legacy per-token analyses (months / per-month) — superseded by `featureBucketing.ipynb`

### `featureBucketing.ipynb` — bucketing framework demo

Six-section walkthrough:

1. Load model, data, tokenizer
2. View feature dashboard
3. Per-person inspection
4. Framework — register a bucketer, run the sweep, save to `buckets.pt`
5. Months bucket — all month tokens grouped (matches `features_for_token_set` from the old notebook)
6. Per-month bucket — one month at a time, shared `other` baseline

If `buckets.pt` was already produced by `runInference.py`, Sections 5-6 are pure CPU slicing — milliseconds per query.

## 6. PSC workflow

### Submit a sweep

```bash
DATA=/jet/home/friedmae/data_storage/LM4_Results/Data/bioS_N-Bd_final_grid
GRID=/jet/home/friedmae/data_storage/LM4_Results/runResults/bioS_N-Bd_final_grid/20260520-134455/grid

sbatch -J L4-allLayers submit_job_psc.sh saes/trainSAE.py \
  --model-dir $GRID/grid-L4-H6/final --data-dir $DATA \
  --layers 0,1,2,3 --sweep
```

The `-J` flag overrides the SBATCH `--job-name` directive so `squeue -u <user>` shows useful names and log files (e.g. `logs/L4-allLayers-<jobid>.out`) inherit it. Other useful overrides:

- `-t HH:MM:SS` — different wall-clock time (default in `submit_job_psc.sh` is 24h)
- `-o <path>` — custom log path
- `--gres=gpu:v100-32:1` — switch to V100 for cheaper short jobs

### Post-sweep inference

```bash
sbatch -J inf-L4 submit_job_psc.sh saes/runInference.py \
  --sweep-dir saes/sae_runs/grid-L4-H6/sweep-<id> \
  --model-dir $GRID/grid-L4-H6/final --data-dir $DATA
```

### Sync to laptop

```bash
rsync -av friedmae@bridges2:Interp_LM4/saes/{sae_runs,sae_inference}/ \
    ~/Code/Project\ Code/CRL-Interp/Interp_LM4/saes/
```

Same command brings over both SAEs and inference outputs; re-running picks up only the new trials.

## 7. Design decisions worth knowing

### Why pre-aggregate buckets instead of caching raw activations?

Caching `[100k, 64, d_sae]` raw SAE activations would be ~150 GB. Pre-aggregating to `[vocab_size, d_sae]` is 45 MB and covers every "features for token class X" question. Trades flexibility (you must enumerate axes at compute time) for cheap, fast queries.

### Why L1-normalize before averaging?

Without normalization, a single position with a huge max activation dominates the mean. L1-normalization makes the `[d_sae]` vector at each position a *distribution* — every position contributes the same total mass, so the average reads as "what fraction of activation mass does each feature claim, conditional on the bucket?".

### Why `<model_name>` as a directory layer (not a name prefix)?

`ls saes/sae_runs/` shows base models at a glance. `rm -rf saes/sae_runs/grid-L4-H6/` cleanly removes one model's outputs. The trial name stays focused on hyperparameters, not on encoding which base model it was trained against.

### Why hook auto-detect in the notebook?

Hand-tracking `HOOK = "blocks.N.hook_mlp_out"` alongside `SAE_PATH` was a constant footgun when sweeping layers — easy to update one and forget the other. `sae.cfg.hook_name` is the source of truth; the notebook just reads it.

## 8. Files at a glance

| path                          | role                                                            |
|-------------------------------|-----------------------------------------------------------------|
| `saes/trainSAE.py`            | training entry point, CLI + sweep                               |
| `saes/runInference.py`        | post-sweep inference (stats + buckets + dashboards)             |
| `saes/evalSweep.py`           | quick eval over a sweep dir                                     |
| `saes/evalSAE.py`             | `sae_eval` / `print_report` / `load_sae` helpers                |
| `saes/sae_explorer.py`        | bucketing framework + dashboard helpers + DLA/steer/ablate      |
| `saes/analyzingSAE.ipynb`     | full per-SAE exploration notebook                               |
| `saes/featureBucketing.ipynb` | bucketing framework demo notebook                               |
| `submit_job_psc.sh`           | sbatch wrapper for PSC                                          |
| `util/computeInference.py`    | legacy single-SAE inference script (superseded by runInference) |
