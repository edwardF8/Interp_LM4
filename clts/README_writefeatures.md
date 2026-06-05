# writeFeatures — feature-in-node rank tester

Given a **people subset** and a **target CLT feature** `(layer, fidx)`, measure where
that feature ranks among the **direct input features** of an output-token node (e.g.
the ` August` logit) in the attribution graph — across all 46 bio templates and all
the people. Reports a rank histogram, a strict "meaningful across tokens" metric, and
a unified co-influencer table that labels error/token nodes by the **token role** they
sit on (e.g. `err@last_name[final]@L2`).

Built for the trained `grid-L4-H6` CLT, but every path takes the model/CLT/data as
arguments, so it works on any scan.

> Spec + plan: `docs/superpowers/specs/2026-06-04-writefeatures-feature-node-rank-design.md`,
> `docs/superpowers/plans/2026-06-04-writefeatures-feature-node-rank.md`.

---

## What it answers

For one feature and one output node, across many people × many phrasings:

1. **Rank histogram** — where does the feature land among the node's direct input
   features? Buckets: `top1 / top2 / top3 / top4 / top5 / 6-10 / >10 / absent`.
   (Direct edge into the logit = `adjacency_matrix[logit_row, :n_features]`; a feature
   is summed across the token positions where it fires, then ranked.)
2. **Meaningful across tokens (strict)** — does the feature drive the node from ≥2
   token positions *each independently* strong (each in the node-level top-K)? A
   "one big + one tiny" pair does **not** count. (Contrast: a looser "fired at ≥2
   positions" flag is also reported.)
3. **Unified co-influencers** — *all* input nodes to the logit ranked together:
   features (`L3 F4768`), **error** nodes (`err@<role>@L<layer>`), and token nodes
   (`tok@<role>`), each labeled by the role of the token they sit on. This is where
   the model's non-feature ("unexplained", error-node) contribution shows up.

**Token roles** (for error/token labels): `BOS`, `first_name`, `middle_name`,
`last_name`, exactly one `last_name[final]` per graph (the final subword of the last
name — always emitted so the bucket is stable whether the name is one token or
several), `template:{born,birth,day,date,…}` (a configurable allow-list), else
`template:other`; the final/recall position also carries a `(recall)` suffix.

**Unskew guarantee:** the rank histogram and the meaningful-across-tokens metric are
computed over **feature columns only**. Adding error/token nodes is an additive,
read-only second pass that feeds *only* the co-influencer view — it never moves the
feature rank. (Enforced by a regression test.)

---

## File map

| File | What it is |
|---|---|
| `clts/writefeatures.py` | All logic, dependency-injected (`model`/`graph`/`sampler`/`tokenizer` are args). Pure helpers + the `run_hypothesis` sweep. |
| `tests/test_writefeatures.py` | 28 tests (27 pure + 1 artifact-gated integration test on the real model). |
| `clts/writeFeatures.ipynb` | Thin notebook: MODEL CONFIG cell → loading → **EDIT cell** → run/report. Interactive local use. |
| `clts/run_writefeatures_hpc.py` | HPC worker: shardable, cache-free, aggregate-only population sweep over every person per month. |
| `clts/merge_writefeatures_hpc.py` | Sums the shard aggregates into per-(month, feature) reports + `summary.csv`. |
| `scripts/writefeatures_psc.sbatch` | PSC (Bridges-2) SLURM array job: GPU-shared, conda `lm4-ct`, the `$REMOTE_BASE` paths, preflight. |
| `scripts/setup_ct_env_psc.sh` | Builds the `lm4-ct` conda env (circuit-tracer) fast with `uv`. |

---

## Local use (notebook)

Open `clts/writeFeatures.ipynb` with the `clts/.venv-ct` kernel, run the loading
cells once, then edit **one cell** and run the last cell:

```python
# ===================== EDIT THIS CELL =====================
PEOPLE          = people_in_month("August")  # or people_by_ids([...]) / people_by_idx([...]) / sample_in_month("August", 20)
TARGET_FEATURE  = (3, 4768)                  # (layer, feature_idx) to locate
TARGET          = "month"                    # each person's own birth-month token; or pin e.g. " August"
TEMPLATES       = "all"                      # all 46, or [0, 5, 12], or ["{name} popped out on {birthday}."]
N_PEOPLE_CAP    = 20                         # cap; capping RANDOM-SAMPLES (seeded), never first-N
SEED            = 0
TOP_K           = 10                         # unified co-influencer depth
MULTI_TOK_TOP_K = 5                          # node-level top-K for "meaningful across tokens"
POS_SPAN_FLAG   = 3                          # loose flag: fired >=2 positions spanning >= this many tokens
RANK_BY_ABS     = False                      # False = signed (promoters first); True = |edge|
TEMPLATE_WORD_LABELS = {"born", "birth", "day", "date"}
INCLUDE_TOKEN_NODES  = True
SUBSET_LABEL    = "august"                   # used in the saved report filename
# ==========================================================
```

To swap models, edit the MODEL CONFIG cell (`MODEL_DIR / CLT_DIR / DATA_DIR /
SCAN_NAME / DEVICE`). Reports + cache land under
`clt_storage/clt_feature_explorer/<scan>/hyptest/`; the notebook caches the per-
`(prompt, target)` edge decomposition so swapping `TARGET_FEATURE` is instant.

Run the tests:
```bash
clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -v
```

---

## Population sweep on HPC (PSC Bridges-2)

The full run is **all ~50k people × 46 templates ≈ 2.3M attribution graphs**, so it
shards across a SLURM array. **No attribution cache is written** (each graph is built
in memory, metrics extracted, discarded), so the output is **~100–150 MB** of
aggregates + reports — *not* the ~270 GB an interactive cache would cost. The
per-month target features live in `MONTH_FEATURES` at the top of
`run_writefeatures_hpc.py`; months with two features evaluate both from the same graph.

### 0. Build the env once

```bash
bash scripts/setup_ct_env_psc.sh          # conda env 'lm4-ct' via uv; prints "OK torch=..."
```
A dedicated env is required: circuit-tracer pins `transformers<=4.57.3` /
`safetensors>=0.5.0`, which conflict with the SAE env's `sae-dashboard` / `decode-clt`.

### 1. Calibrate the GPU rate (model is tiny, so measure first)

```bash
module purge; module load cuda anaconda3; eval "$(conda shell.bash hook)"; conda activate lm4-ct
export REMOTE_BASE=/jet/home/friedmae/data_storage/LM4_Results
export MODEL_DIR=$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid/grid-L4-H6/final
export CLT_DIR=$REMOTE_BASE/clt_runs/grid-L4-H6/sweep-cfbp6man/mult16_l02_lr0.0001_ep50_n10000/final
export DATA_DIR=$REMOTE_BASE/Data/bioS_N-Bd_final_grid
python -u clts/run_writefeatures_hpc.py --model-dir "$MODEL_DIR" --clt-dir "$CLT_DIR" \
  --data-dir "$DATA_DIR" --scan-name grid-L4-H6 \
  --months August --limit-people 20 --device cuda --batch-size 1024 --progress-every 5
```
From the printed `(N/s)`: `sec_per_graph = 1/(N×46)`; size the array so each shard
fits `--time`: **`NUM_SHARDS ≥ 2.3e6 × sec_per_graph / 28800`** (8 h).

### 2. Submit the array

```bash
# sanity: one shard first
NUM_SHARDS=200 sbatch --array=0-0 scripts/writefeatures_psc.sbatch
# then the full run (array size MUST equal NUM_SHARDS):
NUM_SHARDS=32 sbatch --array=0-31 scripts/writefeatures_psc.sbatch
```
Shrink the job with `EXTRA_ARGS="--templates 0,5,12"` (fewer prompts) or
`EXTRA_ARGS="--months August,January"` (subset).

### 3. Merge once the array finishes

```bash
conda activate lm4-ct
python -u clts/merge_writefeatures_hpc.py --scan-name grid-L4-H6
```

### 4. Read results

```bash
cat $REMOTE_BASE/clt_feature_explorer/grid-L4-H6/hpc/reports/summary.csv
```
One row per (month, feature): `top1_pct`, `top1_3_pct`, `meaningful_pct`,
`top_error_coinfluencer`. Full per-(month,feature) JSONs and per-month combined JSONs
are in the same `reports/` folder. (`storage_root()` resolves to `$REMOTE_BASE` on PSC,
so everything lands in remote storage automatically.)

---

## Example result (August, `L3 F4768`)

From the 3-person × 46-template smoke (coarse, but representative):
- **Rank among direct inputs:** top-1 36%, top-2 34%, top-3 23% → **~94% in the top 3**,
  never absent.
- **Meaningful across tokens:** ~0% (fires at ≥2 positions only ~4% of the time, and
  never with both positions strong).
- **Unified co-influencers:** the #1 contributor to the ` August` node is
  `err@last_name[final]@L2` — an *error* node on the final subword of the last name
  (≈6% of the logit's total edge mass, ≈62% of the error mass). Token nodes
  (`tok@*`) rank near zero.

So `L3/4768` is robustly a top-3 *promoter* of the August node across phrasings, and
the dominant *non-feature* signal is unexplained reconstruction error localized on the
last name.

---

## Gotchas / notes

- **Leading-space in-vocab check.** `people_in_month` filters names with a leading
  space (`f" {first} {last}"`) — the condensed vocab only has the `Ġ`-prefixed name
  fragments, so without the space the filter silently drops ~97% of people. (Fixed; see
  the comment in `people_in_month`.)
- **Cache schema tag.** The notebook cache keys include `"schema": "node_all_v1"`, so
  pre-v1 cache files are ignored, not mixed in.
- **Tiny model → modest GPU speedup.** 4 layers / d_model 384: per-graph cost is
  attribution-algorithm overhead, not big matmuls, so GPU helps but isn't dramatic —
  array width is the bigger lever. Always calibrate.
