# writeFeatures month atlas — feature-swappable, population-scale

**Date:** 2026-06-11
**Status:** Design approved (pending spec review)

## Purpose

Make **any** CLT feature's contribution to the ` <month>` logit node queryable on the
Mac **without recompute**. Today's HPC sweep (`clts/run_writefeatures_hpc.py`) bakes a
fixed feature list (`MONTH_FEATURES`) in at compute time and discards the per-graph
edges, so studying a newly-discovered feature requires a full HPC re-run. This design
replaces the discard with a compact **atlas**: store the top-N feature edges per graph
(feature-independent), ship ~1–2 GB to the Mac, and replay the *entire existing
per-feature report* — rank histogram, loose multi-position, strict cross-token, and
unified co-influencers — for any feature, locally, forever.

This is the first lens of a broader "understand any feature's importance" goal:
*how much does a feature contribute to a logit node*. Scope here is deliberately
narrowed to the **month** target only (day/year deferred — see Out of scope).

## Scope

- **In scope:** an *atlas* output mode for the month sweep (store top-N edges + identity
  per graph instead of aggregate-only); a large seeded **sample** (~500 people/month);
  a Mac-side loader that turns the atlas into the four existing metrics for any feature;
  reuse of the existing pure functions in `clts/writefeatures.py`.
- **Out of scope:** day and year targets (the 2-token year wrinkle, `' 1816' →
  [237, 256]`, is deferred); the other five bio fields; any change to the four metric
  definitions; the intrinsic/logit-lens importance pass.

## Background (verified)

- Each `attribute()` graph exposes logit *nodes* only for the **final** prompt position,
  so the ` <month>` logit only exists when the prompt is truncated right before the
  date. (Day/year would need their own truncations → separate graphs; out of scope.)
- The four metrics are **pure functions of the per-graph edge list** (`feature_rank`,
  loose `n_positions`/`span`, `feature_multitoken`, `unified_top_labels` →
  `build_report`). Storing that edge list ⇒ all four replay for any feature.
- The full per-graph decomposition is ~128 KB/graph (4096 features); the existing
  month sweep at full population is ~2.3M graphs ≈ 270 GB — why it was discarded.
  The atlas record (~7–8 KB/graph: top-N=100 aggregated features + per-position edges
  + top-K nodes + small error/token blocks + identity/prompt) is ~16× smaller per
  graph; at the **~500/month sample** (~275k graphs) that is **under ~2 GB** (full
  population would be ~15–20 GB, available later by re-running without the sample cap).

## Key decisions (locked)

1. **Month-only**, full-population *targets* unchanged (each person's own birth-month
   token), but a **seeded ~500-person sample per month** (`--limit-people 500
   --seed 0`, the existing knob) instead of every person. ~275k graphs (~10× cheaper
   than the prior run).
2. **Top-N = 100.** Every histogram bucket you use (`top1…top5 / 6-10 / >10 / absent`)
   stays exact; only ranks beyond 100 collapse to ">N", which never affects a bucket.
3. **Atlas storage preserves all four metrics exactly** (see Per-graph record). The
   subtlety: the rank histogram ranks by **summed** edge across positions, so naive
   per-node top-N truncation is wrong. The record therefore keeps (a) top-N
   *aggregated* features and (b) the global top-K *raw nodes* separately.
4. **No recompute for new features.** Bringing in a feature = re-running the Mac
   loader with a new `TARGET_FEATURE`; it reads the atlas, never calls `attribute()`.
5. **Atlas is additive, not a rewrite.** `run_writefeatures_hpc.py` gains an
   `--atlas` mode; the existing aggregate path is untouched and stays the default.

## Per-graph atlas record

One record per (person × template) graph. Fields and what each one preserves:

```
{
  # identity (so build_report can label/aggregate)
  "month": str, "ds_idx": int, "id": int, "name": str,
  "t_key": int|str, "prompt": str, "target_token": str,   # e.g. " August"

  # (a) rank histogram (summed) + loose multi-position + span, exact for rank<=N
  "top_features": [
     {"layer": int, "fidx": int, "edge": float,           # summed across positions
      "positions": [int, ...], "per_pos_edges": [float, ...]}
     ... top-N=100 by |summed edge| ...
  ],

  # (b) strict node-level multitoken metric (node_rank <= multi_tok_top_k)
  "top_nodes": [ {"layer","pos","fidx","edge"} ... global top-K raw nodes (K=20) ],

  # (c) "absent" vs ">N"
  "n_features_present": int,        # total feature-node count in the graph

  # (d) unified co-influencer view (already small: layer-major error + token blocks)
  "errors": [ {"layer","pos","edge"} ... ],     # n_tokens * n_layers
  "tokens": [ {"pos","edge"} ... ],             # n_tokens
}
```

Why both `top_features` and `top_nodes`:
- **`top_features`** (aggregated, summed) feeds `feature_rank` directly — exact summed
  rank, positions, and span for any feature in the top-N. A feature outside top-N is
  reported as ">N" (present) or "absent" (via `n_features_present` vs the count seen).
- **`top_nodes`** (raw, global order) feeds `feature_multitoken`: `n_meaningful` counts
  the target feature's positions whose **global node rank ≤ `multi_tok_top_k`** (=5).
  Those nodes are always in the global top handful, so K=20 (> 5) determines the strict
  metric exactly for any feature; positions outside top-20 have node_rank > 5 and are
  correctly non-meaningful.

`per_pos_edges` is kept so a future change to `multi_tok_top_k` or `rank_by_abs` can be
honored from the atlas without recompute.

## Architecture

Three pieces; only the first is new HPC code.

1. **`run_writefeatures_hpc.py --atlas`** — new output mode. For each sampled
   (person × template): build the graph, compute `node_input_all` (already exists),
   then emit the per-graph atlas record (above) instead of folding it into an
   aggregate. Sharded exactly as today (`work[shard::num_shards]`, `--skip-existing`).
   Output: `storage_root()/clt_feature_explorer/<scan>/atlas/<month>/shard{idx}.pt`
   (a list of records), keeping the existing `hpc/` aggregate output untouched.
2. **`merge_writefeatures_atlas.py`** — concatenate shard record-lists per month into
   `atlas/<month>.pt` (or leave sharded; merge is a convenience). Tiny CPU job.
3. **Mac loader (notebook cell + `clts/writefeatures.py` helper)** —
   `load_atlas(scan, months) -> records`; then for a chosen `TARGET_FEATURE`, map each
   record through the **existing** `feature_rank` / `feature_multitoken` /
   co-influencer labeling and call the **existing** `build_report` / `format_report` /
   `save_report`. The only new logic is reconstructing the metric inputs from the
   truncated record (a `feature_rank_from_atlas` / `feature_multitoken_from_atlas`
   adapter that yields identical outputs to the originals for rank ≤ N).

## Data flow

`--atlas` sweep (HPC, once) → per-shard record lists → optional merge → ship `atlas/`
to Mac (~1–2 GB) → Mac loader reads records → for any `TARGET_FEATURE`, adapters feed
the existing metric functions → `build_report` → same JSON/CSV/printout you get today,
now for any feature, no `attribute()` calls.

## Components / helpers (new)

- `atlas_record(node_all, *, top_n, top_k_nodes, person, ds_idx, t_key, prompt,
  target_token, month)` — pure: turns a `node_input_all` dict + identity into the
  per-graph record. Unit-tested with synthetic `node_all`.
- `feature_rank_from_atlas(record, target_feature, rank_by_abs)` — returns the same
  shape as `feature_rank`; uses `top_features` for rank/positions/span, falls back to
  `{"rank": None}` (absent) or a ">N" sentinel when the feature isn't in `top_features`
  but `n_features_present` says the node set was larger.
- `feature_multitoken_from_atlas(record, target_feature, multi_tok_top_k, rank_by_abs)`
  — same shape as `feature_multitoken`; uses `top_nodes`.
- `load_atlas(scan, months)` / `report_from_atlas(records, target_feature, ...)` —
  the notebook-facing convenience that loops records and calls `build_report`.

The existing `build_report`, `format_report`, `save_report`, `rank_bucket`,
`aggregate_by_feature`, `unified_top_labels`, `token_roles` are **reused unchanged**.

## Cost / risk

- **Compute:** ~275k graphs (~10× cheaper than the prior month sweep). Same sharded
  PSC array; `--skip-existing` makes it resumable. Calibrate per-graph rate on one
  shard first (the runner already documents this).
- **Storage:** ~7–8 KB/record × ~275k ≈ **~2 GB** raw, less torch-saved. One transfer.
- **Truncation risk (the only correctness risk):** a feature whose *summed* rank ≤ 100
  must appear in `top_features`. Mitigation: truncate `top_features` by |summed edge|
  **after** `aggregate_by_feature` (not per-node), and keep all positions of each
  retained feature. A regression test asserts atlas-replayed metrics are identical to
  the originals on a full (untruncated) synthetic graph for features within top-N.
- **Sample vs population:** ~500/month gives tight histograms; if a later feature needs
  population scale, re-run `--atlas` without `--limit-people` (additive, same cache).

## Testing / verification

- **Equivalence (headline):** build a synthetic `node_all`, run the original
  `feature_rank`/`feature_multitoken`/`build_report` on it AND the `_from_atlas`
  adapters on `atlas_record(node_all, top_n=100, ...)`; assert **identical** outputs for
  every feature whose summed rank ≤ N, and correct ">N"/absent for the rest.
- **Summed-rank truncation:** construct a feature with several small per-position edges
  whose **sum** ranks it high but no single node is in the per-node top — confirm it is
  retained in `top_features` and its rank/positions/span replay exactly.
- **Multitoken from `top_nodes`:** the big+tiny vs two-strong cases from the original
  spec replay identically from `top_nodes` (K=20, multi_tok_top_k=5).
- **Absent vs >N:** a feature not in `top_features` with `n_features_present` larger
  than the stored count buckets as ">10"/absent correctly (never crashes).
- **End-to-end (artifact-gated):** run `--atlas` on one month, ~3 people, 2 templates;
  load the records on the "Mac path" and produce a report; confirm it matches a direct
  `run_hypothesis` report on the same (person × template) set for the baked feature.
- **Sample reproducibility:** `--limit-people 500 --seed 0` is reproducible; a
  different seed differs; neither is the first 500.
