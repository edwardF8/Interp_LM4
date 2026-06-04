# writeFeatures.ipynb — feature-in-node rank tester

**Date:** 2026-06-04
**Status:** Design approved (pending spec review)

## Purpose

Test a hypothesis of the form: *"Does feature `L{layer}/{idx}` show up among the
direct input features of a given output-token node, across many people and many
prompt phrasings?"*

The canonical first use case: take every person born in August, and measure where
feature `L3/4768` ranks among the direct inputs to the ` August` output (logit)
node in the attribution graph — top-1 / top-2 / … / >5 / absent — aggregated over
all people and all 46 bio templates.

The notebook is built so **future iterations swap only two things — the people
subset and the feature** — via one clearly-marked config cell. Everything else is
stable, hidden machinery.

## Scope

- **In scope:** a new standalone notebook `clts/writeFeatures.ipynb`; the rank
  metric, the per-graph computation, the cross-(person × template) aggregation,
  caching, and a saved report.
- **Out of scope:** changes to `inference_explorer.ipynb` (the earlier request to
  simplify its section 3a was dropped); changes to `build_attribution_graph.py` or
  `circuit_tracer`.

## Key decisions (locked)

1. **Ranking metric = direct edges into the node.** For output node `L`, the
   incoming edge vector is `adjacency_matrix[L_row, :n_features]` (rows = targets,
   cols = sources — confirmed in `circuit_tracer/graph.py`). No multi-hop.
2. **Feature position handling = sum across positions.** The target feature is a
   `(layer, idx)` pair; it can fire at several token positions (several feature
   nodes). Aggregate by `(layer, idx)`, summing the incoming edge weights across
   all positions, then rank. Separately **flag** when the feature fired at ≥2
   positions spanning ≥ `POS_SPAN_FLAG` tokens.
3. **Template set = all 46 dataset bio templates** (`data.bio_text.FIELD_SPECS`),
   each truncated right before the date so the next token is the month.
4. **Ranking direction:** signed edge weight by default (promoters on top);
   `RANK_BY_ABS = True` switches to magnitude.
5. **Two complementary metrics, both from one cached raw edge list:**
   (a) the summed-edge headline rank above, and (b) a strict **"meaningful
   contribution across tokens"** metric — rank every feature *node* and require the
   target feature to have ≥2 distinct positions *each* in the node-level top-K, so a
   "one big + one small" pair does not qualify. The loose position flag still reports
   those cases for contrast.
6. **Randomized people:** capping a subset draws a seeded random sample, never the
   first N.
7. **Output:** print inline **and** save a per-run report file.
8. **Location:** pure, model-independent logic lives in a tested module
   `clts/writefeatures.py` (unit-tested in `tests/test_writefeatures.py`), matching
   the repo pattern where heavy logic sits in `clts/*.py` and notebooks are thin
   wrappers. The notebook `clts/writeFeatures.ipynb` holds the editable surfaces
   (model config, people/feature EDIT cell), the model-dependent glue
   (`attribute_fast`, `node_input_edges`, cache), and the run/report cell, importing
   the pure logic. (Chosen over a pure-notebook layout because the venv has no
   `nbconvert` to test a notebook headless, and the repo already tests `clts/*` via
   pytest.)
9. **Non-feature nodes labeled by token role (unified co-influencer view).** The
   "other nodes that influence the output" view is a **unified, labeled ranking** of
   ALL input nodes to the logit — features (`L3 F4768`), **error** nodes
   (`err@<role>@L<layer>`), and **token/embedding** nodes (`tok@<role>`) — ranked
   together by edge. Error/token nodes are abstracted by the **role of the token they
   sit on** (see the token-role section), so labels are stable across people and the
   46 templates. Verified mechanics: error block is **layer-major**, decode
   `layer, pos = divmod(j - n_features, n_tokens)`; token block is `pos = j - error_end`
   (`circuit_tracer/graph.py:49-64`, `create_graph_files.py:54-61`). Empirically, on
   ` Gage Wyatt Clay was born on → August`, edge mass into the logit splits
   **feature 93.9% / error 6.0% / token 0.03%**, and ~62% of the error mass is a
   single node: `err@last_name[final]@L2` (on ` Clay`). Token nodes are included but expected
   to rank near-zero.
10. **The headline metric stays unskewed (guardrail).** The target-feature rank
   histogram and the cross-token metric are computed over **feature columns only** —
   adding error/token nodes is an additive, read-only second pass that feeds ONLY the
   unified co-influencer view, never the feature-rank computation. A regression test
   asserts the feature-rank output is byte-identical with the error view on vs. off.

## Architecture

Standalone notebook, blocks in run order:

1. **Imports (hidden, run once):** stdlib + torch + repo helpers.
2. **Model config (VISIBLE, editable):** a clearly-labeled cell holding the five
   model constants, separate from the people/feature EDIT cell so it isn't touched
   every run but is a first-class swappable knob (see below). Re-run this + the model
   load cell to trace a different model/CLT.
3. **Model load + helpers (hidden, run once):** load the circuit-tracer replacement
   `model` once (cached, ~10 s first time); people-selection + template helpers.
4. **Machinery (hidden):** the rank computation, the sweep, the aggregation/report,
   and the cache.
5. **EDIT cell (VISIBLE):** the people/feature config surface.
6. **Run + report cell:** calls the sweep and prints/saves the four outputs.

### Model config cell (visible, edit to swap models)

```python
# ===================== MODEL CONFIG (edit to swap models) =====================
MODEL_DIR = REPO / "model/grid-L4-H6"
CLT_DIR   = REPO / "clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final"
DATA_DIR  = REPO / "data/bioS_N-Bd_final_grid"
SCAN_NAME = "grid-L4-H6"   # also routes the cache/report dir: clt_feature_explorer/<scan>/hyptest/
DEVICE    = "cpu"
# ==============================================================================
```

`SCAN_NAME` namespaces the cache and saved reports, so swapping the model writes to
its own folder with no cross-contamination. Defaults match `inference_explorer.ipynb`
(`grid-L4-H6`, the only trained CLT today).

### The EDIT cell (the only thing future-Claude changes)

```python
# ===================== EDIT THIS CELL =====================
PEOPLE          = people_in_month("August")  # or people_by_ids([...]) / people_by_idx([...]) / sample_in_month("August", 20)
TARGET_FEATURE  = (3, 4768)                  # (layer, feature_idx) to locate
TARGET          = "month"                    # "month" = each person's own birth-month token; or pin e.g. " August"
TEMPLATES       = "all"                      # all 46 dataset templates, or list of indices, or list of strings
N_PEOPLE_CAP    = 20                         # cap on subset size; capping RANDOM-SAMPLES the pool (None = use all)
SEED            = 0                           # RNG seed for the random people sample (reproducible)
TOP_K           = 10                         # co-influencer table depth (top-K inputs per node); does NOT change the rank histogram
MULTI_TOK_TOP_K = 5                           # node-level top-K used by the "meaningful across tokens" metric (strict)
POS_SPAN_FLAG   = 3                          # loose flag: feature fired at >=2 positions spanning >= this many tokens
RANK_BY_ABS     = False                      # False = signed (promoters first); True = |edge|
TEMPLATE_WORD_LABELS = {"born", "birth", "day", "date"}  # template words kept as literal roles; others -> template:other
INCLUDE_TOKEN_NODES  = True                  # include token/embedding nodes in the unified co-influencer view (expected near-zero)
# ==========================================================
```

`PEOPLE` is a list of `(dataset_idx, person)` tuples (same shape as
`people_in_month` returns). When `N_PEOPLE_CAP` is set and smaller than the pool,
the subset is a **random sample** drawn with `SEED` (not the first N), so people
choice is randomized but reproducible across re-runs with the same seed.

## Components / helpers

**Module vs notebook split.** Essentially all logic lives in `clts/writefeatures.py`
via dependency injection (functions take `model` / `graph` / `sampler` / `tokenizer`
as arguments), so everything is importable and testable:
- *Pure (unit-tested with synthetic data):* the adjacency indexing
  (`incoming_feature_edges`, `find_logit_row`, the error/token decode in
  `node_input_all`), `aggregate_by_feature`, `feature_rank`, `feature_multitoken`,
  `rank_bucket`, `sample_people`, `resolve_templates`, `template_prompt` (string
  path), `resolve_target`, `token_roles`, `label_nodes`, `build_report`,
  `save_report`, and the cache (`load_or_build_edges`, tested with a stub builder).
- *Model-dependent (artifact-gated integration test, mirroring
  `tests/test_attribution.py`'s `_HAS_ARTIFACTS` gate):* `attribute_fast`,
  `node_input_all`, the people selectors, and `run_hypothesis`.

The notebook `clts/writeFeatures.ipynb` is a thin wrapper: model-config cell, model
load, tiny partials so the EDIT cell reads cleanly (`people_in_month("August")`),
the EDIT cell, and a run/report cell that calls `writefeatures.run_hypothesis(...)`
then `build_report` / `save_report`. No business logic in the notebook.

### Loading helpers
- `name_in_vocab(name)`, `people_in_month(month, in_vocab_only=True)` — copied from
  `inference_explorer.ipynb`.
- `people_by_ids(ids)`, `people_by_idx(idxs)` — thin wrappers returning the same
  `(dataset_idx, person)` tuple list, so the EDIT cell has one consistent shape
  regardless of how people are chosen.
- `sample_in_month(month, n, seed=0)` — random sample of `n` in-vocab people born in
  `month` (replaces an order-based "first n"; randomized by design).
- `sample_people(pool, n, seed)` — the generic randomizer the run uses to apply
  `N_PEOPLE_CAP`/`SEED` to any `PEOPLE` pool (returns the pool unchanged if
  `n is None` or `n >= len(pool)`).
- `MONTH_STRINGS`, plus the 46 templates from `data.bio_text.FIELD_SPECS["birthday"]`.

### Prompt + target helpers
- `template_prompt(person, t_idx)` — render bio template `t_idx`, truncate right
  before `{birthday}` (rstrip), guaranteeing the next token is the month. Reuses
  the `recall_prompt` truncation idea but for an arbitrary template index.
- `resolve_target(TARGET, person)` — `"month"` → `" " + person["birthmonth"]`;
  otherwise use `TARGET` literally as the token string.

### Core metric
- `attribute_fast(prompt, target)` → `Graph`. Calls `circuit_tracer.attribute(...)`
  with the **in-memory `model`** (no disk reload, no viewer-file writing). Params:
  `max_n_logits=10, desired_logit_prob=0.95, max_feature_nodes=4096, batch_size=256,
  offload=None`, target forced via `attribution_targets=[target]`.
- `node_input_edges(graph, target_token)` → the **raw per-node edge list**
  `[{"layer": L, "pos": p, "fidx": f, "edge": w}, ...]`, one entry per feature node,
  for the edges into the target logit. Implementation:
  - find the logit row whose token matches `target_token` (match against
    `graph.logit_targets` / `graph.logit_token_ids`; do **not** assume row 0);
  - `e = graph.adjacency_matrix[logit_row, :n_features].cpu()`;
  - for each feature node `i`: `(layer, pos, fidx) = graph.active_features[
    graph.selected_features[i]].tolist()`, with edge `e[i]`.
  This list is **independent of which feature we test** → it is what gets cached.
  Both metrics below derive from it (no re-attribution needed when the feature or
  thresholds change).

Two derived metrics, both computed from the cached edge list:

- **(headline) summed-edge rank** — `aggregate_by_feature(edges)` groups the list by
  `(layer, fidx)`, summing `edge` and collecting `positions`; then
  `feature_rank(agg, target_feature, rank_by_abs)` → `{"rank": int|None,
  "positions": [...], "span": int, "edge": float, "n_features_in_node": int}`.
  Rank = 1-based position of `target_feature` after sorting the aggregated entries by
  `edge` (signed desc, or `abs` desc); `None` ⇒ absent. `span = max(pos) - min(pos)`
  (0 if single position). This is the loose view: one big position can carry it.

- **(new) meaningful contribution across tokens** —
  `feature_multitoken(edges, target_feature, multi_tok_top_k, rank_by_abs)`. Rank
  **every feature node** (not the per-`(layer,fidx)` sum) by `edge` (signed or abs).
  For the target feature, gather its nodes as `[{"pos": p, "node_rank": r, "edge": w},
  ...]`. Define `n_meaningful = #{distinct positions whose node_rank ≤ multi_tok_top_k}`
  and `is_meaningful_multitoken = n_meaningful ≥ 2`. Because **each** contributing
  position must independently land in the node-level top-K, a "one large + one tiny"
  pair does **not** qualify — which is exactly the distinction from the loose
  `span`/positions flag above.

### Non-feature nodes & token-role labeling (unified co-influencer view)
The cached object is extended to the **full** logit-row decomposition (still
feature-independent), so the headline metric and the unified view share one cache:
- `node_input_all(graph, target_token, tokenizer)` → `{"features": [{layer,pos,fidx,
  edge}], "errors": [{layer,pos,edge}], "tokens": [{pos,edge}]}`. Errors decode via
  `layer, pos = divmod(j - n_features, n_tokens)` (layer-major, verified); tokens via
  `pos = j - error_end`. The `"features"` list is exactly the old `node_input_edges`
  output, so `feature_rank`/`feature_multitoken` are unchanged and read **only**
  `["features"]` (the unskewed guarantee).
- `token_roles(prompt, person, tokenizer, *, template_word_labels)` → `roles[pos]`
  for every **graph** position (length `n_tokens`, `roles[0] == "BOS"` for the
  prepended BOS — the +1 shift is handled here). Uses the fast tokenizer's
  `return_offsets_mapping=True`; assigns each token to a name field by char-overlap
  with the known `{first} {middle} {last}` spans. Roles: `BOS`, `first_name`,
  `middle_name`, `last_name`, and exactly **one** `last_name[final]` per graph (the
  final subword of the last name — always emitted, whether the last name is one token
  or several; earlier subwords of a multi-token last name stay `last_name`),
  `template:<w>` for `w ∈ template_word_labels`, else `template:other`; the **final**
  position also carries an orthogonal `recall` marker (kept alongside its lexical role).
- `label_nodes(node_all, roles)` → labeled rows for the unified table: features keep
  `L{layer} F{fidx}` (aggregated by `(layer,fidx)`, summed across positions); error
  nodes → `err@{role}@L{layer}` (aggregated by `(role, layer)`, summed across the
  positions sharing that role); token nodes → `tok@{role}` (aggregated by `role`).
  Each row: `{kind, label, edge}`.

`node_all`/`roles` are **only** consumed by the unified co-influencer aggregation
(output 4 below). The feature-rank path never sees error/token rows.

### Sweep + cache
- Cache keyed by a hash of `(prompt, target_token, build_params)` → stored
  `node_input_all` dict (feature + error + token blocks; torch-saved). Dir:
  `storage_root()/clt_feature_explorer/<scan>/hyptest/`. (Roles are recomputed at
  report time from the prompt/person — cheap, and lets `TEMPLATE_WORD_LABELS` change
  without invalidating the cache.)
- `run_hypothesis(people, target_feature, TARGET, templates, *, n_cap, seed, top_k,
  multi_tok_top_k, rank_by_abs, ...)`: applies `sample_people(people, n_cap, seed)`
  first (randomized cap), then for each `(person, t_idx)` builds/looks-up the cached
  edge list and computes both metrics. Returns a list of per-(person, template)
  records: `{ds_idx, id, name, t_idx, prompt, target_token, rank, span, positions,
  n_positions, multitoken: {n_meaningful, is_meaningful, per_pos}, edges_topk}`
  plus the cached edge lists for co-influencer aggregation. Prints a per-person
  progress line.

### Aggregation + report
- `report_hypothesis(records, top_k, pos_span_flag)` prints and returns:
  1. **Rank histogram** — counts and % across the **fixed** buckets
     `{top1, top2, top3, top4, top5, 6–10, >10, absent}` over all
     (person × template) records. These buckets are independent of `TOP_K`.
  2. **Loose multi-position summary** — fraction of graphs where the feature fired at
     ≥2 positions (any strength); span distribution; the list of flagged
     `(person, template)` cases with span ≥ `POS_SPAN_FLAG`. This is the "useful to
     know, even if it's large+small" view.
  3. **Meaningful cross-token contribution** (the strict new metric) — fraction of
     graphs where `is_meaningful_multitoken` is true (≥2 positions each in the
     node-level top-`MULTI_TOK_TOP_K`); distribution of `n_meaningful`; and a
     side-by-side count vs. the loose flag (2) so the gap — graphs that fire at ≥2
     positions but are *not* meaningfully multi-token (the large+small cases) — is
     explicit.
  4. **Unified co-influencers (features + error + token nodes)** — across all graphs,
     every input node ranked **together** by edge into the logit and labeled by kind
     + role (`L3 F4768`, `err@last_name[final]@L2`, `tok@first_name`); reports how often each
     label lands in the per-graph top-`TOP_K` of the unified ranking, with mean edge
     weight. This is where **error contribution is surfaced** next to features.
     `INCLUDE_TOKEN_NODES=False` drops the `tok@*` rows. This view is additive — it
     does **not** affect outputs 1–3.
- **Saved report:** write `clt_storage/clt_feature_explorer/<scan>/hyptest/
  report_L{layer}F{idx}_<subset-slug>.json` (and a flat `.csv` of the per-record
  ranks + multitoken fields) containing the config, the histogram, both
  multi-position summaries (loose + meaningful), and the unified co-influencer table.
  `<subset-slug>` derives from the people selection + seed (e.g. `august-n20-s0`).

## Data flow

EDIT cell sets config → `run_hypothesis` random-samples the people pool
(`sample_people`, seeded), iterates `people × templates`, building each `prompt` and
`target_token`, computing/caching `node_input_all` (feature + error + token blocks),
then `feature_rank` (summed) and `feature_multitoken` (node-level) over the feature
block, plus the labeled unified rows (`token_roles` + `label_nodes`) per record →
`build_report` aggregates the four outputs → prints inline + writes JSON/CSV.

## Error handling

- Skip a (person, template) record if `attribute_fast` raises (e.g. OOV token) — log
  a one-line warning and continue; report how many were skipped.
- If the forced `target_token` is not a single token, skip that record with a note.
- If `target_token`'s logit row isn't present in the graph, record it as `absent`
  with a distinct reason, not a crash.

## Cost / risk

- 46 templates × N people `attribute()` calls on CPU. Mitigations: caching,
  `N_PEOPLE_CAP`, per-person progress print, and the ability to set
  `TEMPLATES` to a small list while iterating.
- **First implementation step:** verify `attribute_fast` runs on the in-memory
  `model`, returns a graph whose `adjacency_matrix` indexes as expected, and that
  the forced-target logit row is findable — on one (person, template) pair, timed.

## Testing / verification

- Smoke test on one August person, one template: confirm the ` August` logit row is
  found, the edge vector has length `n_features`, and `feature_rank` returns a
  sane rank for a known strong feature.
- Multi-token check: construct/inspect a record where the feature has one strong and
  one weak position — confirm it counts in the loose multi-position summary but
  **not** in the meaningful cross-token metric (i.e. the two metrics diverge as
  intended).
- Randomization check: with a fixed `SEED`, the sampled subset is reproducible; with
  a different `SEED`, it differs — and neither is just the first N of the pool.
- Run the full August / `L3/4768` case end to end; confirm the histogram sums to
  `n_sampled × n_templates − skipped`, and the saved JSON/CSV match the printout.
- Re-run after editing only `TARGET_FEATURE` (or `MULTI_TOK_TOP_K`): confirm it
  returns from cache (no new `attribute()` calls), since both metrics derive from the
  cached object.
- **Token-role check:** on the prompt ` Gage Wyatt Clay was born on` (BOS prepended),
  `token_roles` yields `["BOS", "first_name", "first_name", "middle_name",
  "last_name[final]", "template:other", "template:born", "template:other(recall)"]` —
  the single-token last name `Clay` is `last_name[final]` (always one per graph), and
  the final position carries the `(recall)` suffix. A multi-token name
  (`Rawlings → Raw|lings`) yields `last_name` then `last_name[final]`.
- **Error decode check:** synthetic adjacency — an error column at index
  `n_features + 2*n_tokens + 4` decodes to `(layer=2, pos=4)`.
- **Unskew guardrail (regression):** `build_report` produces a byte-identical
  rank histogram + cross-token summary whether the unified view includes error/token
  nodes or not; the feature-rank path reads only `node_all["features"]`.
- **Empirical anchor:** on the Gage/Clay graph, `err@last_name[final]@L2` is the top error
  row into the ` August` logit and token rows are near-zero — matching the verified
  6.0% / 0.03% error/token edge-mass split.
