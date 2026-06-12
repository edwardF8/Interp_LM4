# Intervention Explorer notebook — design

**Date:** 2026-06-12
**Artifact:** `clts/intervention_explorer.ipynb` (new, standalone)
**Kernel:** `clts/.venv-ct` (circuit-tracer 0.4.1), same as the other clts notebooks.

## Purpose

Load a saved attribution circuit (or build one for a chosen person), then explore
**feature interventions** (ablate / clamp CLT features) and **attention
interventions** (head knockouts, edge cuts) against it, and **validate** that the
circuit's edge weights predict what constrained interventions actually do.

Companions: `inference_explorer.ipynb` (inference + graph building),
`explore_attribution.ipynb` (primary attribution/labeling). This notebook shares
the same `clt_storage/clt_graphs/<scan>/explore/` dir, so circuits built here
appear in the same viewer dropdowns and vice versa.

## Approach (chosen)

Standalone notebook with all helpers in hidden cells, mirroring
`inference_explorer.ipynb`'s style (knob cells you edit + re-run; machinery in
collapsible cells; loud asserts). No new `.py` module — interventions are
interactive exploration; extract a module later if sweeps outgrow the notebook.

Rejected: thin notebook + `clts/interventions.py` (premature; adds reload
ceremony); new section inside `inference_explorer.ipynb` (user asked for a
notebook; that one is already long).

## Structure

### §0 · Loading (hidden, run once)

Same pattern as `inference_explorer.ipynb`: `CLT_CHOICES` registry keyed
L1/L2/L4/L8, `CHOICE` knob, paths via `clts.storage.storage_root()`, cached
`model = load_replacement_model(...)` (set `model = None` to force reload),
`BioSampler` + `CondensedTokenizer` in-vocab name check (leading-space form
`f" {first} {last}"`), `pick_person` / `people_in_month` helpers copied from
inference_explorer.

### §1 · Pick a person, get their circuit

Knob cell: `PERSON_IDX` **or** `MONTH` + `SUBSET_IDX` (month-filter mode), plus
`TEMPLATE` (default `"{name} was born on the memorable date of"`) and
`GRAPH_TARGET` (None = auto, same semantics as `feature_graph`).

`load_circuit(prompt, target=None)`:

1. Slugify the prompt exactly as `feature_graph()` does
   (`re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:60]`).
2. If `<explore>/<slug>.pt` exists → `Graph.from_pt` (instant); report
   `📂 loaded saved circuit`. Read `.report.json` sidecar if present.
3. Else → `clts.build_attribution_graph.build_graph(...)` into the shared
   explore/ dir (writes `.pt`, viewer `.json`, `.report.json`); report
   `🆕 built new circuit`.
4. Sanity checks: graph scan matches `SCAN_NAME` (assert);
   `model.get_activations(graph.input_tokens)` recomputed and compared to
   `graph.activation_values` at the selected features — **warn** on drift
   (circuit built with a different CLT/model than currently loaded).
5. Stash everything in a `_CIR` dict (graph, prompt, tokens, target token ids,
   influence ranking) — the session handle all later sections read.
6. Print a ranked top-features table: rank, `L{layer} F{idx}`, position + the
   token at that position, activation, node influence
   (`compute_node_influence`, logits weighted by `logit_probabilities`, same
   math as `show_top_features`). Rank numbers are the handles used in §2.

Secondary cell: `list_circuits()` — scan `<explore>/*.pt`, numbered table with
slug, prompt (from the `.pt`'s `input_string` or the report sidecar), target
token, replacement score. `load_circuit_by_index(i)` loads one directly,
bypassing the person picker.

### §2 · Feature interventions

Knob cell:

```python
INTERVENTIONS = [
    ablate(rank=1),                    # set top-ranked circuit feature to 0
    clamp(rank=3, value=50.0),         # set by rank to a chosen value
    set_feature(layer=3, feat=1438, pos=-1, value=0.0),   # explicit tuple
]
run_intervention(INTERVENTIONS, constrained=False, freeze_attention=True)
```

- `ablate`/`clamp` resolve rank → `(layer, pos, feature_idx, value)` via the
  `_CIR` ranking; `set_feature` is the raw escape hatch.
- `run_intervention` runs `model.feature_intervention` on
  `graph.input_tokens` (positions align with the circuit exactly, BOS
  included). `constrained=True` ⇒ `constrained_layers=range(model.cfg.n_layers)`
  (direct-effects regime; matches what the graph predicts).
- Report: clean vs intervened top-k tokens side by side; Δp / Δlogit / Δrank
  for the circuit's target token; the 10 features whose activations moved most
  (from the returned activation cache vs the clean cache).

### §3 · Attention interventions

- `show_attention(layer)` — per-head pattern heatmaps for the circuit prompt
  (≤8 heads, short prompts: cheap matplotlib grid). Orients before cutting.
- Hook builders: `zero_head(layer, head)` (zeros `hook_z[:, :, head, :]`),
  `cut_edge(layer, head, dst, src, renorm=True)` (zeros a `hook_pattern` edge,
  optionally renormalizes the dst row).
- `run_attention_intervention(attn_hooks)` — `model.run_with_hooks` on the
  circuit tokens, same before/after report as §2.
- Combined runner: `run_intervention(INTERVENTIONS, attn_hooks=[...])` builds
  feature hooks via `model._get_feature_intervention_hooks(...,
  freeze_attention=False)` and appends the attention hooks — one owner of
  `hook_pattern` (never mix freeze + edit on the same hook point).

### §4 · Validation — does the circuit predict interventions?

For the top-N (default 10) features by influence:

1. Ablate each alone, `constrained=True`, `freeze_attention=True`.
2. Actual effect: Δ(target logit) between clean and intervened runs.
3. Predicted effect: −A[target-logit row, feature column] from
   `graph.adjacency_matrix` (the direct feature→logit edge; exact edge-weight
   semantics verified against circuit-tracer's attribution code during
   implementation and noted in the cell).
4. Output: table (feature, predicted Δ, actual Δ, ratio) + scatter with
   correlation.

Known caveat, stated in the section text: the adjacency edge is only the
*direct* feature→logit path; features acting mainly through intermediate
features will under-predict. The scatter shows this rather than hiding it.
L1 circuits are the cleanest validation target (single layer — nothing to
propagate through).

## Error handling

- Loud asserts in §0 for missing MODEL_DIR / CLT_DIR / DATA_DIR (copied
  pattern).
- §1: assert explore dir exists or is creatable; assert scan match.
- §§2–4: `NameError("run load_circuit(...) first")` guards via the `_CIR`
  stash, mirroring inference_explorer's `_LAST` pattern.
- Activation-drift check warns (does not abort): drift means the circuit was
  built with different artifacts and intervention results won't line up.

## Verification

Execute the notebook end-to-end with `CHOICE="L1"` via the `.venv-ct`
interpreter against an existing saved L1 circuit (several exist, e.g.
`gianna-adeline-rawlings-was-born-on-the-memorable-date-of`). Additionally
sanity-check §4 numbers on L1, where constrained-intervention semantics are
cleanest.
