# CLT Attribution Graph — Run Guide

Build CLT attribution graphs for the BioS Llama models using
[circuit-tracer](https://github.com/decoderesearch/circuit-tracer) v0.4.1.
The pipeline spans PSC (feature dashboard generation) and Mac (graph build +
interactive viewer).

---

## Environment

All attribution code runs in the isolated venv `clts/.venv-ct` (Python 3.11,
circuit-tracer v0.4.1). It is **separate** from the training/SAE venv at
`CRL-Interp/.venv` — circuit-tracer pins `transformers<=4.57.3` which
conflicts with the training stack.

Setup instructions: `clts/circuit_env/README.md`.

**eps note:** the model is built at TransformerLens default eps=1e-5 to match
how the CLT was trained/evaluated — NOT the HF checkpoint's rms_norm_eps=1e-6.
See `clts/tl_model.py` for the rationale.

---

## Storage

`clts/storage.py:storage_root()` resolves in order:

1. `$CLT_STORAGE_ROOT` if set
2. PSC path `/jet/home/friedmae/data_storage/LM4_Results` if it exists
3. Repo-root `clt_storage/` (Mac fallback; gitignored)

| Artifact | Path |
|----------|------|
| Attribution graphs | `storage_root()/clt_graphs/<scan>/<slug>/` |
| Feature dashboards | `storage_root()/clt_features/<scan>/<idx>.json` |

---

## Step 1 — PSC: generate feature dashboards (one-time per CLT)

Run on PSC (needs CUDA; uses `--device cuda`):

```bash
clts/.venv-ct/bin/python clts/gen_feature_dashboards.py \
    --model-dir model/grid-L4-H6 \
    --clt-dir   clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final \
    --data-dir  data/bioS_N-Bd_final_grid \
    --scan-name grid-L4-H6 \
    --device cuda
# writes storage_root()/clt_features/grid-L4-H6/<idx>.json
```

Key flags (all optional beyond the four required):

| Flag | Default | Notes |
|------|---------|-------|
| `--features-root` | `storage_root()/clt_features` | root dir for JSON files |
| `--n-per-person` | `2` | bios per person in corpus |
| `--context-size` | `64` | tokens per corpus row |
| `--n-people` | all | cap corpus size |
| `--top-k` | `20` | top examples per feature |
| `--n-bins` | `40` | histogram bins |

The file index is `cantor_pair(layer, feature_index)` — matching the
circuit-tracer viewer's `Node.feature_node` encoding.

### Sync to Mac

After dashboards are generated, pull them with the bundled sync script:

```bash
./scripts/sync_from_psc.sh
# full pipeline: bundle (PSC) -> transfer -> extract
# Stages can be run separately: bundle | status | wait | transfer | extract | clean
```

The tar transform rewrites `clt_features/` → `clt_storage/clt_features/`, so
files land exactly where `storage_root()` resolves on the Mac.

---

## Step 2 — Mac: build an attribution graph

```bash
clts/.venv-ct/bin/python clts/build_attribution_graph.py \
    --model-dir model/grid-L4-H6 \
    --clt-dir   clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final \
    --data-dir  data/bioS_N-Bd_final_grid \
    --scan-name grid-L4-H6 \
    --slug      bday-recall
```

Output (written to `storage_root()/clt_graphs/grid-L4-H6/bday-recall/`):

| File | Contents |
|------|----------|
| `bday-recall.json` | viewer-ready graph (scan set to `./data/grid-L4-H6`) |
| `graph-metadata.json` | circuit-tracer metadata index |
| `bday-recall.pt` | raw graph tensor |
| `bday-recall.report.json` | fidelity report (see below) |

Key flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--prompt` | birthday recall auto-prompt | override with any string |
| `--target` | top logit | attribution target token |
| `--graph-dir` | `storage_root()/clt_graphs/<scan>/<slug>` | output dir |
| `--device` | `cpu` | use `cuda` on PSC |
| `--max-feature-nodes` | `4096` | max nodes before pruning |

**Why `--scan-name`?** The graph's `scan` field is written as
`./data/<scan-name>`. The circuit-tracer viewer routes `scan` strings starting
with `./` to the local server; strings without `./` fall through to Anthropic's
CDN (which does not have our custom model's dashboards). Providing the correct
`--scan-name` is required for the viewer to show populated feature cards.

---

## Step 3 — Mac: serve the viewer

```bash
GRAPH_DIR=$(clts/.venv-ct/bin/python -c \
    'import sys; sys.path.insert(0,"."); from clts.storage import storage_root as s; print(s()/"clt_graphs"/"grid-L4-H6"/"bday-recall")')
FEATS_DIR=$(clts/.venv-ct/bin/python -c \
    'import sys; sys.path.insert(0,"."); from clts.storage import storage_root as s; print(s()/"clt_features"/"grid-L4-H6")')

clts/.venv-ct/bin/python clts/serve_ui.py \
    --graph-dir    "$GRAPH_DIR" \
    --features-dir "$FEATS_DIR" \
    --scan-name    grid-L4-H6
# open http://localhost:8032
```

`serve_ui.py` creates a symlink `<graph-dir>/grid-L4-H6 -> <features-dir>` so
the viewer's `GET /data/grid-L4-H6/<idx>.json` requests are satisfied locally.
The symlink is only created once; re-runs are safe.

Flags: `--graph-dir` (required), `--features-dir` (optional), `--scan-name`
(required when `--features-dir` is set), `--port` (default `8032`).

---

## Per-graph fidelity report

`<slug>.report.json` is printed to stdout during the build and written to disk.
Example fields:

```json
{
  "prompt": "Gage Clay was born on",
  "scan_name": "grid-L4-H6",
  "top_logit_token": " the",
  "target_logit_prob": 0.81,
  "replacement_score": 0.62,
  "completeness_score": 0.71,
  "error_influence_share": 0.38,
  "n_feature_nodes_after_pruning": 124
}
```

| Metric | Meaning |
|--------|---------|
| `replacement_score` | fraction of target logit explained by CLT features |
| `completeness_score` | circuit-tracer's completeness score |
| `error_influence_share` | `1 - replacement_score`; fraction unexplained |
| `n_feature_nodes_after_pruning` | feature nodes surviving the pruning step |

**Low `replacement_score` / high `error_influence_share`** means the CLT
under-explains this prompt. Prefer attributing with the CLT that has the
highest `final_eval/ce_recovered` in its training run.

---

## Prompt caveat (important)

The default birthday prompt `"<Name> was born on"` makes the model predict
`" the"` (~0.81 prob) — a date *prefix*, not the date token itself. The bios
format is `"…born on the Nth of Month…"`, so the circuit traced is "predicting
the word 'the'", not the birthday-date recall circuit.

To surface the birthday-DATE recall circuit, end the prompt further along:

```bash
--prompt "Gage Clay was born on the"
# model now predicts the ordinal/date token
```

Or target a later position with `--target`. This is model/prompt behavior, not
a tooling bug.

---

## Validation (optional, gold-standard)

Ablates the most influential CLT feature and reports the change in the
target-logit probability. A large drop confirms the attributed feature is
causally relevant.

```bash
clts/.venv-ct/bin/python clts/validate_graph.py \
    --model-dir model/grid-L4-H6 \
    --clt-dir   clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final \
    --data-dir  data/bioS_N-Bd_final_grid \
    --scan-name grid-L4-H6
```

Flags: `--model-dir`, `--clt-dir`, `--data-dir`, `--scan-name` (all required);
`--prompt`, `--device`, `--max-feature-nodes` (optional).

---

## Acceptance checklist

- [ ] Graph builds without error; report JSON is printed to stdout.
- [ ] `<slug>.json`, `graph-metadata.json`, `<slug>.pt`, `<slug>.report.json`
      are all written under `storage_root()/clt_graphs/<scan>/<slug>/`.
- [ ] `report["replacement_score"]` is finite and in [0, 1].
- [ ] Browser: `http://localhost:8032` renders the graph.
- [ ] Browser: clicking a feature node shows a populated dashboard
      (MANUAL live-browser check — requires synced `clt_features/`).

**Position-0 caveat:** circuit-tracer zeroes position 0 in attribution. Do not
place essential content at the first token position.

---

## Quick reference

```text
clts/.venv-ct/bin/python clts/gen_feature_dashboards.py   # PSC: generate dashboards
./scripts/sync_from_psc.sh                                 # Mac: pull dashboards
clts/.venv-ct/bin/python clts/build_attribution_graph.py  # Mac: build a graph
clts/.venv-ct/bin/python clts/serve_ui.py                 # Mac: serve viewer
clts/.venv-ct/bin/python clts/validate_graph.py           # optional: ablation check
```
