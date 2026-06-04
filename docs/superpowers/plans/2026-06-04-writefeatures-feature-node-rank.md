# writeFeatures Feature-in-Node Rank Tester — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable tester that, for a people subset and a target CLT feature, measures where that feature ranks among the *direct input features* of an output-token node across all 46 bio templates, plus a strict "meaningful across tokens" metric and a common-co-influencers table — surfaced through an easy-to-edit notebook `clts/writeFeatures.ipynb`.

**Architecture:** All logic lives in a unit-tested module `clts/writefeatures.py` (dependency-injected `model`/`graph`/`sampler`/`tokenizer` args, so it is importable and testable). The notebook is a thin wrapper: a model-config cell, model load, an `# EDIT THIS CELL` people/feature surface, and a run/report cell calling the module. Heavy `attribute()` results are cached so swapping the feature or thresholds needs no recompute.

**Tech Stack:** Python 3.11, PyTorch, `circuit_tracer` (the `attribute()` algorithm + `Graph`), pytest (existing `tests/` suite), Jupyter notebook. Run everything with the venv interpreter `clts/.venv-ct/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-04-writefeatures-feature-node-rank-design.md`

---

## Key facts the implementer must know

- **Adjacency convention (confirmed in `circuit_tracer/graph.py`):** `adjacency_matrix[target, source]`. Rows = targets. To get the direct edges *into* a logit node, read its **row**: `A[logit_row, :n_features]`.
- **Node layout in the matrix:** `[features (n_features) | error (n_tokens*n_layers) | tokens (n_tokens) | logits (n_logits)]`. So the logit at rank `k` is row `n_features + n_tokens*n_layers + n_tokens + k`.
- **Feature node → (layer, pos, fidx):** `graph.active_features[graph.selected_features[i]].tolist()` for feature node `i`.
- **Graph fields used:** `adjacency_matrix`, `selected_features`, `active_features`, `logit_targets`, `logit_token_ids` (property, tensor of vocab ids per logit node), `input_tokens`, `cfg.n_layers`.
- **Build a graph:** `from circuit_tracer import attribute; attribute(prompt=..., model=model, attribution_targets=[target_token], max_n_logits=10, desired_logit_prob=0.95, batch_size=256, max_feature_nodes=4096, offload=None, verbose=False)`.
- **Token-id of a target string:** `model.tokenizer.encode(target, add_special_tokens=False)` → list; must be length 1.
- **46 templates:** `from data.bio_text import FIELD_SPECS; FIELD_SPECS["birthday"]["templates"]` (import `util.bio_sampler` first — it sets the `Training_On_LM4` path).
- **Recall-prompt truncation:** render full bio (`sampler.render(person, t_idx)`), find the date string `f"{birthmonth} {birthday}, {birthyear}"`, take `bio[:bio.index(date)].rstrip()`. Next token is then ` <Month>`.
- **Storage root:** `from clts.storage import storage_root` → repo `clt_storage/`. Cache + reports go in `storage_root()/clt_feature_explorer/<SCAN_NAME>/hyptest/`.
- **Tests run with:** `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -v`. Model-dependent tests gate on artifacts existing (mirror `tests/test_attribution.py`'s `_HAS_ARTIFACTS`).

## File Structure

- **Create `clts/writefeatures.py`** — all logic (pure + DI-glue). One responsibility: turn (model, people, feature, target) into ranked-edge records and a report.
- **Create `tests/test_writefeatures.py`** — unit tests (synthetic) + one artifact-gated integration test.
- **Create `clts/writeFeatures.ipynb`** — thin notebook wrapper (config + run/report).
- **Touch nothing else.** (`inference_explorer.ipynb` is explicitly out of scope.)

Data shapes (used throughout):
- **edge** = `{"layer": int, "pos": int, "fidx": int, "edge": float}` (one per feature node).
- **agg** = `{(layer, fidx): {"edge": float_sum, "positions": [int, ...]}}`.
- **record** = `{"ds_idx", "id", "name", "t_key", "prompt", "target_token", "rank": int|None, "bucket": str, "span": int, "positions": [int], "n_positions": int, "n_meaningful": int, "is_meaningful": bool, "node_top_features": [{"layer","fidx","edge"}]}`.

---

## Task 1: Spike — de-risk in-memory `attribute()` reuse + adjacency indexing

Confirms the core assumptions on the real model before writing product code: that `attribute()` can run repeatedly on one in-memory model, that the forced-target logit row is findable, and that `A[logit_row, :n_features]` yields a sensible ranking. Throwaway — not committed.

**Files:**
- Create (temporary): `scripts/_spike_writefeatures.py`

- [ ] **Step 1: Write the spike script**

```python
# scripts/_spike_writefeatures.py  (TEMPORARY — delete after observing output)
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import torch
from clts.export_tokenizer import ensure_hf_tokenizer
from clts.load_replacement_model import load_replacement_model
from util.bio_sampler import BioSampler
from util.condensed_tokenizer import CondensedTokenizer
from circuit_tracer import attribute

MODEL_DIR = REPO / "model/grid-L4-H6"
CLT_DIR   = REPO / "clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final"
DATA_DIR  = REPO / "data/bioS_N-Bd_final_grid"

model = load_replacement_model(MODEL_DIR, CLT_DIR, ensure_hf_tokenizer(DATA_DIR), "grid-L4-H6", device="cpu")
sampler = BioSampler(DATA_DIR / "people.json", fields=("birthday",))
# first August person
person = next(p for p in sampler.people if p["birthmonth"] == "August")
bio = sampler.render(person, 0)
date = f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}"
prompt = bio[:bio.index(date)].rstrip()
target = " August"
print("prompt:", repr(prompt), "target:", repr(target))

for trial in (1, 2):
    t0 = time.time()
    g = attribute(prompt=prompt, model=model, attribution_targets=[target],
                  max_n_logits=10, desired_logit_prob=0.95, batch_size=256,
                  max_feature_nodes=4096, offload=None, verbose=False)
    print(f"trial {trial}: attribute() took {time.time()-t0:.1f}s")

n_features = len(g.selected_features)
n_tokens = len(g.input_tokens)
n_layers = g.cfg.n_layers
n_error = n_tokens * n_layers
logit_ids = g.logit_token_ids.tolist()
tid = model.tokenizer.encode(target, add_special_tokens=False)
print("target ids:", tid, "logit_ids:", logit_ids)
k = next(i for i, x in enumerate(logit_ids) if x == tid[0])
logit_row = n_features + n_error + n_tokens + k
e = g.adjacency_matrix[logit_row, :n_features].cpu()
order = torch.argsort(e, descending=True)[:8]
print(f"n_features={n_features} n_tokens={n_tokens} n_layers={n_layers} logit_row={logit_row}")
print("top-8 direct input features into", repr(target), ":")
for i in order.tolist():
    layer, pos, fidx = g.active_features[g.selected_features[i]].tolist()
    print(f"  L{layer} F{fidx} @pos{pos}  edge={e[i]:.4f}")
```

- [ ] **Step 2: Run the spike**

Run: `clts/.venv-ct/bin/python scripts/_spike_writefeatures.py`
Expected: both trials complete; trial 2 not dramatically slower than trial 1 (confirms reuse is fine); it prints `n_features`, a valid `logit_row`, and a top-8 list of `L# F# @pos#` rows with descending `edge` values. **Record the trial timings and `n_features`** — they inform the cost note and confirm the indexing works.

- [ ] **Step 3: Delete the spike (do not commit it)**

Run: `rm scripts/_spike_writefeatures.py`
Expected: file gone. Nothing to commit for this task.

> If trial 2 is as slow as a fresh model reload, or the logit row isn't found, STOP and report — the rest of the plan assumes in-memory reuse works. (It should: `attribute()` is designed to run repeatedly on a model.)

---

## Task 2: Module skeleton + `rank_bucket` + `sample_people`

**Files:**
- Create: `clts/writefeatures.py`
- Test: `tests/test_writefeatures.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_writefeatures.py
"""Unit tests for clts/writefeatures.py (feature-in-node rank tester)."""
from __future__ import annotations

import clts.writefeatures as wf


def test_rank_bucket_boundaries():
    assert wf.rank_bucket(1) == "top1"
    assert wf.rank_bucket(2) == "top2"
    assert wf.rank_bucket(5) == "top5"
    assert wf.rank_bucket(6) == "6-10"
    assert wf.rank_bucket(10) == "6-10"
    assert wf.rank_bucket(11) == ">10"
    assert wf.rank_bucket(None) == "absent"


def test_sample_people_reproducible_and_random():
    pool = list(range(100))
    a = wf.sample_people(pool, 5, seed=0)
    b = wf.sample_people(pool, 5, seed=0)
    c = wf.sample_people(pool, 5, seed=1)
    assert a == b                       # reproducible with same seed
    assert a != c                       # different seed -> different sample
    assert a != [0, 1, 2, 3, 4]         # not just the first N
    assert len(a) == 5 and set(a) <= set(pool)


def test_sample_people_no_cap_returns_all():
    pool = [(0, "x"), (1, "y")]
    assert wf.sample_people(pool, None, seed=0) == pool
    assert wf.sample_people(pool, 5, seed=0) == pool   # n >= len -> all
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clts.writefeatures'` (module not created yet).

- [ ] **Step 3: Create the module with these two functions**

```python
# clts/writefeatures.py
"""Feature-in-node rank tester.

Given a people subset and a target CLT feature (layer, fidx), measure where that
feature ranks among the DIRECT input features of an output-token node, across many
bio templates. Pure helpers + dependency-injected model glue, so the notebook
`clts/writeFeatures.ipynb` stays a thin wrapper. See
docs/superpowers/specs/2026-06-04-writefeatures-feature-node-rank-design.md.
"""
from __future__ import annotations

import random

_RANK_BUCKETS = ["top1", "top2", "top3", "top4", "top5", "6-10", ">10", "absent"]


def rank_bucket(rank):
    """Map a 1-based rank (or None) to a fixed histogram bucket."""
    if rank is None:
        return "absent"
    if 1 <= rank <= 5:
        return f"top{rank}"
    if 6 <= rank <= 10:
        return "6-10"
    return ">10"


def sample_people(pool, n, seed):
    """Random (seeded) subset of `pool`. Returns all of `pool` when `n` is None or
    `n >= len(pool)`. Never the first N — choice is randomized but reproducible."""
    pool = list(pool)
    if n is None or n >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): rank_bucket + seeded sample_people

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Adjacency indexing — `find_logit_row` + `incoming_feature_edges`

The riskiest pure logic (the row/col convention). Tested with a tiny synthetic adjacency matrix so it's verifiable without the model.

**Files:**
- Modify: `clts/writefeatures.py`
- Test: `tests/test_writefeatures.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_writefeatures.py
import torch


def test_find_logit_row():
    # logit token ids for the 2 logit nodes, in node order
    logit_ids = [101, 202]
    # layout: n_features=3, n_error=4, n_tokens=2  -> logits start at 3+4+2=9
    row, k = wf.find_logit_row(logit_ids, 202, n_features=3, n_error=4, n_tokens=2)
    assert (row, k) == (10, 1)          # second logit -> base 9 + k(1)
    row, k = wf.find_logit_row(logit_ids, 999, n_features=3, n_error=4, n_tokens=2)
    assert row is None and k is None


def test_incoming_feature_edges_reads_target_row():
    # 3 feature nodes; build A so that row 5 (our "logit") has known edges into them.
    N = 6
    A = torch.zeros(N, N)
    A[5, 0] = 0.10   # feature node 0
    A[5, 1] = 0.50   # feature node 1
    A[5, 2] = -0.30  # feature node 2
    A[0, 5] = 9.99   # column into row 0 must be IGNORED (we read row, not column)
    selected_features = torch.tensor([0, 1, 2])
    # active_features rows are [layer, pos, fidx]
    active_features = torch.tensor([[3, 7, 4768],   # node 0
                                    [2, 7, 11],     # node 1
                                    [3, 1, 4768]])  # node 2 (same fidx, diff pos)
    edges = wf.incoming_feature_edges(A, selected_features, active_features, logit_row=5)
    assert edges == [
        {"layer": 3, "pos": 7, "fidx": 4768, "edge": 0.10},
        {"layer": 2, "pos": 7, "fidx": 11,   "edge": 0.50},
        {"layer": 3, "pos": 1, "fidx": 4768, "edge": -0.30},
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "logit_row or incoming" -v`
Expected: FAIL — `AttributeError: module 'clts.writefeatures' has no attribute 'find_logit_row'`.

- [ ] **Step 3: Add the two functions**

```python
# add to clts/writefeatures.py

def find_logit_row(logit_token_ids, target_token_id, *, n_features, n_error, n_tokens):
    """Row index of the logit node whose vocab id == target_token_id.

    Node layout: [features | error | tokens | logits]; logit k is at
    n_features + n_error + n_tokens + k. Returns (row, k), or (None, None) if the
    target token is not among the graph's logit nodes."""
    base = n_features + n_error + n_tokens
    for k, tid in enumerate(logit_token_ids):
        if int(tid) == int(target_token_id):
            return base + k, k
    return None, None


def incoming_feature_edges(adjacency, selected_features, active_features, logit_row):
    """Direct edges INTO `logit_row` from every feature node (reads a ROW, since
    adjacency_matrix is [target, source]). Returns one edge dict per feature node."""
    n_features = len(selected_features)
    e = adjacency[logit_row, :n_features]
    out = []
    for i in range(n_features):
        layer, pos, fidx = active_features[int(selected_features[i])].tolist()
        out.append({"layer": int(layer), "pos": int(pos), "fidx": int(fidx),
                    "edge": round(float(e[i]), 6)})
    return out
```

> Note: `round(..., 6)` keeps the synthetic-test asserts exact (0.10 not 0.0999999). Real edges are floats; 6 dp is plenty for ranking.

- [ ] **Step 4: Run tests to verify they pass**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "logit_row or incoming" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): adjacency indexing (find_logit_row, incoming_feature_edges)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `aggregate_by_feature` + `feature_rank` + `top_node_features`

The headline metric: sum each feature's edges across positions, rank, and bucket.

**Files:**
- Modify: `clts/writefeatures.py`
- Test: `tests/test_writefeatures.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_writefeatures.py

EDGES = [
    {"layer": 3, "pos": 7, "fidx": 4768, "edge": 0.40},
    {"layer": 3, "pos": 1, "fidx": 4768, "edge": 0.20},   # same feature, 2nd position
    {"layer": 2, "pos": 7, "fidx": 11,   "edge": 0.50},
    {"layer": 1, "pos": 7, "fidx": 99,   "edge": -0.90},  # strong inhibitor
]


def test_aggregate_by_feature_sums_positions():
    agg = wf.aggregate_by_feature(EDGES)
    assert agg[(3, 4768)]["edge"] == 0.60
    assert agg[(3, 4768)]["positions"] == [1, 7]
    assert agg[(2, 11)]["edge"] == 0.50


def test_feature_rank_signed_vs_abs():
    agg = wf.aggregate_by_feature(EDGES)
    # signed: (3,4768)=0.60 > (2,11)=0.50 > (1,99)=-0.90  -> rank 1
    r = wf.feature_rank(agg, (3, 4768), rank_by_abs=False)
    assert r["rank"] == 1 and r["positions"] == [1, 7] and r["span"] == 6
    # abs: (1,99)=0.90 > (3,4768)=0.60 > (2,11)=0.50  -> (3,4768) rank 2
    r_abs = wf.feature_rank(agg, (3, 4768), rank_by_abs=True)
    assert r_abs["rank"] == 2


def test_feature_rank_absent():
    agg = wf.aggregate_by_feature(EDGES)
    r = wf.feature_rank(agg, (0, 12345), rank_by_abs=False)
    assert r["rank"] is None and r["positions"] == [] and r["span"] == 0


def test_top_node_features():
    agg = wf.aggregate_by_feature(EDGES)
    top = wf.top_node_features(agg, top_k=2, rank_by_abs=False)
    assert [(t["layer"], t["fidx"]) for t in top] == [(3, 4768), (2, 11)]
    assert top[0]["edge"] == 0.60
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "aggregate or feature_rank or top_node" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'aggregate_by_feature'`.

- [ ] **Step 3: Add the functions**

```python
# add to clts/writefeatures.py

def aggregate_by_feature(edges):
    """Group edge dicts by (layer, fidx): sum `edge` across positions, collect
    sorted positions."""
    agg = {}
    for e in edges:
        key = (e["layer"], e["fidx"])
        slot = agg.setdefault(key, {"edge": 0.0, "positions": []})
        slot["edge"] = round(slot["edge"] + e["edge"], 6)
        slot["positions"].append(e["pos"])
    for slot in agg.values():
        slot["positions"] = sorted(set(slot["positions"]))
    return agg


def _sorted_features(agg, rank_by_abs):
    """(layer, fidx) keys sorted by summed edge, descending."""
    keyfn = (lambda k: abs(agg[k]["edge"])) if rank_by_abs else (lambda k: agg[k]["edge"])
    return sorted(agg.keys(), key=keyfn, reverse=True)


def feature_rank(agg, target_feature, rank_by_abs=False):
    """1-based rank of `target_feature` (layer, fidx) among aggregated features by
    summed edge (signed desc, or |edge| desc). rank=None when the feature is absent
    from the node's inputs."""
    order = _sorted_features(agg, rank_by_abs)
    if target_feature not in agg:
        return {"rank": None, "positions": [], "span": 0, "edge": 0.0,
                "n_features_in_node": len(agg)}
    rank = order.index(target_feature) + 1
    positions = agg[target_feature]["positions"]
    span = (max(positions) - min(positions)) if positions else 0
    return {"rank": rank, "positions": positions, "span": span,
            "edge": agg[target_feature]["edge"], "n_features_in_node": len(agg)}


def top_node_features(agg, top_k, rank_by_abs=False):
    """Top-K (layer, fidx) inputs to the node, for co-influencer aggregation."""
    order = _sorted_features(agg, rank_by_abs)[:top_k]
    return [{"layer": l, "fidx": f, "edge": agg[(l, f)]["edge"]} for (l, f) in order]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "aggregate or feature_rank or top_node" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): summed-edge rank (aggregate_by_feature, feature_rank, top_node_features)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `feature_multitoken` — strict "meaningful across tokens" metric

Ranks every feature NODE (not the per-feature sum); the target feature counts as a meaningful cross-token contributor only if ≥2 distinct positions each land in the node-level top-K. A "one big + one tiny" pair must NOT qualify.

**Files:**
- Modify: `clts/writefeatures.py`
- Test: `tests/test_writefeatures.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_writefeatures.py

def _edges_with_target_at(ranks_and_positions, n_filler=20):
    """Build an edge list where the target feature (9, 4768) sits at given
    node-level ranks. `ranks_and_positions` = list of (desired_rank, pos). Filler
    features occupy the other ranks with descending edges."""
    # Give every node a distinct edge so node-rank == sorted order. Use edge = 1000 - rank.
    edges = []
    target_ranks = {r for r, _ in ranks_and_positions}
    pos_for_rank = {r: p for r, p in ranks_and_positions}
    total = n_filler + len(ranks_and_positions)
    for rank in range(1, total + 1):
        edge = float(total - rank + 1)   # rank 1 -> largest edge
        if rank in target_ranks:
            edges.append({"layer": 9, "pos": pos_for_rank[rank], "fidx": 4768, "edge": edge})
        else:
            edges.append({"layer": 0, "pos": rank, "fidx": 1000 + rank, "edge": edge})
    return edges


def test_multitoken_two_strong_positions_is_meaningful():
    edges = _edges_with_target_at([(1, 7), (3, 2)])   # ranks 1 and 3, distinct positions
    mt = wf.feature_multitoken(edges, (9, 4768), multi_tok_top_k=5)
    assert mt["n_positions"] == 2
    assert mt["n_meaningful"] == 2
    assert mt["is_meaningful"] is True


def test_multitoken_big_plus_tiny_is_not_meaningful():
    edges = _edges_with_target_at([(1, 7), (18, 2)])  # one strong (rank1), one weak (rank18)
    mt = wf.feature_multitoken(edges, (9, 4768), multi_tok_top_k=5)
    assert mt["n_positions"] == 2          # loose multi-position: still fired at 2 positions
    assert mt["n_meaningful"] == 1         # only rank1 clears top-5
    assert mt["is_meaningful"] is False    # the key distinction


def test_multitoken_absent_feature():
    edges = _edges_with_target_at([(1, 7)])
    mt = wf.feature_multitoken(edges, (0, 999999), multi_tok_top_k=5)
    assert mt["n_positions"] == 0 and mt["is_meaningful"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "multitoken" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'feature_multitoken'`.

- [ ] **Step 3: Add the function**

```python
# add to clts/writefeatures.py

def feature_multitoken(edges, target_feature, multi_tok_top_k=5, rank_by_abs=False):
    """Strict cross-token metric. Rank every feature NODE by edge (signed or abs);
    the target feature is a 'meaningful' multi-token contributor iff >=2 distinct
    positions each have a node whose node-level rank <= multi_tok_top_k. A
    one-big-one-tiny pair does NOT qualify (only the big one clears the threshold)."""
    layer, fidx = target_feature
    keyfn = (lambda e: abs(e["edge"])) if rank_by_abs else (lambda e: e["edge"])
    ordered = sorted(edges, key=keyfn, reverse=True)
    per_pos = []
    for node_rank, e in enumerate(ordered, start=1):
        if e["layer"] == layer and e["fidx"] == fidx:
            per_pos.append({"pos": e["pos"], "node_rank": node_rank, "edge": e["edge"]})
    n_positions = len({p["pos"] for p in per_pos})
    meaningful_positions = {p["pos"] for p in per_pos if p["node_rank"] <= multi_tok_top_k}
    n_meaningful = len(meaningful_positions)
    return {"n_positions": n_positions, "n_meaningful": n_meaningful,
            "is_meaningful": n_meaningful >= 2, "per_pos": per_pos}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "multitoken" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): strict meaningful-across-tokens metric (feature_multitoken)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Prompt/target/template helpers — `resolve_target`, `resolve_templates`, `template_prompt`, `birthday_templates`

**Files:**
- Modify: `clts/writefeatures.py`
- Test: `tests/test_writefeatures.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_writefeatures.py

def test_resolve_target():
    person = {"birthmonth": "August"}
    assert wf.resolve_target("month", person) == " August"
    assert wf.resolve_target(" August", person) == " August"   # literal pass-through
    assert wf.resolve_target(None, person) is None


def test_resolve_templates():
    all_t = ["{name} was born on {birthday}.", "{name} arrived on {birthday}."]
    assert wf.resolve_templates("all", all_t) == [(0, 0), (1, 1)]
    assert wf.resolve_templates([1], all_t) == [(1, 1)]
    out = wf.resolve_templates(["{name} popped out on {birthday}."], all_t)
    assert out == [("str0", "{name} popped out on {birthday}.")]


def test_template_prompt_string_path():
    person = {"first_name": "Gage", "middle_name": "Wyatt", "last_name": "Clay",
              "birthmonth": "August", "birthday": 24, "birthyear": 1712}
    # string template: substitute {name}, cut before {birthday}, single leading space
    p = wf.template_prompt(person, "{name} popped out on {birthday}.", sampler=None)
    assert p == " Gage Wyatt Clay popped out on"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "resolve_target or resolve_templates or template_prompt" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'resolve_target'`.

- [ ] **Step 3: Add the functions**

```python
# add to clts/writefeatures.py

def resolve_target(target, person):
    """Map the TARGET knob to a token string. "month" -> the person's own birth
    month as a leading-space token; None -> None; anything else -> literal."""
    if target is None:
        return None
    if target == "month":
        return " " + str(person["birthmonth"])
    return target


def birthday_templates():
    """The 46 dataset bio templates for the birthday field (training-identical)."""
    import util.bio_sampler  # noqa: F401  (sets the Training_On_LM4 import path)
    from data.bio_text import FIELD_SPECS
    return list(FIELD_SPECS["birthday"]["templates"])


def resolve_templates(templates, all_templates):
    """Normalise the TEMPLATES knob to a list of (t_key, t_val):
      "all"            -> [(i, i) for all 46]              (t_val int -> render via sampler)
      [int, ...]       -> [(i, i) for those indices]
      [str, ...]       -> [("str{j}", str)]               (t_val str -> format directly)
    """
    if templates == "all":
        return [(i, i) for i in range(len(all_templates))]
    if templates and isinstance(templates[0], int):
        return [(i, i) for i in templates]
    return [(f"str{j}", s) for j, s in enumerate(templates)]


def _full_name(person):
    return f"{person['first_name']} {person['middle_name']} {person['last_name']}"


def template_prompt(person, t_val, sampler):
    """Build the recall prompt that ends right before the birth date (so the next
    token is the month). int t_val -> render the trained template via `sampler` and
    truncate; str t_val -> substitute {name} and cut at {birthday}."""
    if isinstance(t_val, int):
        bio = sampler.render(person, t_val)
        date = f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}"
        return bio[:bio.index(date)].rstrip()
    head = t_val.split("{birthday}")[0].replace("{name}", _full_name(person))
    if head and not head[0].isspace():
        head = " " + head
    return head.rstrip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "resolve_target or resolve_templates or template_prompt" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): target/template helpers (resolve_target, resolve_templates, template_prompt)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Report assembly — `build_report`, `format_report`, `save_report`

**Files:**
- Modify: `clts/writefeatures.py`
- Test: `tests/test_writefeatures.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_writefeatures.py
import json


def _rec(rank, n_positions, span, is_meaningful, n_meaningful, top_feats):
    return {"ds_idx": 1, "id": 22, "name": "Gage Clay", "t_key": 0,
            "prompt": " Gage Clay was born on", "target_token": " August",
            "rank": rank, "bucket": wf.rank_bucket(rank), "span": span,
            "positions": [1, 7] if n_positions == 2 else [7],
            "n_positions": n_positions, "n_meaningful": n_meaningful,
            "is_meaningful": is_meaningful,
            "node_top_features": top_feats}


def test_build_report_aggregates():
    records = [
        _rec(1, 2, 6, True, 2, [{"layer": 3, "fidx": 4768, "edge": 0.6}]),
        _rec(3, 2, 5, False, 1, [{"layer": 3, "fidx": 4768, "edge": 0.4}]),
        _rec(None, 0, 0, False, 0, []),
    ]
    rep = wf.build_report({"records": records, "n_skipped": 1, "n_sampled": 2},
                          top_k=10, pos_span_flag=3, multi_tok_top_k=5,
                          config={"target_feature": [3, 4768]})
    h = rep["rank_histogram"]["counts"]
    assert h["top1"] == 1 and h["top3"] == 1 and h["absent"] == 1
    assert rep["loose_multipos"]["n_ge2_positions"] == 2
    assert rep["meaningful_crosstoken"]["n_meaningful"] == 1
    assert rep["meaningful_crosstoken"]["vs_loose"]["gap"] == 1   # 2 loose - 1 meaningful
    co = rep["co_influencers"][0]
    assert (co["layer"], co["fidx"], co["count"]) == (3, 4768, 2)
    assert abs(co["mean_edge"] - 0.5) < 1e-9


def test_format_report_is_string():
    rep = wf.build_report({"records": [_rec(1, 1, 0, False, 1, [])], "n_skipped": 0,
                           "n_sampled": 1}, top_k=10, pos_span_flag=3,
                          multi_tok_top_k=5, config={"target_feature": [3, 4768]})
    s = wf.format_report(rep)
    assert isinstance(s, str) and "top1" in s


def test_save_report_writes_files(tmp_path):
    records = [_rec(1, 2, 6, True, 2, [{"layer": 3, "fidx": 4768, "edge": 0.6}])]
    rep = wf.build_report({"records": records, "n_skipped": 0, "n_sampled": 1},
                          top_k=10, pos_span_flag=3, multi_tok_top_k=5,
                          config={"target_feature": [3, 4768]})
    paths = wf.save_report(rep, records, tmp_path, layer=3, fidx=4768, subset_slug="august-n1-s0")
    assert paths["json"].endswith("report_L3F4768_august-n1-s0.json")
    loaded = json.loads(open(paths["json"]).read())
    assert loaded["rank_histogram"]["counts"]["top1"] == 1
    csv_text = open(paths["csv"]).read()
    assert "rank" in csv_text and "Gage Clay" in csv_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "build_report or format_report or save_report" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'build_report'`.

- [ ] **Step 3: Add the functions**

```python
# add to clts/writefeatures.py
import csv as _csv
import json as _json
from collections import defaultdict
from pathlib import Path


def build_report(result, *, top_k, pos_span_flag, multi_tok_top_k, config):
    """Aggregate per-(person, template) records into the four outputs."""
    records = result["records"]
    n = len(records)

    hist = {b: 0 for b in _RANK_BUCKETS}
    for r in records:
        hist[r["bucket"]] += 1
    hist_pct = {b: (hist[b] / n if n else 0.0) for b in _RANK_BUCKETS}

    loose_ge2 = sum(1 for r in records if r["n_positions"] >= 2)
    span_dist, flagged = {}, []
    for r in records:
        span_dist[r["span"]] = span_dist.get(r["span"], 0) + 1
        if r["n_positions"] >= 2 and r["span"] >= pos_span_flag:
            flagged.append({"id": r["id"], "name": r["name"], "t_key": r["t_key"],
                            "span": r["span"]})

    meaningful_ge2 = sum(1 for r in records if r["is_meaningful"])
    nmean_dist = {}
    for r in records:
        nmean_dist[r["n_meaningful"]] = nmean_dist.get(r["n_meaningful"], 0) + 1

    cnt, esum = defaultdict(int), defaultdict(float)
    for r in records:
        for f in r["node_top_features"]:
            key = (f["layer"], f["fidx"])
            cnt[key] += 1
            esum[key] += f["edge"]
    co = [{"layer": l, "fidx": fi, "count": cnt[(l, fi)],
           "frac": (cnt[(l, fi)] / n if n else 0.0),
           "mean_edge": round(esum[(l, fi)] / cnt[(l, fi)], 6)} for (l, fi) in cnt]
    co.sort(key=lambda d: (-d["count"], -abs(d["mean_edge"])))

    return {
        "config": config,
        "n_records": n, "n_skipped": result.get("n_skipped", 0),
        "n_sampled": result.get("n_sampled"),
        "rank_histogram": {"counts": hist, "pct": hist_pct},
        "loose_multipos": {"n_ge2_positions": loose_ge2,
                           "frac": (loose_ge2 / n if n else 0.0),
                           "span_distribution": span_dist, "flagged": flagged},
        "meaningful_crosstoken": {"n_meaningful": meaningful_ge2,
                                  "frac": (meaningful_ge2 / n if n else 0.0),
                                  "n_meaningful_distribution": nmean_dist,
                                  "vs_loose": {"loose_ge2": loose_ge2,
                                               "meaningful_ge2": meaningful_ge2,
                                               "gap": loose_ge2 - meaningful_ge2}},
        "co_influencers": co[:top_k],
    }


def format_report(report):
    """Human-readable multi-line summary string of a report dict."""
    L = []
    cfg = report.get("config", {})
    L.append(f"feature {cfg.get('target_feature')}  target {cfg.get('target')!r}  "
             f"n_records={report['n_records']}  skipped={report['n_skipped']}")
    L.append("\nRank among direct input features:")
    h, p = report["rank_histogram"]["counts"], report["rank_histogram"]["pct"]
    for b in _RANK_BUCKETS:
        L.append(f"  {b:>6}: {h[b]:>4}  ({p[b]*100:5.1f}%)")
    lm = report["loose_multipos"]
    L.append(f"\nLoose multi-position (fired >=2 positions): {lm['n_ge2_positions']} "
             f"({lm['frac']*100:.1f}%)  flagged(span)>= : {len(lm['flagged'])}")
    mc = report["meaningful_crosstoken"]
    L.append(f"Meaningful across tokens (strict): {mc['n_meaningful']} "
             f"({mc['frac']*100:.1f}%)   gap vs loose: {mc['vs_loose']['gap']}")
    L.append("\nCommon co-influencers (layer, fidx | count | mean_edge):")
    for c in report["co_influencers"]:
        L.append(f"  L{c['layer']} F{c['fidx']:<6}  x{c['count']:<4}  "
                 f"mean_edge={c['mean_edge']:+.4f}")
    return "\n".join(L)


def save_report(report, records, out_dir, *, layer, fidx, subset_slug):
    """Write report JSON + a flat per-record CSV. Returns {'json','csv'} paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"report_L{layer}F{fidx}_{subset_slug}"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(_json.dumps(report, indent=2))
    csv_path = out_dir / f"{stem}.csv"
    cols = ["id", "ds_idx", "name", "t_key", "prompt", "target_token", "rank",
            "bucket", "span", "n_positions", "n_meaningful", "is_meaningful"]
    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(cols)
        for r in records:
            w.writerow([r.get(c) for c in cols])
    return {"json": str(json_path), "csv": str(csv_path)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "build_report or format_report or save_report" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): report assembly (build_report, format_report, save_report)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Cache — `edge_cache_key` + `load_or_build_edges`

The expensive `attribute()` output is cached so swapping the feature/thresholds needs no recompute. Tested with a stub builder (a counter), no model needed.

**Files:**
- Modify: `clts/writefeatures.py`
- Test: `tests/test_writefeatures.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_writefeatures.py

def test_edge_cache_key_stable_and_distinct():
    p = {"max_n_logits": 10}
    k1 = wf.edge_cache_key(" Gage was born on", " August", p)
    k2 = wf.edge_cache_key(" Gage was born on", " August", p)
    k3 = wf.edge_cache_key(" Gage was born on", " July", p)
    assert k1 == k2 and k1 != k3 and len(k1) == 16


def test_load_or_build_edges_caches(tmp_path):
    calls = {"n": 0}
    def build():
        calls["n"] += 1
        return [{"layer": 3, "pos": 7, "fidx": 4768, "edge": 0.5}]
    a = wf.load_or_build_edges(tmp_path, "abc123", build)
    b = wf.load_or_build_edges(tmp_path, "abc123", build)   # cache hit
    assert a == b and calls["n"] == 1                       # build ran exactly once
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "cache_key or load_or_build" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'edge_cache_key'`.

- [ ] **Step 3: Add the functions**

```python
# add to clts/writefeatures.py
import hashlib as _hashlib

DEFAULT_BUILD_PARAMS = {"max_n_logits": 10, "desired_logit_prob": 0.95,
                        "max_feature_nodes": 4096, "batch_size": 256}


def edge_cache_key(prompt, target_token, build_params):
    """Stable 16-char key for the per-(prompt, target) edge list."""
    blob = _json.dumps({"prompt": prompt, "target": target_token,
                        "params": build_params}, sort_keys=True)
    return _hashlib.sha1(blob.encode()).hexdigest()[:16]


def load_or_build_edges(cache_dir, key, build_fn):
    """Return cached edges for `key`, else call build_fn(), cache, and return.
    build_fn is only invoked on a cache miss."""
    import torch
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    edges = build_fn()
    torch.save(edges, path)
    return edges
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "cache_key or load_or_build" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): edge-list cache (edge_cache_key, load_or_build_edges)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Model glue + sweep — `attribute_fast`, `node_input_edges`, people selectors, `run_hypothesis` (+ artifact-gated integration test)

Wires the pieces together. The new functions take `model`/`graph`/`sampler`/`ct` as arguments (so they're importable/testable). One integration test exercises the real model end-to-end, gated on artifacts like `tests/test_attribution.py`.

**Files:**
- Modify: `clts/writefeatures.py`
- Test: `tests/test_writefeatures.py`

- [ ] **Step 1: Write the failing integration test**

```python
# append to tests/test_writefeatures.py
import os

_MODEL_DIR = "model/grid-L4-H6"
_CLT_DIR = "clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final"
_DATA_DIR = "data/bioS_N-Bd_final_grid"
_HAS_ARTIFACTS = os.path.isdir(_MODEL_DIR) and os.path.isdir(_CLT_DIR) and os.path.isdir(_DATA_DIR)


@pytest.mark.skipif(not _HAS_ARTIFACTS, reason="model/CLT/data artifacts not present")
def test_run_hypothesis_end_to_end(tmp_path):
    from pathlib import Path
    from clts.export_tokenizer import ensure_hf_tokenizer
    from clts.load_replacement_model import load_replacement_model
    from util.bio_sampler import BioSampler
    from util.condensed_tokenizer import CondensedTokenizer

    model = load_replacement_model(Path(_MODEL_DIR), Path(_CLT_DIR),
                                   ensure_hf_tokenizer(Path(_DATA_DIR)), "grid-L4-H6",
                                   device="cpu")
    sampler = BioSampler(Path(_DATA_DIR) / "people.json", fields=("birthday",))
    ct = CondensedTokenizer.from_remap_path(Path(_DATA_DIR) / "old_to_new.json")

    people = wf.people_in_month(sampler, ct, "August")[:1]      # 1 person
    result = wf.run_hypothesis(model, sampler, ct, people, target_feature=(3, 4768),
                               target="month", templates=[0, 1],  # 2 templates
                               cache_dir=tmp_path, n_cap=1, seed=0)
    assert result["n_sampled"] == 1
    assert len(result["records"]) >= 1
    r = result["records"][0]
    assert r["target_token"] == " August"
    assert (r["rank"] is None) or isinstance(r["rank"], int)
    assert "is_meaningful" in r and isinstance(r["node_top_features"], list)

    rep = wf.build_report(result, top_k=10, pos_span_flag=3, multi_tok_top_k=5,
                          config={"target_feature": [3, 4768], "target": "month"})
    assert rep["n_records"] == len(result["records"])
```

- [ ] **Step 2: Run the integration test to verify it fails**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py::test_run_hypothesis_end_to_end -v`
Expected: FAIL — `AttributeError: ... has no attribute 'people_in_month'` (functions not added yet). If artifacts are absent it will SKIP; in that case verify manually in Task 11 instead.

- [ ] **Step 3: Add the glue + sweep functions**

```python
# add to clts/writefeatures.py

def attribute_fast(model, prompt, target_token, *, max_n_logits=10,
                   desired_logit_prob=0.95, max_feature_nodes=4096, batch_size=256):
    """Run circuit-tracer attribution toward `target_token` on an ALREADY-LOADED
    model (no disk reload, no viewer files). Returns a Graph."""
    from circuit_tracer import attribute
    return attribute(prompt=prompt, model=model, attribution_targets=[target_token],
                     max_n_logits=max_n_logits, desired_logit_prob=desired_logit_prob,
                     batch_size=batch_size, max_feature_nodes=max_feature_nodes,
                     offload=None, verbose=False)


def node_input_edges(graph, target_token, tokenizer):
    """Raw per-node direct edges into the `target_token` logit node. Returns the
    edge list, or None if that token is not one of the graph's logit nodes."""
    ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(ids) != 1:
        return None
    n_features = len(graph.selected_features)
    n_tokens = len(graph.input_tokens)
    n_error = n_tokens * graph.cfg.n_layers
    logit_ids = graph.logit_token_ids.tolist()
    row, _k = find_logit_row(logit_ids, ids[0], n_features=n_features,
                             n_error=n_error, n_tokens=n_tokens)
    if row is None:
        return None
    return incoming_feature_edges(graph.adjacency_matrix.cpu(),
                                  graph.selected_features, graph.active_features, row)


def _name_in_vocab(ct, name):
    try:
        ct.encode(name)
        return True
    except KeyError:
        return False


def people_in_month(sampler, ct, month, in_vocab_only=True):
    """All (dataset_idx, person) born in `month`, in dataset order, in-vocab only."""
    out = []
    for ds_idx, p in enumerate(sampler.people):
        if p["birthmonth"] != month:
            continue
        if in_vocab_only and not _name_in_vocab(ct, f"{p['first_name']} {p['last_name']}"):
            continue
        out.append((ds_idx, p))
    return out


def people_by_ids(sampler, ids):
    wanted = set(ids)
    return [(i, p) for i, p in enumerate(sampler.people) if p["id"] in wanted]


def people_by_idx(sampler, idxs):
    return [(i, sampler.people[i]) for i in idxs]


def sample_in_month(sampler, ct, month, n, seed=0):
    return sample_people(people_in_month(sampler, ct, month), n, seed)


def _absent_record(ds_idx, person, t_key, prompt, target_token):
    return {"ds_idx": ds_idx, "id": person["id"],
            "name": f"{person['first_name']} {person['last_name']}", "t_key": t_key,
            "prompt": prompt, "target_token": target_token, "rank": None,
            "bucket": "absent", "span": 0, "positions": [], "n_positions": 0,
            "n_meaningful": 0, "is_meaningful": False, "node_top_features": []}


def run_hypothesis(model, sampler, ct, people, target_feature, target, templates, *,
                   cache_dir, n_cap=20, seed=0, top_k=10, multi_tok_top_k=5,
                   pos_span_flag=3, rank_by_abs=False, build_params=None, progress=print):
    """For each (sampled person x template): build/lookup the cached edge list and
    compute both metrics. Returns {records, n_skipped, n_sampled}."""
    build_params = dict(build_params or DEFAULT_BUILD_PARAMS)
    all_t = birthday_templates()
    tmpl_list = resolve_templates(templates, all_t)
    sampled = sample_people(list(people), n_cap, seed)
    records, n_skipped = [], 0

    for ds_idx, person in sampled:
        progress(f"  person id={person['id']} {person['first_name']} {person['last_name']}")
        for t_key, t_val in tmpl_list:
            try:
                prompt = template_prompt(person, t_val, sampler)
                target_token = resolve_target(target, person)
                if target_token is None:
                    n_skipped += 1
                    continue
                ids = model.tokenizer.encode(target_token, add_special_tokens=False)
                if len(ids) != 1:
                    n_skipped += 1
                    continue
                key = edge_cache_key(prompt, target_token, build_params)

                def _build():
                    g = attribute_fast(model, prompt, target_token, **build_params)
                    return node_input_edges(g, target_token, model.tokenizer)

                edges = load_or_build_edges(cache_dir, key, _build)
                if edges is None:
                    records.append(_absent_record(ds_idx, person, t_key, prompt, target_token))
                    continue
                agg = aggregate_by_feature(edges)
                fr = feature_rank(agg, target_feature, rank_by_abs=rank_by_abs)
                mt = feature_multitoken(edges, target_feature,
                                        multi_tok_top_k=multi_tok_top_k, rank_by_abs=rank_by_abs)
                records.append({
                    "ds_idx": ds_idx, "id": person["id"],
                    "name": f"{person['first_name']} {person['last_name']}",
                    "t_key": t_key, "prompt": prompt, "target_token": target_token,
                    "rank": fr["rank"], "bucket": rank_bucket(fr["rank"]),
                    "span": fr["span"], "positions": fr["positions"],
                    "n_positions": mt["n_positions"], "n_meaningful": mt["n_meaningful"],
                    "is_meaningful": mt["is_meaningful"],
                    "node_top_features": top_node_features(agg, top_k, rank_by_abs),
                })
            except Exception as exc:   # one bad (person, template) must not kill the sweep
                n_skipped += 1
                progress(f"    skip (t={t_key}): {type(exc).__name__}: {exc}")
    return {"records": records, "n_skipped": n_skipped, "n_sampled": len(sampled)}
```

- [ ] **Step 4: Run the full test file**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -v`
Expected: all unit tests PASS; `test_run_hypothesis_end_to_end` PASSES (or SKIPS if artifacts are absent). The end-to-end run prints two per-person progress lines and completes.

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): model glue + run_hypothesis sweep with integration test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Build the notebook `clts/writeFeatures.ipynb`

A thin wrapper. Created as a minimal valid `.ipynb` via Write, then cells are inserted with NotebookEdit. The kernel should be the `clts/.venv-ct` interpreter (set in Jupyter after opening).

**Files:**
- Create: `clts/writeFeatures.ipynb`

- [ ] **Step 1: Write the minimal notebook skeleton**

Create `clts/writeFeatures.ipynb` with this exact content (one placeholder cell; real cells are added next):

```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# writeFeatures — placeholder"]}
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python (.venv-ct)", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Insert the title markdown cell (replace the placeholder)**

Use NotebookEdit (`notebook_path` absolute = `/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4/clts/writeFeatures.ipynb`, `edit_mode=replace`, `cell_number=0`, `cell_type=markdown`):

```
# writeFeatures — feature-in-node rank tester

Given a **people subset** and a **target feature** `(layer, fidx)`, measure where that
feature ranks among the **direct input features** of an output-token node (e.g. the
` August` logit), across all 46 bio templates and all people. Reports a rank histogram,
a strict "meaningful across tokens" metric, and the common co-influencer features.

**To use:** run the loading cells once, then edit only the **MODEL CONFIG** cell (rarely)
and the **EDIT THIS CELL** cell (people + feature), and run the final cell.

All logic lives in the tested module `clts/writefeatures.py`; this notebook is a thin wrapper.
Kernel: the `clts/.venv-ct` interpreter.
```

- [ ] **Step 3: Insert the MODEL CONFIG cell**

NotebookEdit (`edit_mode=insert`, `cell_id` = the title cell's id, `cell_type=code`):

```python
# ===================== MODEL CONFIG (edit to swap models) =====================
from pathlib import Path
import sys

REPO = Path.cwd()
if REPO.name == "clts":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

MODEL_DIR = REPO / "model/grid-L4-H6"
CLT_DIR   = REPO / "clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final"
DATA_DIR  = REPO / "data/bioS_N-Bd_final_grid"
SCAN_NAME = "grid-L4-H6"   # also namespaces the cache/report dir
DEVICE    = "cpu"
# ==============================================================================
print("model config set:", SCAN_NAME)
```

- [ ] **Step 4: Insert the loading cell (imports, model, helpers, partials)**

NotebookEdit (`edit_mode=insert`, after the MODEL CONFIG cell, `cell_type=code`):

```python
import importlib
import torch

import clts.writefeatures as wf
importlib.reload(wf)   # pick up edits to writefeatures.py without a kernel restart

from clts.export_tokenizer import ensure_hf_tokenizer
from clts.load_replacement_model import load_replacement_model
from clts.storage import storage_root
from util.bio_sampler import BioSampler
from util.condensed_tokenizer import CondensedTokenizer

ct = CondensedTokenizer.from_remap_path(DATA_DIR / "old_to_new.json")
sampler = BioSampler(DATA_DIR / "people.json", fields=("birthday",))

if globals().get("model") is None:
    model = load_replacement_model(
        MODEL_DIR, CLT_DIR, ensure_hf_tokenizer(DATA_DIR), SCAN_NAME, device=DEVICE)
    print(f"model loaded — n_layers={model.cfg.n_layers}, d_vocab={model.cfg.d_vocab}")
else:
    print("model already loaded (set `model = None` and re-run to reload)")

CACHE_DIR = storage_root() / "clt_feature_explorer" / SCAN_NAME / "hyptest"

# Tiny partials so the EDIT cell reads cleanly.
people_in_month = lambda month: wf.people_in_month(sampler, ct, month)
people_by_ids   = lambda ids:   wf.people_by_ids(sampler, ids)
people_by_idx   = lambda idxs:  wf.people_by_idx(sampler, idxs)
sample_in_month = lambda month, n, seed=0: wf.sample_in_month(sampler, ct, month, n, seed)
print("helpers ready · cache dir:", CACHE_DIR)
```

- [ ] **Step 5: Insert the EDIT cell**

NotebookEdit (`edit_mode=insert`, after the loading cell, `cell_type=code`):

```python
# ===================== EDIT THIS CELL =====================
PEOPLE          = people_in_month("August")  # or people_by_ids([...]) / people_by_idx([...]) / sample_in_month("August", 20)
TARGET_FEATURE  = (3, 4768)                  # (layer, feature_idx) to locate
TARGET          = "month"                    # "month" = each person's own birth-month token; or pin e.g. " August"
TEMPLATES       = "all"                      # "all" 46 templates, or [0, 5, 12], or ["{name} popped out on {birthday}."]
N_PEOPLE_CAP    = 20                         # cap; capping RANDOM-SAMPLES the pool (None = use all)
SEED            = 0                           # RNG seed for the random people sample
TOP_K           = 10                         # co-influencer table depth (does NOT change the rank histogram)
MULTI_TOK_TOP_K = 5                           # node-level top-K for the "meaningful across tokens" metric (strict)
POS_SPAN_FLAG   = 3                          # loose flag: feature fired at >=2 positions spanning >= this many tokens
RANK_BY_ABS     = False                      # False = signed (promoters first); True = |edge|
SUBSET_LABEL    = "august"                   # short label for the saved report filename
# ==========================================================
print(f"{len(PEOPLE)} people in pool · target feature {TARGET_FEATURE} · target {TARGET!r}")
```

- [ ] **Step 6: Insert the RUN + REPORT cell**

NotebookEdit (`edit_mode=insert`, after the EDIT cell, `cell_type=code`):

```python
result = wf.run_hypothesis(
    model, sampler, ct, PEOPLE, target_feature=TARGET_FEATURE, target=TARGET,
    templates=TEMPLATES, cache_dir=CACHE_DIR, n_cap=N_PEOPLE_CAP, seed=SEED,
    top_k=TOP_K, multi_tok_top_k=MULTI_TOK_TOP_K, pos_span_flag=POS_SPAN_FLAG,
    rank_by_abs=RANK_BY_ABS,
)

config = {"target_feature": list(TARGET_FEATURE), "target": TARGET,
          "templates": TEMPLATES, "n_cap": N_PEOPLE_CAP, "seed": SEED,
          "top_k": TOP_K, "multi_tok_top_k": MULTI_TOK_TOP_K,
          "pos_span_flag": POS_SPAN_FLAG, "rank_by_abs": RANK_BY_ABS,
          "subset_label": SUBSET_LABEL, "scan": SCAN_NAME}
report = wf.build_report(result, top_k=TOP_K, pos_span_flag=POS_SPAN_FLAG,
                         multi_tok_top_k=MULTI_TOK_TOP_K, config=config)
print(wf.format_report(report))

slug = f"{SUBSET_LABEL}-n{N_PEOPLE_CAP}-s{SEED}"
paths = wf.save_report(report, result["records"], CACHE_DIR,
                       layer=TARGET_FEATURE[0], fidx=TARGET_FEATURE[1], subset_slug=slug)
print("\nsaved:", paths["json"])
print("saved:", paths["csv"])
```

- [ ] **Step 7: Commit**

```bash
git add clts/writeFeatures.ipynb
git commit -m "feat(writefeatures): thin notebook wrapper (model config + EDIT + run/report)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: End-to-end verification (real model) + cache re-run check

Verifies the assembled tool on the canonical case and confirms the cache makes a feature swap free. Done with a throwaway driver script because the venv can't execute notebooks headless; the script calls the exact same module entry points the notebook uses.

**Files:**
- Create (temporary): `scripts/_run_writefeatures_e2e.py`

- [ ] **Step 1: Write the driver**

```python
# scripts/_run_writefeatures_e2e.py  (TEMPORARY)
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import clts.writefeatures as wf
from clts.export_tokenizer import ensure_hf_tokenizer
from clts.load_replacement_model import load_replacement_model
from clts.storage import storage_root
from util.bio_sampler import BioSampler
from util.condensed_tokenizer import CondensedTokenizer

DATA_DIR = REPO / "data/bioS_N-Bd_final_grid"
model = load_replacement_model(REPO / "model/grid-L4-H6",
    REPO / "clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final",
    ensure_hf_tokenizer(DATA_DIR), "grid-L4-H6", device="cpu")
sampler = BioSampler(DATA_DIR / "people.json", fields=("birthday",))
ct = CondensedTokenizer.from_remap_path(DATA_DIR / "old_to_new.json")
cache = storage_root() / "clt_feature_explorer" / "grid-L4-H6" / "hyptest"

people = wf.people_in_month(sampler, ct, "August")
t0 = time.time()
result = wf.run_hypothesis(model, sampler, ct, people, target_feature=(3, 4768),
                           target="month", templates="all", cache_dir=cache,
                           n_cap=3, seed=0)   # small N for the smoke run
print(f"first run: {time.time()-t0:.1f}s  records={len(result['records'])} skipped={result['n_skipped']}")
n_expected = result["n_sampled"] * len(wf.birthday_templates())
hist_total = len(result["records"]) + result["n_skipped"]
assert hist_total == n_expected, (hist_total, n_expected)

rep = wf.build_report(result, top_k=10, pos_span_flag=3, multi_tok_top_k=5,
                      config={"target_feature": [3, 4768], "target": "month"})
print(wf.format_report(rep))

# cache check: swap the feature, time it -> should be near-instant (no attribute calls)
t1 = time.time()
result2 = wf.run_hypothesis(model, sampler, ct, people, target_feature=(2, 11),
                            target="month", templates="all", cache_dir=cache, n_cap=3, seed=0)
dt = time.time() - t1
print(f"second run (feature swap): {dt:.2f}s  (should be << first run; cache hit)")
assert dt < (time.time() - t0) / 2, "cache did not speed up the feature swap"
print("OK")
```

- [ ] **Step 2: Run the driver**

Run: `clts/.venv-ct/bin/python scripts/_run_writefeatures_e2e.py`
Expected:
- First run completes; prints `records=` and `skipped=`, with `records + skipped == n_sampled * 46`.
- `format_report` prints the rank histogram, loose vs. meaningful cross-token lines, and a co-influencer list.
- Second run (feature swapped to `(2, 11)`) is far faster (cache hit) and prints `OK`.

If `records + skipped != n_sampled * 46`, or the second run isn't substantially faster, STOP and debug before proceeding.

- [ ] **Step 3: Confirm the report files exist**

Run: `ls -1 clt_storage/clt_feature_explorer/grid-L4-H6/hyptest/*.pt | wc -l && ls clt_storage/clt_feature_explorer/grid-L4-H6/hyptest/report_* 2>/dev/null`
Expected: a non-zero count of `.pt` cache files (≈ `n_sampled * 46`). (The driver doesn't call `save_report`; report files are produced when the notebook's run cell executes — that's exercised manually in Step 5.)

- [ ] **Step 4: Delete the driver (not committed)**

Run: `rm scripts/_run_writefeatures_e2e.py`
Expected: file gone.

- [ ] **Step 5: Manual notebook check (human/in-IDE)**

Open `clts/writeFeatures.ipynb`, select the `.venv-ct` kernel, Run All. Confirm: the model loads, the EDIT cell prints the pool size, and the final cell prints `format_report` output and two `saved:` lines pointing at `report_L3F4768_august-n20-s0.json/.csv`. Then edit `TARGET_FEATURE` to another feature and re-run only the last cell — it should return quickly (cache hit) with a new report.

- [ ] **Step 6: Full test-suite sanity + final commit**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -v`
Expected: all PASS (integration test PASS or SKIP).

```bash
git add -A
git commit -m "chore(writefeatures): verified end-to-end (August / L3 F4768); cache speeds feature swap

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Direct-edge metric → Tasks 3–4 (`incoming_feature_edges`, `feature_rank`). ✓
- Sum-across-positions + span flag → Task 4 (`aggregate_by_feature`, `feature_rank.span`). ✓
- All 46 templates → Task 6 (`birthday_templates`, `resolve_templates`, `template_prompt`). ✓
- Signed default + `RANK_BY_ABS` → Tasks 4–5 (rank_by_abs threaded). ✓
- Strict meaningful-across-tokens metric → Task 5 (`feature_multitoken`). ✓
- Randomized people (seeded) → Task 2 (`sample_people`), Task 9 (`sample_in_month`). ✓
- Inline + saved report (JSON + CSV) → Task 7 (`format_report`, `save_report`). ✓
- Caching (feature-independent edge list) → Task 8 + Task 9 wiring; cache speedup asserted in Task 11. ✓
- Visible MODEL CONFIG cell + thin notebook → Task 10. ✓
- Storage path `clt_feature_explorer/<scan>/hyptest/` → Tasks 9–11. ✓
- First-step de-risk of in-memory `attribute()` + indexing → Task 1. ✓

**Placeholder scan:** No "TBD"/"implement later"; every code step has complete, runnable code.

**Type consistency:** `record` keys are identical across `run_hypothesis` (Task 9), `_absent_record` (Task 9), `build_report`/`save_report` (Task 7), and the test fixtures (Task 7). `rank_by_abs` and `multi_tok_top_k` names are consistent everywhere. `find_logit_row` keyword args match between Task 3 and Task 9. `node_input_edges` returns the same edge-dict shape that `aggregate_by_feature`/`feature_multitoken` consume.

> **Notebook caveat for the executor:** NotebookEdit assigns cell ids automatically; when inserting, pass the *current last cell's* `cell_id` so each new cell lands at the end in order. After Task 10, open the notebook once to confirm cell order (title → MODEL CONFIG → loading → EDIT → run/report) before the Task 11 manual check.

---

# Addendum — Unified co-influencers + token-role labeling (Tasks 12–16)

> This layer is **strictly additive** on top of the working Tasks 1–11 tool. It surfaces **error nodes** (and token nodes) that feed the output logit, labeled by the **token role** they sit on (e.g. `err@last_name@L2`), in a **unified** co-influencer ranking. The headline feature-rank metric (outputs 1–3) must stay **byte-identical** — Task 15 adds a regression test that enforces this.
>
> **Verified mechanics (from the investigation workflow):** error block is **layer-major** → `layer, pos = divmod(j - n_features, n_tokens)`; token block → `pos = j - (n_features + n_tokens*n_layers)`; the graph prepends a **BOS** at position 0 (so role arrays are length `n_tokens` with `roles[0]="BOS"`); `model.tokenizer` is a `GPT2TokenizerFast` supporting `return_offsets_mapping=True`. Empirical anchor on ` Gage Wyatt Clay was born on → August`: edge mass into the logit = **feature 93.9% / error 6.0% / token 0.03%**, top error node `err@last_name@L2` on ` Clay`.

## Task 12: Token-role assignment — `assign_roles` + `token_roles`

`assign_roles` is pure (takes precomputed char offsets) and fully unit-tested; `token_roles` is the thin tokenizer-calling wrapper (tested in Task 14's integration test).

**Files:**
- Modify: `clts/writefeatures.py`
- Test: `tests/test_writefeatures.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_writefeatures.py

def test_assign_roles_basic():
    person = {"first_name": "Gage", "middle_name": "Wyatt", "last_name": "Clay"}
    text = " Gage Wyatt Clay was born on"
    # offsets for tokens [' G','age',' Wyatt',' Clay',' was',' born',' on']
    offsets = [(0, 2), (2, 5), (5, 11), (11, 16), (16, 20), (20, 25), (25, 28)]
    roles = wf.assign_roles(offsets, text, person, {"born", "birth", "day", "date"})
    # single-token last name 'Clay' is still last_name[final] (always exactly one)
    assert roles == ["first_name", "first_name", "middle_name", "last_name[final]",
                     "template:other", "template:born", "template:other"]
    assert roles.count("last_name[final]") == 1


def test_assign_roles_marks_last_name_final_when_multitoken():
    person = {"first_name": "Gianna", "middle_name": "Adeline", "last_name": "Rawlings"}
    text = " Gianna Adeline Rawlings was born on"
    # 'Gianna'->' G'|'ian'|'na'  'Adeline'->' Adeline'  'Rawlings'->' Raw'|'lings'
    offsets = [(0, 2), (2, 5), (5, 7), (7, 15), (15, 19), (19, 24),
               (24, 28), (28, 33), (33, 36)]
    roles = wf.assign_roles(offsets, text, person, {"born"})
    assert roles[4] == "last_name"          # ' Raw'
    assert roles[5] == "last_name[final]"   # 'lings' (final subword of a multi-token last name)
    assert roles[7] == "template:born"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "assign_roles" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'assign_roles'`.

- [ ] **Step 3: Add the functions**

```python
# add to clts/writefeatures.py
import re as _re


def _name_field_spans(text, person):
    """Char (start, end) spans of first/middle/last name in `text`, found in order
    (the prompt always begins ' {first} {middle} {last} ...')."""
    spans, cursor = {}, 0
    for field in ("first_name", "middle_name", "last_name"):
        val = str(person[field])
        i = text.index(val, cursor)
        spans[field] = (i, i + len(val))
        cursor = i + len(val)
    return spans


def _lexical_role(text, a, b, spans, template_word_labels):
    for field, role in (("first_name", "first_name"), ("middle_name", "middle_name"),
                        ("last_name", "last_name")):
        s, e = spans[field]
        if a < e and b > s:          # char overlap (handles leading-space tokens)
            return role
    word = _re.sub(r"[^a-z]", "", text[a:b].lower())
    if word in template_word_labels:
        return f"template:{word}"
    return "template:other"


def assign_roles(offsets, text, person, template_word_labels):
    """Per prompt-token lexical role (NO BOS). The final 'last_name' token is ALWAYS
    relabeled `last_name[final]` (exactly one per graph, whether the last name is one
    token or several), so the bucket is stable regardless of tokenization; earlier
    subwords of a multi-token last name stay `last_name`."""
    spans = _name_field_spans(text, person)
    roles = [_lexical_role(text, a, b, spans, template_word_labels) for (a, b) in offsets]
    last_idxs = [i for i, r in enumerate(roles) if r == "last_name"]
    if last_idxs:
        roles[last_idxs[-1]] = "last_name[final]"
    return roles


def token_roles(prompt, person, tokenizer, *, template_word_labels):
    """Role per GRAPH position. The graph prepends BOS at position 0, so this returns
    ['BOS', <role per prompt token>...] (length == graph n_tokens). The final position
    also carries a '(recall)' suffix on its lexical role."""
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    roles = ["BOS"] + assign_roles(enc["offset_mapping"], prompt, person, template_word_labels)
    roles[-1] = roles[-1] + "(recall)"
    return roles
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "assign_roles" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): token-role assignment (assign_roles, token_roles)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 13: Error/token decode + unified labeling — `decode_error_nodes`, `decode_token_nodes`, `label_nodes`, `unified_top_labels`

Pure functions over a synthetic adjacency + role list. This proves the layer-major error decode and the labeled aggregation without the model.

**Files:**
- Modify: `clts/writefeatures.py`
- Test: `tests/test_writefeatures.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_writefeatures.py

def test_decode_error_nodes_layer_major():
    # n_features=3, n_tokens=2, n_layers=2 -> error block cols [3,7), token cols [7,9)
    N = 10
    A = torch.zeros(N, N)
    logit_row = 9
    # error col for (layer=1, pos=0) is at error-block index 1*2+0=2 -> matrix col 3+2=5
    A[logit_row, 5] = 0.7
    errs = wf.decode_error_nodes(A, logit_row, n_features=3, n_tokens=2, n_layers=2)
    assert {"layer": 1, "pos": 0, "edge": 0.7} in errs
    assert len(errs) == 4                      # n_tokens*n_layers


def test_decode_token_nodes():
    N = 10
    A = torch.zeros(N, N)
    logit_row = 9
    A[logit_row, 7] = 0.2   # token block starts at 3 + 2*2 = 7 ; pos 0
    A[logit_row, 8] = 0.9   # pos 1
    toks = wf.decode_token_nodes(A, logit_row, n_features=3, n_tokens=2, n_layers=2)
    assert toks == [{"pos": 0, "edge": 0.2}, {"pos": 1, "edge": 0.9}]


def test_label_nodes_and_unified_top():
    node_all = {
        "features": [{"layer": 3, "pos": 7, "fidx": 4768, "edge": 0.6},
                     {"layer": 3, "pos": 1, "fidx": 4768, "edge": 0.2}],   # sums to 0.8
        "errors":   [{"layer": 2, "pos": 4, "edge": 0.5},
                     {"layer": 0, "pos": 5, "edge": 0.05}],
        "tokens":   [{"pos": 4, "edge": 0.01}],
    }
    # role per position (index = graph pos)
    roles = ["BOS", "first_name", "first_name", "middle_name", "last_name",
             "template:born", "template:other", "template:other(recall)"]
    rows = wf.label_nodes(node_all, roles, include_tokens=True)
    labels = {r["label"]: r["edge"] for r in rows}
    assert labels["L3 F4768"] == 0.8
    assert labels["err@last_name@L2"] == 0.5
    assert labels["err@template:born@L0"] == 0.05
    assert labels["tok@last_name"] == 0.01
    top = wf.unified_top_labels(node_all, roles, top_k=2, rank_by_abs=False)
    assert [t["label"] for t in top] == ["L3 F4768", "err@last_name@L2"]
    # include_tokens=False drops tok@* rows
    no_tok = wf.label_nodes(node_all, roles, include_tokens=False)
    assert all(not r["label"].startswith("tok@") for r in no_tok)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "decode_error or decode_token or label_nodes" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'decode_error_nodes'`.

- [ ] **Step 3: Add the functions**

```python
# add to clts/writefeatures.py

def decode_error_nodes(adjacency, logit_row, n_features, n_tokens, n_layers):
    """Edges into `logit_row` from each error node, decoded layer-major:
    error-block index j' -> (layer, pos) = divmod(j', n_tokens)."""
    e = adjacency[logit_row, n_features:n_features + n_tokens * n_layers]
    out = []
    for jp, w in enumerate(e.tolist()):
        layer, pos = divmod(jp, n_tokens)
        out.append({"layer": layer, "pos": pos, "edge": round(float(w), 6)})
    return out


def decode_token_nodes(adjacency, logit_row, n_features, n_tokens, n_layers):
    """Edges into `logit_row` from each token/embedding node (one per position)."""
    start = n_features + n_tokens * n_layers
    e = adjacency[logit_row, start:start + n_tokens]
    return [{"pos": pos, "edge": round(float(w), 6)} for pos, w in enumerate(e.tolist())]


def label_nodes(node_all, roles, *, include_tokens=True):
    """Unified labeled rows for ALL input nodes to the logit. Features keep
    'L{layer} F{fidx}' (summed across positions); error nodes -> 'err@{role}@L{layer}'
    (summed across positions sharing the role); token nodes -> 'tok@{role}'."""
    rows = []
    for (layer, fidx), slot in aggregate_by_feature(node_all["features"]).items():
        rows.append({"kind": "feature", "label": f"L{layer} F{fidx}", "edge": slot["edge"]})
    err = {}
    for nd in node_all["errors"]:
        label = f"err@{roles[nd['pos']]}@L{nd['layer']}"
        err[label] = round(err.get(label, 0.0) + nd["edge"], 6)
    for label, edge in err.items():
        rows.append({"kind": "error", "label": label, "edge": edge})
    if include_tokens:
        tok = {}
        for nd in node_all["tokens"]:
            label = f"tok@{roles[nd['pos']]}"
            tok[label] = round(tok.get(label, 0.0) + nd["edge"], 6)
        for label, edge in tok.items():
            rows.append({"kind": "token", "label": label, "edge": edge})
    return rows


def unified_top_labels(node_all, roles, *, top_k, rank_by_abs=False, include_tokens=True):
    """Top-K labeled rows across ALL node kinds, ranked by edge (signed or |edge|)."""
    rows = label_nodes(node_all, roles, include_tokens=include_tokens)
    keyfn = (lambda r: abs(r["edge"])) if rank_by_abs else (lambda r: r["edge"])
    rows.sort(key=keyfn, reverse=True)
    return rows[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "decode_error or decode_token or label_nodes" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): error/token decode + unified node labeling

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 14: Promote `node_input_edges` → `node_input_all` and wire the unified view into `run_hypothesis`

Replaces the cached feature-only edge list with the full `{features, errors, tokens}` decomposition (still feature-independent). Feature metrics now read `node_all["features"]` (unchanged behavior); each record gains `unified_top`.

**Files:**
- Modify: `clts/writefeatures.py` (replace `node_input_edges` and `run_hypothesis` from Task 9; update `_absent_record`)
- Test: `tests/test_writefeatures.py` (extend the integration test)

- [ ] **Step 1: Update the integration test to assert the unified fields**

Replace `test_run_hypothesis_end_to_end` (from Task 9) with this version:

```python
@pytest.mark.skipif(not _HAS_ARTIFACTS, reason="model/CLT/data artifacts not present")
def test_run_hypothesis_end_to_end(tmp_path):
    from pathlib import Path
    from clts.export_tokenizer import ensure_hf_tokenizer
    from clts.load_replacement_model import load_replacement_model
    from util.bio_sampler import BioSampler
    from util.condensed_tokenizer import CondensedTokenizer

    model = load_replacement_model(Path(_MODEL_DIR), Path(_CLT_DIR),
                                   ensure_hf_tokenizer(Path(_DATA_DIR)), "grid-L4-H6",
                                   device="cpu")
    sampler = BioSampler(Path(_DATA_DIR) / "people.json", fields=("birthday",))
    ct = CondensedTokenizer.from_remap_path(Path(_DATA_DIR) / "old_to_new.json")

    # token_roles on the canonical prompt (BOS + recall + 'born')
    person0 = wf.people_in_month(sampler, ct, "August")[0][1]
    roles = wf.token_roles(f" {person0['first_name']} {person0['middle_name']} "
                           f"{person0['last_name']} was born on", person0, model.tokenizer,
                           template_word_labels={"born", "birth", "day", "date"})
    assert roles[0] == "BOS" and roles[-1].endswith("(recall)")

    people = wf.people_in_month(sampler, ct, "August")[:1]
    result = wf.run_hypothesis(model, sampler, ct, people, target_feature=(3, 4768),
                               target="month", templates=[0, 1], cache_dir=tmp_path,
                               n_cap=1, seed=0)
    r = result["records"][0]
    assert r["target_token"] == " August"
    assert (r["rank"] is None) or isinstance(r["rank"], int)
    assert isinstance(r["unified_top"], list) and r["unified_top"]
    # at least one error label appears among the unified rows for this prompt
    assert any(t["label"].startswith("err@") for t in r["unified_top"])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py::test_run_hypothesis_end_to_end -v`
Expected: FAIL (`node_input_all`/`unified_top` not present), or SKIP if artifacts absent.

- [ ] **Step 3: Replace `node_input_edges`, `_absent_record`, and `run_hypothesis`**

Delete the old `node_input_edges` (Task 9) and add `node_input_all`:

```python
# replace node_input_edges with:
def node_input_all(graph, target_token, tokenizer):
    """Full decomposition of the edges into the `target_token` logit node:
    {'features': [...], 'errors': [...], 'tokens': [...]}. None if the token is not a
    logit node in this graph. Cached (feature-independent)."""
    ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(ids) != 1:
        return None
    n_features = len(graph.selected_features)
    n_tokens = len(graph.input_tokens)
    n_layers = graph.cfg.n_layers
    n_error = n_tokens * n_layers
    row, _k = find_logit_row(graph.logit_token_ids.tolist(), ids[0],
                             n_features=n_features, n_error=n_error, n_tokens=n_tokens)
    if row is None:
        return None
    A = graph.adjacency_matrix.cpu()
    return {
        "features": incoming_feature_edges(A, graph.selected_features, graph.active_features, row),
        "errors": decode_error_nodes(A, row, n_features, n_tokens, n_layers),
        "tokens": decode_token_nodes(A, row, n_features, n_tokens, n_layers),
    }
```

Update `_absent_record` to carry `unified_top`:

```python
# in _absent_record(...) return dict, add this key:
            "unified_top": [],
```

Replace `run_hypothesis` with this version (changes: cache `node_all`; feature metrics read `node_all["features"]`; compute `roles` + `unified_top`; new kwargs `template_word_labels`, `include_token_nodes`):

```python
def run_hypothesis(model, sampler, ct, people, target_feature, target, templates, *,
                   cache_dir, n_cap=20, seed=0, top_k=10, multi_tok_top_k=5,
                   pos_span_flag=3, rank_by_abs=False,
                   template_word_labels=frozenset({"born", "birth", "day", "date"}),
                   include_token_nodes=True, build_params=None, progress=print):
    """Per (sampled person x template): build/lookup the cached node decomposition,
    compute the feature metrics (feature block only) AND the unified labeled top-K."""
    build_params = dict(build_params or DEFAULT_BUILD_PARAMS)
    key_params = {**build_params, "schema": "node_all_v1"}   # cache schema tag
    all_t = birthday_templates()
    tmpl_list = resolve_templates(templates, all_t)
    sampled = sample_people(list(people), n_cap, seed)
    records, n_skipped = [], 0

    for ds_idx, person in sampled:
        progress(f"  person id={person['id']} {person['first_name']} {person['last_name']}")
        for t_key, t_val in tmpl_list:
            try:
                prompt = template_prompt(person, t_val, sampler)
                target_token = resolve_target(target, person)
                if target_token is None:
                    n_skipped += 1
                    continue
                if len(model.tokenizer.encode(target_token, add_special_tokens=False)) != 1:
                    n_skipped += 1
                    continue
                key = edge_cache_key(prompt, target_token, key_params)

                def _build():
                    g = attribute_fast(model, prompt, target_token, **build_params)
                    return node_input_all(g, target_token, model.tokenizer)

                node_all = load_or_build_edges(cache_dir, key, _build)
                if node_all is None:
                    records.append(_absent_record(ds_idx, person, t_key, prompt, target_token))
                    continue
                feats = node_all["features"]
                agg = aggregate_by_feature(feats)
                fr = feature_rank(agg, target_feature, rank_by_abs=rank_by_abs)
                mt = feature_multitoken(feats, target_feature,
                                        multi_tok_top_k=multi_tok_top_k, rank_by_abs=rank_by_abs)
                roles = token_roles(prompt, person, model.tokenizer,
                                    template_word_labels=template_word_labels)
                unified = unified_top_labels(node_all, roles, top_k=top_k,
                                             rank_by_abs=rank_by_abs,
                                             include_tokens=include_token_nodes)
                records.append({
                    "ds_idx": ds_idx, "id": person["id"],
                    "name": f"{person['first_name']} {person['last_name']}",
                    "t_key": t_key, "prompt": prompt, "target_token": target_token,
                    "rank": fr["rank"], "bucket": rank_bucket(fr["rank"]),
                    "span": fr["span"], "positions": fr["positions"],
                    "n_positions": mt["n_positions"], "n_meaningful": mt["n_meaningful"],
                    "is_meaningful": mt["is_meaningful"],
                    "node_top_features": top_node_features(agg, top_k, rank_by_abs),
                    "unified_top": unified,
                })
            except Exception as exc:
                n_skipped += 1
                progress(f"    skip (t={t_key}): {type(exc).__name__}: {exc}")
    return {"records": records, "n_skipped": n_skipped, "n_sampled": len(sampled)}
```

> The cache object shape changed (list → dict), and the key now includes `"schema": "node_all_v1"`, so old `.pt` files are simply ignored (new keys). No manual cache wipe needed.

- [ ] **Step 4: Run the full suite**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -v`
Expected: all unit tests PASS; integration test PASS (or SKIP if artifacts absent).

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): node_input_all + unified labeled view in run_hypothesis

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 15: Unified co-influencers in `build_report` + the unskew guardrail

Replaces the feature-only `co_influencers` (Task 7) with the unified labeled aggregation, and adds the regression test proving outputs 1–3 are unchanged.

**Files:**
- Modify: `clts/writefeatures.py` (`build_report`, `format_report`)
- Test: `tests/test_writefeatures.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_writefeatures.py

def _rec_unified(rank, unified_top):
    r = _rec(rank, 1, 0, False, 1, [])     # _rec from Task 7
    r["unified_top"] = unified_top
    return r


def test_build_report_unified_co_influencers():
    records = [
        _rec_unified(1, [{"kind": "feature", "label": "L3 F4768", "edge": 0.6},
                         {"kind": "error", "label": "err@last_name@L2", "edge": 0.5}]),
        _rec_unified(2, [{"kind": "error", "label": "err@last_name@L2", "edge": 0.4},
                         {"kind": "feature", "label": "L3 F4768", "edge": 0.3}]),
    ]
    rep = wf.build_report({"records": records, "n_skipped": 0, "n_sampled": 1},
                          top_k=10, pos_span_flag=3, multi_tok_top_k=5,
                          config={"target_feature": [3, 4768]})
    labels = {c["label"]: c for c in rep["co_influencers"]}
    assert labels["err@last_name@L2"]["count"] == 2
    assert labels["err@last_name@L2"]["kind"] == "error"
    assert abs(labels["err@last_name@L2"]["mean_edge"] - 0.45) < 1e-9


def test_unskew_guardrail_outputs_1_to_3_unchanged():
    # Two record sets identical except for unified_top contents; outputs 1-3 must match.
    base = [_rec_unified(1, [{"kind": "feature", "label": "L3 F4768", "edge": 0.6}]),
            _rec_unified(3, [{"kind": "feature", "label": "L1 F5", "edge": 0.2}])]
    witherr = [dict(r, unified_top=r["unified_top"] +
                    [{"kind": "error", "label": "err@last_name@L2", "edge": 9.9}]) for r in base]
    cfg = {"target_feature": [3, 4768]}
    a = wf.build_report({"records": base, "n_skipped": 0, "n_sampled": 1},
                        top_k=10, pos_span_flag=3, multi_tok_top_k=5, config=cfg)
    b = wf.build_report({"records": witherr, "n_skipped": 0, "n_sampled": 1},
                        top_k=10, pos_span_flag=3, multi_tok_top_k=5, config=cfg)
    assert a["rank_histogram"] == b["rank_histogram"]
    assert a["loose_multipos"] == b["loose_multipos"]
    assert a["meaningful_crosstoken"] == b["meaningful_crosstoken"]
    # ...but the co-influencer view DID change (error label now present)
    assert a["co_influencers"] != b["co_influencers"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -k "unified_co_influencers or unskew" -v`
Expected: FAIL — the current `co_influencers` aggregates `node_top_features` (features only), so `err@...` labels are absent and `kind` is missing.

- [ ] **Step 3: Update `build_report` and `format_report`**

In `build_report`, replace the co-influencer block (the `cnt, esum = defaultdict(...)` loop over `r["node_top_features"]` and the `co = [...]` construction) with this unified version:

```python
    # Unified co-influencers: aggregate labeled rows across all records' unified_top.
    cnt, esum, kind_of = defaultdict(int), defaultdict(float), {}
    for r in records:
        for nd in r.get("unified_top", []):
            cnt[nd["label"]] += 1
            esum[nd["label"]] += nd["edge"]
            kind_of[nd["label"]] = nd["kind"]
    co = [{"label": lbl, "kind": kind_of[lbl], "count": cnt[lbl],
           "frac": (cnt[lbl] / n if n else 0.0),
           "mean_edge": round(esum[lbl] / cnt[lbl], 6)} for lbl in cnt]
    co.sort(key=lambda d: (-d["count"], -abs(d["mean_edge"])))
```

(The `"co_influencers": co[:top_k]` line in the returned dict stays unchanged. Outputs 1–3 are untouched, which the guardrail test enforces.)

In `format_report`, replace the co-influencer print block with:

```python
    L.append("\nUnified co-influencers (label | kind | count | mean_edge):")
    for c in report["co_influencers"]:
        L.append(f"  {c['label']:<28} {c['kind']:<7} x{c['count']:<4} "
                 f"mean_edge={c['mean_edge']:+.4f}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_writefeatures.py -v`
Expected: all PASS (the Task 7 `test_build_report_aggregates` assertion on `co_influencers[0]` still referenced the old feature-only shape — update it: the records in that test have no `unified_top`, so `co_influencers` is now empty; change its last two asserts to `assert rep["co_influencers"] == []`).

> Apply that one-line fix to `test_build_report_aggregates` (Task 7) as part of this step, then re-run.

- [ ] **Step 5: Commit**

```bash
git add clts/writefeatures.py tests/test_writefeatures.py
git commit -m "feat(writefeatures): unified co-influencers (feature+error+token) with unskew guardrail

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 16: Notebook knobs + end-to-end re-verify

Surfaces the two new config knobs and threads them through the run cell, then re-verifies the canonical case shows an error node.

**Files:**
- Modify: `clts/writeFeatures.ipynb` (EDIT cell + run/report cell)
- Create (temporary): `scripts/_run_writefeatures_e2e2.py`

- [ ] **Step 1: Update the EDIT cell**

NotebookEdit (`edit_mode=replace`, the EDIT cell): append the two knobs (and `SUBSET_LABEL` already present from Task 10):

```python
# add these two lines to the EDIT cell, before the closing banner:
TEMPLATE_WORD_LABELS = {"born", "birth", "day", "date"}  # template words kept literal; others -> template:other
INCLUDE_TOKEN_NODES  = True                              # include tok@* rows in the unified view (expected near-zero)
```

- [ ] **Step 2: Update the run/report cell**

NotebookEdit (`edit_mode=replace`, the run/report cell): pass the new kwargs into `run_hypothesis` and add them to `config`:

```python
result = wf.run_hypothesis(
    model, sampler, ct, PEOPLE, target_feature=TARGET_FEATURE, target=TARGET,
    templates=TEMPLATES, cache_dir=CACHE_DIR, n_cap=N_PEOPLE_CAP, seed=SEED,
    top_k=TOP_K, multi_tok_top_k=MULTI_TOK_TOP_K, pos_span_flag=POS_SPAN_FLAG,
    rank_by_abs=RANK_BY_ABS, template_word_labels=TEMPLATE_WORD_LABELS,
    include_token_nodes=INCLUDE_TOKEN_NODES,
)

config = {"target_feature": list(TARGET_FEATURE), "target": TARGET,
          "templates": TEMPLATES, "n_cap": N_PEOPLE_CAP, "seed": SEED,
          "top_k": TOP_K, "multi_tok_top_k": MULTI_TOK_TOP_K,
          "pos_span_flag": POS_SPAN_FLAG, "rank_by_abs": RANK_BY_ABS,
          "template_word_labels": sorted(TEMPLATE_WORD_LABELS),
          "include_token_nodes": INCLUDE_TOKEN_NODES,
          "subset_label": SUBSET_LABEL, "scan": SCAN_NAME}
report = wf.build_report(result, top_k=TOP_K, pos_span_flag=POS_SPAN_FLAG,
                         multi_tok_top_k=MULTI_TOK_TOP_K, config=config)
print(wf.format_report(report))

slug = f"{SUBSET_LABEL}-n{N_PEOPLE_CAP}-s{SEED}"
paths = wf.save_report(report, result["records"], CACHE_DIR,
                       layer=TARGET_FEATURE[0], fidx=TARGET_FEATURE[1], subset_slug=slug)
print("\nsaved:", paths["json"])
print("saved:", paths["csv"])
```

- [ ] **Step 3: Write + run the e2e driver**

```python
# scripts/_run_writefeatures_e2e2.py  (TEMPORARY)
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import clts.writefeatures as wf
from clts.export_tokenizer import ensure_hf_tokenizer
from clts.load_replacement_model import load_replacement_model
from clts.storage import storage_root
from util.bio_sampler import BioSampler
from util.condensed_tokenizer import CondensedTokenizer

DATA_DIR = REPO / "data/bioS_N-Bd_final_grid"
model = load_replacement_model(REPO / "model/grid-L4-H6",
    REPO / "clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final",
    ensure_hf_tokenizer(DATA_DIR), "grid-L4-H6", device="cpu")
sampler = BioSampler(DATA_DIR / "people.json", fields=("birthday",))
ct = CondensedTokenizer.from_remap_path(DATA_DIR / "old_to_new.json")
cache = storage_root() / "clt_feature_explorer" / "grid-L4-H6" / "hyptest"

people = wf.people_in_month(sampler, ct, "August")
result = wf.run_hypothesis(model, sampler, ct, people, target_feature=(3, 4768),
                           target="month", templates="all", cache_dir=cache, n_cap=3, seed=0)
rep = wf.build_report(result, top_k=10, pos_span_flag=3, multi_tok_top_k=5,
                      config={"target_feature": [3, 4768], "target": "month"})
print(wf.format_report(rep))
err_labels = [c for c in rep["co_influencers"] if c["kind"] == "error"]
assert err_labels, "expected at least one error node in the unified co-influencers"
print("\nerror co-influencers:", [c["label"] for c in err_labels[:5]])
print("OK")
```

Run: `clts/.venv-ct/bin/python scripts/_run_writefeatures_e2e2.py`
Expected: `format_report` now prints a "Unified co-influencers" block that includes `err@...` rows (expect `err@last_name@L2`-style labels prominent for August/`L3 F4768`); the `assert` passes and prints `OK`.

- [ ] **Step 4: Delete the driver + manual notebook check**

Run: `rm scripts/_run_writefeatures_e2e2.py`
Then open `clts/writeFeatures.ipynb`, Run All, confirm the printed report includes the unified co-influencers with error rows, and a report JSON/CSV is saved.

- [ ] **Step 5: Commit**

```bash
git add clts/writeFeatures.ipynb
git commit -m "feat(writefeatures): notebook knobs for token-role labels + token nodes; e2e verified

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Addendum Self-Review

**Spec coverage (decisions 9–10):**
- Unified co-influencer of features + error + token nodes → Tasks 13 (`label_nodes`/`unified_top_labels`), 15 (`build_report`). ✓
- Error node → token-role label (`err@last_name@L2`) → Tasks 12 (`token_roles`), 13 (`label_nodes`). ✓
- Layer-major error decode + BOS handling → Tasks 13 (`decode_error_nodes`), 12 (`token_roles` prepends BOS). ✓
- `TEMPLATE_WORD_LABELS` config + `template:other` collapse → Tasks 12, 16. ✓
- `INCLUDE_TOKEN_NODES` toggle → Tasks 13 (`include_tokens`), 14/16 (threaded). ✓
- Unskew guardrail (outputs 1–3 byte-identical) → Task 15 (`test_unskew_guardrail_outputs_1_to_3_unchanged`). ✓
- Empirical anchor (`err@last_name@L2`, error 6% / token 0.03%) → Task 16 e2e assertion. ✓

**Type consistency:** `node_all` is `{"features","errors","tokens"}` everywhere (Tasks 13–14). `unified_top` rows are `{"kind","label","edge"}` produced by `label_nodes` (Task 13), stored by `run_hypothesis` (Task 14), consumed by `build_report` (Task 15) and the integration/e2e checks (Tasks 14, 16). `roles` is a list indexed by graph position with `roles[0]=="BOS"` (Tasks 12–14). `feature_rank`/`feature_multitoken` still read the feature list only — now `node_all["features"]` (Task 14), preserving outputs 1–3.

**Carry-over fix flagged:** Task 15 Step 4 updates the Task 7 `test_build_report_aggregates` co-influencer asserts (records without `unified_top` → `co_influencers == []`), since the co-influencer source changed from `node_top_features` to `unified_top`.

**Placeholder scan:** none — every step has complete code and an exact command.
