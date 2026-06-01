# Broader CLT Training Sweep — Design

**Date:** 2026-05-31
**File touched:** `clts/trainCLT.py` (`build_sweep_config`, docstring/CLI help)

## Problem

The first CLT sweep (`expansion {8,16} × l0 {2,5,10} × lr {3e-5,1e-4}`, 12 runs,
`n_examples=10k`, `epochs=50`) did not reach a high enough `ce_recovered`, and
the resulting attribution graphs had **error nodes that were too large** — they
absorbed too much of each node's contribution.

Error-node magnitude is the share of activation the transcoder fails to
reconstruct, so it tracks reconstruction error directly. The best run
(`apricot`, exp16/l0=2/lr=1e-4) still had poor mid-layer reconstruction:
`nmse_L1 ≈ 0.44`, `nmse_L2 ≈ 0.55` (only ~45–55% of variance explained).

## Diagnosis (from wandb export)

Ranking all 12 runs by `clt_eval/ce_recovered` shows three monotonic trends,
all pointing **out of the original grid corner**:

- **lr:** `1e-4` beat `3e-5` on all 6 pairings → fix at `1e-4`, drop the dead low end.
- **l0_coefficient:** lowest (`2`) won → push down; drop the always-worst `10`.
- **expansion:** `16` beat `8` everywhere → capacity is under-used; push up.

The optimum sat in the literal corner of the grid (max expansion, min l0, max
lr), so the grid must be extended past that corner.

## Data availability

- `data/.../people.json`: **50,000 people**; BioSampler exposes **46 templates**
  → ~2.3M unique (person × template) bio renderings possible.
- An "example" = one 512-token packed row (~15–20 EOS-prefixed bios), **not** one
  bio. `n_examples=10k` ≈ 5.1M tokens ≈ ~200k renderings — already touches every
  identity several times but uses only ~9% of distinct paraphrase pairs.
- `n_examples=50k` ≈ 25.6M tokens ≈ ~1M renderings (~43% of pairs): **5× more
  distinct paraphrasings/packings, each seen fewer times.** Genuine diversity,
  with headroom to ~100k+ before pairs exhaust. So raising `n_examples` buys real
  unique data, not repetition.

## Step-budget identity

`total_steps = epochs × n_examples × context_size / BATCH_SIZE`
            `= epochs × n_examples × 512 / 4096`.

At **n_examples=50k, epochs=10 → 62,500 steps**, *identical* to the old best
run's budget (`10k × 50`), but on **5× more unique data**. Same compute, better
coverage — this is why epochs drops from 50 to 10.

## Two-stage workflow

1. **Sweep stage (this change):** train the grid below cheaply to *rank*
   `(expansion, l0)` regions.
2. **Final stage (no code needed):** train the winning config(s) longer/bigger
   via the existing single-run CLI to produce the ship-quality CLT. Optionally
   re-test `lr=2e-4` on the winner there.

Caveat carried into the final stage: rankings only transfer if budgets stay
within ~3–5×; keeping the sweep at the old step budget (62.5k) protects this.
"Raise epochs" alone re-shows the same tokens and risks packing-overfit that
flatters 64-row eval CE without shrinking error nodes — so the final-stage lever
is more `n_examples`, with epochs only as high as needed to converge.

## The new grid

```python
"parameters": {
    "expansion":      {"values": [4, 8, 16, 32]},  # capacity curve below + above old max of 16
    "l0_coefficient": {"values": [1.0, 2.0, 5.0]},  # balance band; dropped always-worst 10, added 1
    "lr":             {"value": 1e-4},              # fixed — won all 6 pairings; 2e-4 deferred to final stage
    "n_examples":     {"value": 50_000},            # 5× unique data vs old 10k
    "epochs":         {"value": 10},                # holds step budget == old best run (62.5k)
},
```

**4 × 3 = 12 runs.**

### Wall-clock budget

`wandb agent` on one GPU runs trials sequentially. Fitting the old runtimes
(exp8 ≈ 3,960s, exp16 ≈ 6,810s) gives `runtime ≈ 1110 + 356·expansion`, which
reproduces the old 12-run sweep at **17.95h** vs. its actual **18h** — so the
model is trusted.

| expansion | per run | × 3 l0 |
|---|---|---|
| 4  | ~42 min | ~2.1 h |
| 8  | ~66 min | ~3.3 h |
| 16 | ~1.9 h  | ~5.7 h |
| 32 | ~3.5 h  | ~10.5 h |
| **total** | | **~21.5 h** |

Fits one `submit_job_psc.sh` job (`--time=24:00:00`), next to the old 18h.

**`expansion=64` is deliberately excluded from the grid:** at ~6.6h/run × 3 it
would add ~19.5h and push the sweep to ~41h, well past the 24h cap. It moves to
the final stage — if `ce_recovered` is still climbing at expansion=32, test 64 as
a single targeted run on the winning l0.

Keep `method=grid`, `metric=final_eval/ce_recovered (maximize)`, and the existing
hyperband early-terminate.

## Rationale vs. goals

- **Shrink error nodes** ← the `expansion {4→64}` curve reveals how much capacity
  mid-layer reconstruction needs; 5× data fights overfitting to fixed packings.
- **Balance sparsity** ← `l0 {1,2,5}` improves reconstruction without driving the
  already-high active-feature counts (~300 at L3) further up.

## Contingency

If `expansion=32` OOMs, fall back to `{4,8,16,24}`. If the sweep is faster than
expected and time allows, `64` can be appended back into the grid (~+19.5h →
~41h, requires a >24h walltime or parallel agents).

## Out of scope

- Eval still uses 64 rows (noisy but adequate for ranking). Bumping to 128 was
  offered and declined for now.
- No `--final` mode added; the existing single-run CLI covers the final stage.
