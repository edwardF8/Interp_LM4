# CLT metrics reference

Every metric logged by `trainCLT.py` / `evalCLT.py`, with its exact definition
from the code and how to read it. Definitions are pulled from
[`clts/evalCLT.py`](evalCLT.py) (`compute_layer_metrics`, `ce_recovered_full`,
`ce_recovered_per_layer`) and [`clts/clt.py`](clt.py) (`compute_loss`).

Alot of the metrics have a L0/L1, etc in front of it. 

## Setup: what the CLT does

The cross-layer transcoder **replaces the MLPs**. For each layer `L` it reads the
residual stream *into* the MLP (`x = hook_resid_mid`) and predicts the MLP's
*output* (`y = hook_mlp_out`):

```
a[L]      = JumpReLU(W_enc[L] · x[L] + b_enc[L])      # features that fire
ŷ[L']     = b_dec[L'] + Σ_{L ≤ L'} a[L] · W_dec[L→L']  # cross-layer: earlier features feed later outputs
```

- `JumpReLU(z) = z · 1[z > threshold]` (threshold init 0.1) — a feature is
  "active" only when its pre-activation clears its learned threshold.
- `d_t = expansion × d_model` is the number of features per layer (the dictionary
  size). Capacity scales with `expansion`.
- Reconstruction quality = how close `ŷ` is to the true MLP output `y`. **This is
  what the attribution-graph "error node" measures**: the error node at a layer is
  the leftover `y − ŷ`, so smaller reconstruction error ⇒ smaller error nodes.

Activations are flattened to `[B·T, d_model]`, so every metric is **per token
position**, averaged over all positions in the batch.

---

## Namespaces — same metrics, different when/where

| Prefix | When | Data | Use |
|---|---|---|---|
| `clt_train/*` | every 30 steps | current **training** batch | live training curves (noisy, seen data) |
| `clt_eval/*` | every 600 steps | 64-row **held-out** set | convergence tracking during the run |
| `final_eval/*` | once at the end | 64-row **held-out** set | **the numbers you compare across runs** |

Train vs eval differing is normal (train = seen batches + it's the raw loss
value; eval = held-out recompute). A large train↔eval gap = overfitting.

---

## Reconstruction metrics

### `mse_L{L}` — per-layer reconstruction MSE
`mean((ŷ[L] − y[L])²)` over batch and `d_model` dims. Raw squared error of
predicting layer `L`'s MLP output. **Units depend on activation scale**, so it's
only comparable across runs on the *same* layer — use `nmse` for an absolute read.

### `mse_total` — mean per-layer MSE
`(1/N) · Σ_L mse_L` — the **average** across layers (not the sum). This is the
`recon` term in the loss.

### `nmse_L{L}` — normalized MSE (the one to watch) ⭐
`mse_L / Var(y[L])` = **fraction of the layer's activation variance the CLT fails
to explain** (FVU). Scale-free:

- `nmse = 0` → perfect reconstruction.
- `nmse = 1` → no better than predicting the constant mean.
- `1 − nmse` = R² = variance explained.

This is the direct knob on error-node size. In the first sweep the best run had
`nmse_L1 ≈ 0.44`, `nmse_L2 ≈ 0.55` — i.e. ~45–55% of mid-layer variance
unexplained, which is why the error nodes dominated. **Goal: drive `nmse` down.**

---

## Sparsity / feature-activity metrics

### `l0_L{L}` — average active features per token
`mean over tokens of count(a[L] > 0)`. How many of the `d_t` features fire on a
typical token. Lower = sparser = more interpretable; higher = denser = usually
better reconstruction. (Best run hit `l0_L3 ≈ 303` active features — quite dense.)
Interpret relative to `d_t = expansion × d_model`.

There is a **tension**: lowering `l0_coefficient` or raising `expansion` improves
`nmse` but pushes `l0` up. Watch both together.

### `dead_frac_L{L}` — fraction of never-firing features
`fraction of features whose activation is 0 across the entire eval batch`. Dead
features are wasted dictionary capacity. High `dead_frac` (e.g. >0.3) means much
of the `expansion` you paid for isn't being used — a sign the sparsity penalty is
too strong or `expansion` is too large for the data.

---

## Training-loss components (`clt_train/*` only)

Total loss = `recon + sparsity + preact` (from `compute_loss`).

### `mse_total`, `mse_L{L}`
The `recon` term — same MSE as above, on the training batch.

### `sparsity_loss`
`l0_coefficient · Σ_L Σ_features tanh(4 · |a · ‖dec_norm‖|)`, averaged over batch.
A **smooth surrogate for L0** (each active feature contributes ≈1 once `tanh`
saturates), **weighted by each feature's decoder norm** so features that move the
reconstruction more are penalized more. This is what's actually minimized to
induce sparsity — *not* the raw `l0` count (which is non-differentiable).

### `preact_loss`
`3e-6 · Σ_L mean(preact²)` — a tiny L2 penalty on encoder pre-activations. Keeps
pre-activations from drifting large so the JumpReLU thresholds stay well-behaved.
Intentionally small; if it ever dominates, something is off.

### `l0_coef_effective`
`l0_coefficient · min(1, step / l0_warmup)` where `l0_warmup = total_steps / 10`.
The sparsity coefficient **ramps from 0 to `l0_coefficient` over the first 10% of
steps**, then holds. Early in training this is below the configured value by
design — don't read sparsity off the first ~10%.

### `lr`
Current learning rate after warmup. `lr_warmup = total_steps / 50` (first 2% of
steps ramp 0→`lr`). Both warmups are in **steps**, so they're unaffected by the
epoch count.

---

## Cross-entropy / faithfulness metrics

These test the CLT as an actual model component: swap the real MLP(s) for the
CLT's prediction and measure the language-model loss.

### `ce_orig`
Baseline next-token cross-entropy of the **unmodified** model on the eval tokens.
Reference point (lower is the original model's quality).

### `ce_clt`
CE when **all** MLP outputs are simultaneously replaced by CLT predictions. The
real end-to-end faithfulness number — closer to `ce_orig` is better.

### `ce_zero`
CE when **all** MLP outputs are replaced by **zeros**. A worst-case reference:
how bad the model gets with the MLPs effectively deleted.

### `ce_recovered` — the headline sweep metric ⭐
```
ce_recovered = (ce_zero − ce_clt) / (ce_zero − ce_orig)
```
**Fraction of the MLPs' contribution to CE that the CLT recovers.**

- `1.0` → CLT is as good as the real MLPs.
- `0.0` → no better than deleting the MLPs.
- `< 0` → worse than zeroing (broken).

This is the sweep's selection metric (`metric: final_eval/ce_recovered`,
maximize). First sweep topped out at ~0.915; the goal is higher.

### `ce_clt_L{L}` (`final_eval/*` only)
CE when replacing **only** layer `L`'s MLP (one at a time, others left intact).
Per-layer faithfulness diagnostic: compare each to `ce_orig` to see **which
layer's transcoder hurts the model most** — that's where reconstruction is
failing in a way that matters for the output, and where to focus.

---

## How they connect (cheat-sheet)

- **Large attribution error nodes** ⇐ high `nmse_L{L}` ⇐ too little capacity
  (`expansion`), too much sparsity pressure (`l0_coefficient`), or undertraining.
- **Poor `ce_recovered`** is usually downstream of high `nmse` — but check
  `ce_clt_L{L}` to find the culprit layer; a layer can reconstruct "okay" in MSE
  yet still tank CE if it misses output-critical directions.
- **Wasted capacity** ⇐ high `dead_frac` (penalty too strong / `expansion` too big).
- **Too dense to interpret** ⇐ high `l0` (penalty too weak) — the cost of chasing
  `nmse` down. The sweep's job is to find the `expansion × l0` point that
  minimizes `nmse`/maximizes `ce_recovered` without `l0`/`dead_frac` going wild.
