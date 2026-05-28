# Cross-Layer Transcoder Training Pipeline — Design

**Status:** approved, ready for implementation plan
**Date:** 2026-05-27
**Subproject:** 1 of 3 (CLT training → attribution-graph engine → interactive UI)

## Goal

Train Anthropic-style Cross-Layer Transcoders (CLTs) on any of the project's existing Llama checkpoints under [model/](../../../model/), producing trained CLTs whose on-disk format is consumable by Anthropic's `circuit-tracer` library without conversion. The trainer must be generic over model shape — `n_layers`, `d_model`, `n_heads` are all read from the loaded model's config.

## Non-goals

- Attribution graphs (subproject #2)
- Interactive feature browsing / circuit UI (subproject #3)
- Convergence-quality eval beyond reconstruction MSE and CE-recovered
- Multi-GPU / DDP support
- Per-layer transcoder or single-layer transcoder variants (CLT only)

## Background

Cross-Layer Transcoders (Lindsey et al., 2025, ["Circuit Tracing"](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)) extend transcoders so that features at source layer `L` can contribute to MLP outputs at L, L+1, …, N-1. They are the substrate Anthropic uses for attribution-graph circuit tracing.

The project already has a working per-layer SAE pipeline ([saes/trainSAE.py](../../../saes/trainSAE.py)) built on `sae_lens`, with `CondensedTokenizer`, `BioSampler`, and `DiverseBioSubset` data utilities. The CLT trainer reuses the data pipeline and CLI/storage conventions but cannot reuse `sae_lens`'s `LanguageModelSAETrainingRunner`, `ActivationsStore`, or `SAE` class — those are single-hook by design, while a CLT is one module spanning all layers with a joint multi-hook loss.

### Framework choice

Research surveyed CLT-Forge, EleutherAI/clt-training, Anthropic's circuit-tracer, mntss/clt-gemma-2-2b-426k, and the Dunefsky et al. single-layer transcoder repo. Decision: **build from scratch on transformer_lens**, reusing `sae_lens.saes.jumprelu_sae.JumpReLU` as the activation primitive. This matches the project's existing dependency surface and avoids forcing the data pipeline into a third-party library's loader. The CLT class deliberately mirrors Anthropic's `circuit-tracer.CrossLayerTranscoder` so trained weights load into the consumer tooling without a converter.

## Section 1 — Architecture

Single `nn.Module` matching the on-disk format expected by `circuit-tracer` and `mntss/clt-gemma-2-2b-426k`:

```python
class CrossLayerTranscoder(nn.Module):
    # N = model.cfg.n_layers, D = d_model, d_t = expansion * D
    W_enc:     nn.Parameter           # [N, d_t, D]
    b_enc:     nn.Parameter           # [N, d_t]
    threshold: nn.Parameter           # [N, d_t]    (JumpReLU per-feature)
    W_dec:     nn.ParameterList       # len N; entry i is [d_t, N - i, D]
    b_dec:     nn.Parameter           # [N, D]
```

Hook plumbing (templated, configurable):
- Encoder inputs: `blocks.{L}.hook_resid_mid` for `L ∈ 0..N-1` (residual stream after attention, before MLP — Anthropic-faithful)
- Decoder targets: `blocks.{L'}.hook_mlp_out` for `L' ∈ 0..N-1` (the MLP's additive contribution to the residual stream)

Forward pass given captured `x[L] = resid_mid[L]` and `y[L'] = mlp_out[L']`:

```
a[L]    = JumpReLU( einsum("nd,bd->bn", W_enc[L], x[L]) + b_enc[L], threshold[L] )    for L in 0..N-1
ŷ[L']   = b_dec[L'] + Σ_{L ≤ L'} einsum("nd,bn->bd", W_dec[L][:, L'-L, :], a[L])      for L' in 0..N-1
```

The decoder routing uses `L'-L` to index into `W_dec[L]`'s second axis (which packs all downstream targets for source `L`).

`n_heads` does not enter the CLT — both encoder input and decoder target are `d_model`-shaped, independent of attention factorization.

**Parameter counts** (for `d_model=192`, `expansion=16` → `d_t=3072`):
- N=4 (current checkpoints): 10 decoders, ~8M params total
- N=6: 21 decoders, ~12M params
- N=12: 78 decoders, ~30M params

All tractable single-GPU at fp32.

### Save format

Match circuit-tracer's `to_safetensors` / `_load_state_dict` layout: 2N safetensors files per CLT.

- `W_enc_{i}.safetensors` contains keys: `W_enc_{i}`, `b_enc_{i}`, `b_dec_{i}`, `threshold_{i}`
- `W_dec_{i}.safetensors` contains key: `W_dec_{i}`
- `config.yaml` matches circuit-tracer's schema (model identifier, n_layers, d_model, d_transcoder, encoder/decoder hook templates)

A test (Section 4 test 5) asserts the on-disk keys match circuit-tracer's loader expectations so format drift fails fast.

## Section 2 — Training loop, loss, eval

### Activation capture

For each batch: run `model.run_with_cache(names_filter=...)` once with the base model in `eval()` + `torch.no_grad()`. Capture 2N tensors (N residual-mid, N mlp-out), each shape `[B, T, d_model]`. Reshape to `[B·T, d_model]` (positions are i.i.d. for loss purposes). Feed both sets to the CLT.

Strictly more efficient than `sae_lens`'s per-hook `ActivationsStore` — one forward yields everything. Memory cost: `2·N·B·T·d_model·4 bytes`. For N=4, B=4096/T=64, d_model=192 → ~6 MB.

### Loss

Faithful to Anthropic's methods paper:

```
L_recon    = (1/N) · Σ_{L'} MSE( ŷ[L'], y[L'] )
L_sparsity = λ · Σ_{L} Σ_{i} tanh( c · ||W_dec_norms[L, i]|| · a_i^L )
L_preact   = α · Σ_{L} || preact[L] ||²      (pre-JumpReLU L2 — matches existing SAE recipe)
L_total    = L_recon + L_sparsity + L_preact
```

`W_dec_norms[L, i]` is the L2 norm of feature `i`'s decoder column **summed across all downstream targets** (Anthropic's decoder-norm-weighted formulation — features that write large into many layers get penalized more).

Hyperparam defaults adopted from [saes/trainSAE.py](../../../saes/trainSAE.py):
- `c = 4.0` (tanh scale, matches `jumprelu_tanh_scale`)
- `α = 3e-6` (matches `pre_act_loss_coefficient`)
- `λ` (sparsity coefficient) swept over `{2.0, 5.0, 10.0}` matching existing SAE sweep
- JumpReLU init threshold = `0.1`, bandwidth = `2.0` (match `jumprelu_init_threshold`, `jumprelu_bandwidth`)

### L0 warmup

Ramp `λ` linearly from 0 to its target over the first 10% of training steps. Mirrors `l0_warm_up_steps = total_training_steps // 10` from [trainSAE.py:285](../../../saes/trainSAE.py#L285).

### Optimizer

Adam(β1=0.9, β2=0.999), constant LR with short warmup + small linear decay tail. Matches [trainSAE.py:301-306](../../../saes/trainSAE.py#L301-L306):
- `lr_warm_up_steps = total_training_steps // 50`
- `lr_decay_steps = total_training_steps // 20`

### Dead-feature tracking

Per source layer L, maintain a rolling count of "steps since each feature last fired above threshold." Log per-layer dead-feature fraction every 1000 steps (`feature_sampling_window`). **No revival in v1** — Anthropic doesn't do dead-feature revival for CLTs; if v1 shows pathological dead-feature rates we add it.

### Coverage check (startup)

Compute and print at trainer startup:
- Actual bio-length distribution (min / median / max tokens per bio)
- Computed exposures per person per epoch
- Fraction of people with zero exposures in one epoch (under random sampling)

Cheap insurance against silent undertraining; lets the user bump `--n-examples` manually if coverage looks thin on a new model.

`DiverseBioSubset` is kept unchanged (no `ExhaustiveBioSubset`). Rationale: the model uses compositional features at `d_model=192` (50k people cannot fit as orthogonal directions in 192 dims), so coverage of specific people matters less than coverage of shared subfeatures, which is dense even under light random sampling.

### Eval

Held-out subset of `DiverseBioSubset` with `seed=SAE_seed + 1` (matching SAE convention at [trainSAE.py:172-176](../../../saes/trainSAE.py#L172-L176)). Two modes:

- **`--eval-mode quick`** (default for sweep trials): 64 rows ≈ 1000 bios ≈ 2% of population. Fast, fine for hyperparam ranking. Matches existing SAE eval setup.
- **`--eval-mode full`**: one bio per `(person, template)` pair so every fact in `people.json` is hit at least once. ~2.3M bios ≈ 100M tokens. Used for final eval on sweep winners.

Metrics computed on held-out:
- **Per-layer reconstruction MSE** and **normalized MSE** (MSE / Var(target))
- **Per-layer L0** (avg #features firing above threshold per token)
- **Per-layer dead-feature fraction**
- **Full-replacement CE-recovered** (headline metric): run base model with ALL N MLP outputs simultaneously replaced by CLT predictions; measure CE delta vs original. This is the global analog of your existing SAE `ce_recovered`.
- **Per-layer CE-recovered** (diagnostic): replace one MLP at a time, measure CE delta.
- **Full-mode only**: per-person and per-template CE-recovered breakdowns + "fraction of people whose birthday is still top-1 predicted after full-MLP replacement."

## Section 3 — CLI, sweep, storage, wandb

### File layout

```
clts/
├── __init__.py
├── clt.py          # CrossLayerTranscoder nn.Module (Section 1 architecture)
├── trainCLT.py     # CLI + training loop (this section)
├── evalCLT.py      # held-out eval helpers (Section 2 metrics)
└── clt_runs/       # local symlink to STORAGE_ROOT/clt_runs (gitignored)
```

Mirrors the existing [saes/](../../../saes/) directory structure.

### CLI

```
python clts/trainCLT.py --model-dir <path> --data-dir <path> [flags]

# Required (same flag names as SAE for consistency):
--model-dir              HF Llama checkpoint dir
--data-dir               Dataset dir with people.json + old_to_new.json
--model-name             Identifier for output paths (defaults to parent dir)

# CLT-specific (replaces SAE's --hook / --layers / --hook-template):
--enc-hook-template      Default: "blocks.{layer}.hook_resid_mid"
--dec-hook-template      Default: "blocks.{layer}.hook_mlp_out"
                         CLT spans ALL layers; no --layers flag — n_layers is
                         read from model config at startup.

# Hyperparams:
--expansion              Default 16   (was --sae-mult in SAE)
--l0                     Default 5.0
--lr                     Default 5e-5
--epochs                 Default 30
--n-examples             Default 10_000
--context-size           Default 512

# Eval mode:
--eval-mode              {quick, full}    Default: quick
                         quick = 64 rows, matches SAE convention
                         full  = every (person × template) once + per-person CE breakdown

# Sweep:
--sweep                  Launch wandb grid sweep over (expansion × l0 × lr).
                         No --layers — CLT is one model spanning all layers.
```

### Sweep structure

Fundamentally different from SAE sweeps. SAE sweeps are per-layer (one sweep per layer × hp grid). **CLT sweeps once over the hp grid only** because the CLT is one model spanning all layers.

Grid (matches SAE sweep values):
```python
{
    "expansion":      [8, 16],
    "l0_coefficient": [2.0, 5.0, 10.0],
    "lr":             [3e-5, 1e-4],
    "epochs":         50,
}
```
12 trials per sweep. Hyperband early-termination on `final_eval/ce_recovered` (maximize), matching the SAE sweep setup at [trainSAE.py:218-227](../../../saes/trainSAE.py#L218-L227).

### Trial naming

```
mult{expansion}_l0{x}_lr{r}_ep{e}_n{k}
```

No `L{layer}_` prefix (SAE's `trial_name` adds it; CLT drops it because each CLT spans all layers).

### Storage layout

```
STORAGE_ROOT / clt_runs / <model-name> / <sweep-{id}|standalone> / <trial> / final/
                                                                              ├── W_enc_0.safetensors
                                                                              ├── ...
                                                                              ├── W_enc_{N-1}.safetensors
                                                                              ├── W_dec_0.safetensors
                                                                              ├── ...
                                                                              ├── W_dec_{N-1}.safetensors
                                                                              └── config.yaml      # circuit-tracer schema
```

`STORAGE_ROOT` reused from [trainSAE.py:63](../../../saes/trainSAE.py#L63) (PSC Ocean: `/jet/home/friedmae/data_storage/LM4_Results`). Pre-flight write-probe at startup ([trainSAE.py:117-126](../../../saes/trainSAE.py#L117-L126) style) catches quota / permission issues in <1s.

### wandb

Same project `interpLM4` as SAEs. Metric namespace `clt_*` to avoid collision:

- **Train** (every 30 steps): `clt_train/mse_total`, `clt_train/mse_L{i}`, `clt_train/l0_L{i}`, `clt_train/dead_frac_L{i}`, `clt_train/sparsity_loss`, `clt_train/preact_loss`
- **Held-out** (every 600 steps = 20 log calls): `clt_eval/mse_total`, `clt_eval/mse_L{i}`, `clt_eval/ce_recovered` (full-MLP replacement), `clt_eval/ce_recovered_L{i}` (single-MLP replacement diagnostic)
- **Final** (after training): `final_eval/ce_recovered`, `final_eval/mse_total`, `final_eval/mse_L{i}` for each i, `storage_path`
- **Sweep selection metric**: `final_eval/ce_recovered` (maximize)

### Multi-process / threading

Use the same `_patch_signal_for_worker_threads` shim from [trainSAE.py:386-398](../../../saes/trainSAE.py#L386-L398) when running under `wandb.agent` (SIGINT handler crashes on worker threads otherwise).

### Module-level state for sweep reuse

Mirror the [trainSAE.py:88-93](../../../saes/trainSAE.py#L88-L93) pattern: `setup()` loads the base model, tokenizer, sampler, and eval tokens once per process; every sweep trial reads from globals. Avoids reloading the base model per trial.

## Section 4 — Testing & verification

**File:** `tests/test_clt.py`, single file, parallel to existing [tests/test_sae_explorer.py](../../../tests/test_sae_explorer.py).

### Unit tests (pytest, each <30 lines)

1. **`test_shapes`** — Construct `CrossLayerTranscoder` for `n_layers=4, d_model=8, expansion=2`. Assert:
   - `W_enc.shape == (4, 16, 8)`
   - `len(W_dec) == 4`, `W_dec[i].shape == (16, 4-i, 8)`
   - `b_enc.shape == (4, 16)`, `b_dec.shape == (4, 8)`, `threshold.shape == (4, 16)`

2. **`test_forward_pass_dimensions`** — Feed random `[B=2, T=3, d_model=8]` residual + MLP-out tensors through `clt.forward(...)`. Assert predicted MLP outputs have shape `[2, 3, 8]` per layer, loss is finite scalar.

3. **`test_cross_layer_writes`** — Set `W_enc[0]` so a single feature fires on a fixed input; zero out `W_enc[1..3]`. Set `W_dec[0][:, k, :]` nonzero only for one specific target `k`. Verify the CLT prediction is nonzero only at target layer `k` and zero at other layers. Catches decoder routing off-by-one (most likely correctness bug).

4. **`test_save_load_roundtrip`** — Save CLT to a temp dir, reinstantiate from disk, compare every Parameter elementwise.

5. **`test_circuit_tracer_format_keys`** — Save CLT, open `W_enc_0.safetensors` and `W_dec_0.safetensors` with `safetensors.safe_open`, assert key names match circuit-tracer's loader expectations (`W_enc_0`, `b_enc_0`, `b_dec_0`, `threshold_0` in encoder file; `W_dec_0` in decoder file). Protects subprojects #2 and #3 against format drift.

6. **`test_jumprelu_gradients_flow`** — Backward through a forward pass; assert gradients land on `W_enc`, every `W_dec[i]`, both biases, and `threshold`. Catches accidentally-frozen params (easy to miss with `ParameterList`).

7. **`test_loss_decreases_one_optimizer_step`** — Tiny synthetic data, one Adam step at `lr=1e-2`, verify total loss strictly decreases. End-to-end sanity check on loss + backward + optimizer.

### Smoke test (documented in module docstring, not in CI)

```bash
python clts/trainCLT.py \
    --model-dir model/BD_llama_3heads_12epoch_4layers \
    --data-dir  data/BD_llama_inital \
    --epochs 1 --n-examples 100 --expansion 4
```

Acceptance criteria:
- Trainer starts; prints model + data + storage paths
- Prints Section 2 coverage stats (exposures per person, etc.)
- Trains for one epoch without crashing
- `clt_train/mse_total` decreases from step 0 to final step
- `final_eval/ce_recovered` is reported and is a real number (not NaN)
- Output dir contains all 2N safetensors files + `config.yaml`

### Format-compatibility check (manual, one-time)

After first successful training run, point Anthropic's `circuit-tracer` at the output dir (or a stub of its `_load_state_dict`) and confirm it loads without error. If it fails, test 5 caught the wrong thing and we update both.

### Out of scope for testing

- Convergence quality (too noisy + expensive for CI)
- wandb logging behavior (low value, mock-heavy)
- Sweep agent + hyperband (manually verified on first sweep)
- Multi-GPU / DDP (single-GPU only)

## Dependencies / migrations

- No new top-level dependencies. Reuses `transformer_lens`, `torch`, `safetensors`, `wandb` already in the project. Imports `sae_lens.saes.jumprelu_sae.JumpReLU` only.
- No changes to existing `saes/`, `model/`, or `util/` code. New code lives entirely under new `clts/` directory plus one new test file.

## Open questions deferred to implementation

- Exact `config.yaml` schema circuit-tracer expects — will inspect their loader during implementation and match exactly.
- Whether to expose `--jumprelu-init-threshold` and `--jumprelu-bandwidth` as CLI flags or hard-code at SAE defaults. Default: hard-code; expose only if tuning is needed.
- Whether per-layer CE-recovered (single-MLP replacement diagnostic) is worth the extra eval cost during sweeps, or only computed on final-eval runs. Default: only on final eval, to keep sweep trials fast.
