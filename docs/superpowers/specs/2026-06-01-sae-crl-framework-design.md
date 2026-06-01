# SAE-CRL: Causal-Representation-Learning SAE for Interp_LM4

- **Date:** 2026-06-01
- **Status:** Design approved; pending spec review (one open item flagged in §3)
- **Author:** (brainstormed with Claude Code)
- **Scope this round:** Core only — trainable model + data/windowing pipeline + training driver + eval metrics + checkpoint save/load. Relation analysis/dashboards, synthetic identifiability validation, SAEBench comparison, and circuit-tracer attribution wiring are **out of scope** (see §13).

---

## 1. Goal

Implement the linear *temporal–instantaneous* SAE from Song et al., "LLM Interpretability with Identifiable Temporal-Instantaneous Representation" (NeurIPS 2025; arXiv:2509.23323v2) as a new interpretability method inside `Interp_LM4`, trained on this repo's own small Llama checkpoints, condensed tokenizer, and bioS dataset.

**Faithfulness rule:** follow the **paper's equations**. Where the reference implementation (`github.com/xiangchensong/temp-inst-sae`) *code* contradicts its own paper, follow the paper (each such case is documented in §3). The reference behavior is preserved behind config flags where cheap.

**The method in one paragraph.** A *linear* autoencoder produces latents `z = encode(x)` and reconstructs `x̂ = decode(z)`. The latents carry a learned causal structure: time-delayed matrices `B_τ` (`z_{t−τ} → z_t`) and a strictly-lower-triangular instantaneous matrix `M` (`z_t → z_t`, an acyclic DAG within a token). Training minimizes (1) reconstruction MSE, (2) an **independent-noise** term — L1 on the SEM residual `ε̂ = z_t − M z_t − Σ_τ B_τ z_{t−τ}` (Laplacian prior), and (3) **sparsity** — L1 on `B_τ` and `M`; latent sparsity is enforced by TopK. The trained `B_τ` (max-pooled to `aggB`) and `M` are the discovered concept-to-concept relations.

---

## 2. Inputs — the integration seam (reused, not reinvented)

All four inputs come through the shared seam that `clts/` and `saes/` already use:

| Input | Source | Notes |
|---|---|---|
| Base model | `clts.tl_model.build_hooked_transformer(model_dir, device)` | HF Llama → TransformerLens `HookedTransformer`. Default `model/grid-L4-H6` (4 layers, `d_model=384`, vocab 1836, `n_ctx=512`). |
| Tokenizer | `util.condensed_tokenizer.CondensedTokenizer.from_remap_path(data_dir/"old_to_new.json")` | Reduced GPT-2 vocab, size 1836, `eos=bos=pad=1835`. |
| Data | `util.bio_sampler.BioSampler(data_dir/"people.json", fields=("birthday",))` | Renders byte-identical bios via the sibling `Training_On_LM4` package. `people.json` confirmed present in `data/bioS_N-Bd_final_grid/`. |
| Activations | `model.run_with_cache(tokens, names_filter=...)` at `blocks.{L}.hook_resid_post` | Default layer `L=2`. |

**RMS-norm eps:** leave at the TransformerLens default `1e-5` (the repo convention `build_hooked_transformer` already uses; consistent with existing SAE/CLT runs). Expose `--eps` to override to `1e-6` (the checkpoint's exact `rms_norm_eps`). Document, don't change silently.

---

## 3. Decisions

**Settled (from brainstorming):**
- **Hook = `blocks.{L}.hook_resid_post`** (paper-faithful: the paper decomposes the residual stream out of a block; also the more standard modern SAE substrate). CLI-configurable via a hook template containing `{layer}`. Default `L=2` (mid of 4; note `L=3 ≈` the paper's 8/12 relative depth).
- **Windowing = within-bio only.** No `τ+1` window may cross an EOS/bio boundary. Implemented via one-bio-per-row corpus (§6).
- **`τ = 5`** (default). Forced deviation from the paper's main `τ=20`: birthday-only bios are ~12–20 tokens, so a width-21 within-bio window yields ≈0 windows. `τ=5` is itself a config the paper ran. Configurable, with a bio-length-histogram guardrail (§11).
- **Paper-over-reference-code resolutions:**
  1. **`M` strictly lower-triangular:** `get_M() = tril(M, diagonal=-1)` (zero diagonal — DAG / no self-instantaneous-cause; paper §3, §4.3). *Reference code uses `diagonal=1`, which keeps the diagonal + a super-diagonal — inconsistent with the paper and with the repo's own synthetic code; treated as a bug.*
  2. **No L1 on latents by default:** paper loss is `L_s = Σ_τ‖B_τ‖₁ + ‖M‖₁` only (Eq. 8); latent sparsity is TopK's job (App. A.5.1). *Reference adds an undocumented `l_spZ` term.* Exposed as `--l-spZ` (default `0.0`).
  3. **Drop `w` and `mse_Zt`:** neither is in the paper; both are off-by-default in the reference anyway.

**OPEN — confirm during spec review — TopK placement.** There is a genuine paper-vs-reference-code discrepancy:
- **Paper reading (recommended default):** App. A.5.1 says TopK enforces "sparsity in the hidden feature activations," and Fig. 2 shows the encoder producing `ẑ_t`. So apply **TopK to the encoder output `z = encode(x)`** — making this a proper sparse TopK-SAE whose sparse code is used by *both* the decoder and the SEM. Keep TopK active at **train and eval** (so latents are genuinely sparse, `L0 = k`, and recon/CE-recovered/SAEBench-style metrics are meaningful and comparable to the existing SAEs).
- **Reference-code reading:** applies TopK to the *structural prediction* `Ẑ_t = M z_t + Σ_τ B_τ z_{t−τ}` (leaving `z` dense) and disables TopK at eval — i.e., a dense linear AE plus a sparsified causal prediction.
- **Plan:** default to the paper reading (`--topk-mode latent`); provide `--topk-mode prediction` for exact reference replication. **This is the one new interpretive call not previously surfaced; flagged here for explicit confirmation.**

---

## 4. Package layout — new `sae_CRL/`

(`sae_clrs/` does not actually exist; create `sae_CRL/` fresh. `CRL` = Causal Representation Learning.)

```
sae_CRL/
  __init__.py          # empty; makes sae_CRL an importable package
  sae_crl.py           # class SAE_CRL(nn.Module): the model + loss + save/load
  windows.py           # one-bio-per-row corpus + within-bio τ+1 window indexing
  evalSAE_CRL.py       # activation capture + recon / L0 / CE-recovered / structure metrics
  trainSAE_CRL.py      # driver: setup()+globals, train loop, wandb sweep (clone of trainCLT.py)
  storage.py           # storage_root() resolver (mirrors clts/storage.py)
  tests/
    test_windows.py        # no window crosses a bio boundary; counts correct
    test_model.py          # parameter shapes, forward shapes, loss components, strict-lower M
    test_save_load.py      # save_to_dir → load_from_dir round-trip equality
    test_smoke_train.py    # few-step CPU train: runs end-to-end, total loss decreases
```

Reused verbatim (imported, not copied): `clts.tl_model.build_hooked_transformer`, `util.condensed_tokenizer.CondensedTokenizer`, `util.bio_sampler.BioSampler`. Structural template: `clts/trainCLT.py`, `clts/clt.py`, `clts/evalCLT.py`, `clts/storage.py`.

---

## 5. Model — `SAE_CRL(nn.Module)` (`sae_crl.py`)

Linear, untied encoder/decoder, **no bias** (ablation A.5.3 shows bias doesn't help). Let `x_dim` = activation width (= `d_model` = 384 at `resid_post`), `z_dim` = latent dim (default 3072), `tau` = number of lags (default 5).

**Parameters**
| Param | Shape | Init |
|---|---|---|
| `F_enc` | `[x_dim, z_dim]` | `xavier_normal_` |
| `F_dec` | `[z_dim, x_dim]` | `xavier_normal_` |
| `M` | `[z_dim, z_dim]` | `xavier_normal_` |
| `Bs` | `ParameterList(tau × [z_dim, z_dim])` | **zeros** |

(Memory note: `M` and each `B_τ` are `z_dim²`. At `z_dim=3072`, that's ~9.4M params each → ~56M params total for `τ=5` (~226 MB fp32). Feasible on one GPU; `z_dim` is the main lever if memory is tight.)

**Methods**
- `encode(x) -> z`: `z = x @ F_enc`; if `topk_mode=="latent"` and `topk>0`, apply per-row TopK-by-`|z|` (mask all but top-k). `[*, x_dim] → [*, z_dim]`.
- `decode(z) -> x_hat`: `z @ F_dec`. `[*, z_dim] → [*, x_dim]`.
- `get_M() -> tril(M, diagonal=-1)` (strictly lower-triangular, zero diagonal).
- `predict_latent(Z) -> Ẑ_t`: from a window's encoded latents `Z [batch, tau+1, z_dim]`, `z_t = Z[:, -1]`; `Ẑ_t = z_t @ get_M().T + Σ_{lag=1..tau} Z[:, tau-lag] @ Bs[lag-1].T`. If `topk_mode=="prediction"` and `topk>0`, apply TopK to `Ẑ_t`.
- `forward(window) -> dict`: `window [batch, tau+1, x_dim]`. `Z = encode(window)`; `x_hat_t = decode(Z[:, -1])`; `Ẑ_t = predict_latent(Z)`; `ε̂ = Z[:, -1] − Ẑ_t`. Returns the tensors needed by `compute_loss`.
- `compute_loss(window, alpha, beta, l_spZ=0.0, noise_mode="lap") -> dict`:
  - `recon = mse(x_hat_t, window[:, -1])`
  - `indep = mean(|ε̂|)` if `noise_mode=="lap"` (Laplacian → L1); `trace(cov(ε̂))` if `"gau"`
  - `sparse_B = Σ_{lag} mean(|Bs[lag]|)`, `sparse_M = mean(|get_M()|)`  *(mean reduction so β is `z_dim`-insensitive; the paper writes these as matrix L1 norms)*
  - `sparse_Z = mean(|Ẑ_t|)` (only if `l_spZ>0`)
  - `total = recon + alpha*indep + beta*(sparse_B + sparse_M) + l_spZ*sparse_Z`
  - returns `{total, recon, indep, sparse_B, sparse_M, sparse_Z}` for wandb.
- `aggB() -> [z_dim, z_dim]`: `max_τ |B_τ|` (elementwise max-pool over lags) — the paper's aggregated time-delayed relation matrix; kept now (cheap) for the future analysis tier.
- `save_to_dir(out_dir, model_name)`: safetensors (`F_enc`, `F_dec`, `M`, `B_0..B_{tau-1}`) + `config.yaml` recording `{model_name, hook_name, layer, x_dim, z_dim, tau, topk, topk_mode, noise_mode, eps}`. Mirrors `clt.py`'s on-disk pattern.
- `load_from_dir(in_dir)` (classmethod): infer dims/`tau` from tensor shapes + `config.yaml`.
- Attributes for uniform eval/inference: `.z_dim`, `.hook_name`.

Defaults: `topk=100`, `topk_mode="latent"`, `noise_mode="lap"`.

---

## 6. Data & windowing — `windows.py`

The crux of "respect bio boundaries." One-bio-per-row sidesteps EOS-splitting and guarantees clean boundaries.

- `build_bio_corpus(sampler, tokenizer, n_bios, max_bio_len, seed) -> (tokens, valid_len)`:
  - Sample `n_bios` diverse `(person, exposure)` pairs (mirror `DiverseBioSubset`'s identity+template diversity), render each via `sampler`, encode via `tokenizer`, **EOS-prefix** each bio.
  - Each bio is its **own row**, right-padded with EOS to `max_bio_len` (default 48). Bios longer than `max_bio_len` are truncated.
  - Returns `tokens [n_bios, max_bio_len]` (long) and `valid_len [n_bios]` (real token count per row, excluding right-padding).
- `capture_resid_post(model, tokens_chunk, hook_name) -> acts [chunk_rows, max_bio_len, x_dim]`: `run_with_cache(names_filter=lambda n: n==hook_name, return_type=None)` on a **chunk** of bio rows, kept **per-sequence** (not flattened).
- `window_index(valid_len_chunk, tau) -> list[(local_row, start)]`: for each row, `start ∈ [0, valid_len[r] − (tau+1)]` (stride 1). Rows with `valid_len < tau+1` contribute nothing. **No window straddles a bio.** Windows are gathered as `acts[local_row, start:start+tau+1, :] → [n_windows, tau+1, x_dim]`.
- **Activation streaming (not precompute-all):** training stores only the **token** corpus (`n_bios × max_bio_len` ids — tiny) and re-runs the frozen model **per chunk** to get activations on demand, matching `trainCLT.py`. This lets `n_bios` scale to a real token budget without holding `n_bios × max_bio_len × x_dim` floats in memory. The **held-out eval** corpus is small, so its activations + window index are captured **once** in `setup()`.
- Train corpus uses `seed=0`; held-out eval corpus uses `seed=1` (the repo's train/eval split convention).

---

## 7. Training driver — `trainSAE_CRL.py` (clone of `clts/trainCLT.py`)

- **Module globals** `ARGS, device, model, tokenizer, sampler, eval_acts, eval_index`, populated once by `setup(args)` (reused across wandb sweep trials).
- `setup(args)`: `storage_root()` write-probe (fail fast); `build_hooked_transformer`; `CondensedTokenizer.from_remap_path`; `BioSampler`; build held-out eval corpus + activations + window index (`seed=1`).
- `pick_device()`: cuda → mps → cpu.
- `train_one_run(wandb_config_override=None)`:
  - Read hyperparams from `wandb.config` with `ARGS` fallbacks.
  - Build train **token** corpus (`seed=0`); print bio-length histogram + total bios/windows/tokens; **warn if `τ+1` > median `valid_len`**, **error if total windows == 0** (guardrails).
  - Construct `SAE_CRL`; `Adam(lr, weight_decay=wd)`.
  - Loop over epochs; each epoch shuffle bio order and iterate in **chunks of `chunk_rows` bios**: `capture_resid_post(chunk)` → build within-bio windows → step the optimizer over those windows in sub-batches of `window_batch` (`compute_loss` → `backward` → `step`). Re-running the frozen model per chunk keeps memory flat and scales with `n_bios`. **Sparsity warmup**: linearly ramp `beta`/`alpha` over the first ~10% of steps (mirrors CLT's L0 warmup).
  - `wandb.log` train losses every N steps; held-out eval every M steps.
  - Final: full eval + `save_to_dir(final_dir, model_name)` + `wandb.log({final_eval/*, storage_path})`.
- `build_sweep_config()`: wandb grid (`project="interpLM4"`), candidate axes `z_dim × tau × topk × beta`; metric `final_eval/ce_recovered` (maximize); hyperband early-terminate.
- `_patch_signal_for_worker_threads()`: the SIGINT-off-main-thread shim (needed for `wandb.agent`).
- `parse_args()` + `DEFAULTS` dict; `main()` branches on `--sweep`.

**CLI:** `--model-dir`, `--data-dir`, `--model-name`, `--hook-template "blocks.{layer}.hook_resid_post"`, `--layer`, `--z-dim`, `--tau`, `--topk`, `--topk-mode {latent,prediction}`, `--noise-mode {lap,gau}`, `--alpha`, `--beta`, `--l-spZ`, `--lr`, `--wd`, `--epochs`, `--n-bios`, `--max-bio-len`, `--chunk-rows`, `--window-batch`, `--eps`, `--sweep`.

**Artifacts:** `storage_root()/sae_CRL_runs/<model-name>/{sweep-<id>|standalone}/<trial>/final/` with `<trial> = z{z_dim}_tau{tau}_k{topk}_a{alpha}_b{beta}_lr{lr}_ep{epochs}_n{n_bios}`.

---

## 8. Eval — `evalSAE_CRL.py`

`@torch.no_grad`, given `(model, sae_crl, eval_acts, eval_index, hook_name)`:
- **Reconstruction:** MSE, NMSE, explained variance of `decode(encode(x))` vs `x` on held-out windows (current-step).
- **Latent sparsity:** L0 (mean active latents/token; `= k` when `topk_mode=="latent"`), dead-feature fraction.
- **CE-recovered:** splice `decode(encode(acts))` into the model at `hook_name` via `run_with_hooks`; `ce_recovered = (ce_zero − ce_method)/(ce_zero − ce_clean)`. Same metric/shape as the SAE/CLT eval (directly comparable). Measures the SAE/reconstruction half; the `B_τ`/`M` structure is reported separately.
- **Structure diagnostics:** `sparse_B`, `sparse_M`, independence value `mean(|ε̂|)`, and counts of `|B_τ|`/`|M|` entries above a threshold (sanity, not the full analysis tier).

---

## 9. Storage — `sae_CRL/storage.py`

Mirror `clts/storage.py` (dependency-light). `storage_root()` resolution: (1) `$SAE_CRL_STORAGE_ROOT`; (2) PSC path if it exists; (3) repo-root `sae_CRL_storage/` fallback (Mac). Artifacts under `sae_CRL_runs/`. Add `sae_CRL_storage/` and `sae_CRL/sae_CRL_runs/` to `.gitignore`. (Per project memory: use the resolver, never hardcode `STORAGE_ROOT`.)

---

## 10. Defaults (single source of truth)

| Knob | Default | Source |
|---|---|---|
| base model | `model/grid-L4-H6` (`d_model=384`) | "current" model |
| hook | `blocks.{L}.hook_resid_post`, `L=2` | paper-faithful site |
| `eps` | `1e-5` | TL default / repo convention |
| `z_dim` | 3072 | reference's z=3072 config (~8× of 384) |
| `tau` | 5 | within-bio; paper also ran 5 |
| `max_bio_len` | 48 | per-row pad length |
| `topk` / `topk_mode` | 100 / `latent` | paper App. A.5.1 (TopK on latents) |
| `noise_mode` | `lap` | Eq. 7 / footnote |
| `alpha` (noise) | 0.1 | App. A.5.1 / A.5.3 |
| `beta` (B, M sparsity) | 0.01 | App. A.5.3 |
| `l_spZ` (latent L1) | 0.0 | paper has none (off by default) |
| `lr` / `wd` | 0.01 / 1e-4 | App. A.5.1 (Adam) |
| `n_bios` | 50_000 | streamed per-chunk, so scalable; tune to convergence via loss curves |
| `chunk_rows` / `window_batch` | 512 / 4096 | bios per model-run / windows per optimizer sub-batch |
| `epochs` | 10 | starting point; tune via loss curves |

---

## 11. Error handling & guardrails

- `setup()` write-probes `storage_root()` and aborts early if unwritable.
- `build_bio_corpus` prints a **bio-length histogram** and total bios / windows / activation-tokens; `train_one_run` **warns** if `τ+1` exceeds the median `valid_len` (the τ-vs-short-bio trap), and **errors** if the total window count is 0.
- `--hook-template` must contain `{layer}`.
- `CondensedTokenizer.encode` raises on out-of-vocab tokens; corpus build surfaces that clearly.
- Device-agnostic (`pick_device`); fp32 end-to-end.

---

## 12. Testing strategy

- `test_windows.py`: construct a tiny corpus with known `valid_len`; assert no produced window contains a position `≥ valid_len[b]` and none crosses into another row; assert window count `= Σ_b max(0, valid_len[b] − tau)`.
- `test_model.py`: forward on random input asserts shapes; `get_M()` is strictly lower-triangular (zero diagonal); each loss component is a finite scalar; TopK in `latent` mode yields exactly `k` nonzero latents/row.
- `test_save_load.py`: `save_to_dir → load_from_dir` reproduces all parameters and config.
- `test_smoke_train.py`: 20–50 steps on CPU with tiny dims; assert it runs and `total` loss at the end < at the start.

---

## 13. Out of scope this round (clean drop-in later)

- **Relation analysis / dashboards:** rank top `aggB` / `M` entries into interpretable concept relations; per-feature dashboards (would mirror `saes/sae_explorer.py` + `clts/gen_feature_dashboards.py`). The model already exposes `aggB()` and stores `B_τ`/`M`, so this tier attaches without model changes.
- **Synthetic identifiability validation** (MCC/SHD against ground-truth `A`/`B`/`M`) — the natural confidence check before trusting real activations.
- **SAEBench-style quantitative comparison** vs the existing SAE/CLT runs.
- **circuit-tracer attribution** wiring.

---

## 14. References

- Song, Sun, Li, Zheng, Zhang. *LLM Interpretability with Identifiable Temporal-Instantaneous Representation.* NeurIPS 2025. arXiv:2509.23323v2.
- Reference code: `github.com/xiangchensong/temp-inst-sae` (`examples/linear_idol_model.py`, `examples/main.py`, `synthetic/complete-3.py`).
- Repo templates: `clts/{clt,trainCLT,evalCLT,storage,tl_model}.py`, `saes/{trainSAE,evalSAE}.py`, `util/{condensed_tokenizer,bio_sampler,diverse_subset}.py`.
