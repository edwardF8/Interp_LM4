# SAE-CRL: Causal-Representation-Learning SAE for Interp_LM4 (faithful port)

- **Created:** 2026-06-01 · **Revised:** 2026-06-03 (rewritten from the from-paper draft to a faithful **port** of the reference code)
- **Status:** Design approved. Implementation plan: `docs/superpowers/plans/2026-06-01-sae-CRL-framework.md`.
- **Scope this round:** Core only — trainable model + windowing + training driver + eval metrics + checkpoint save/load. Relation analysis/dashboards, synthetic identifiability validation, SAEBench comparison, and attribution wiring are **out of scope** (§9).

---

## 1. Goal

Port the temporal–instantaneous SAE method from Song et al., "LLM Interpretability with Identifiable Temporal-Instantaneous Representation" (NeurIPS 2025; arXiv:2509.23323v2) into `Interp_LM4`, trained on this repo's small Llama checkpoints, condensed tokenizer, and bioS data. Produce trained time-delayed matrices `B_τ` and an instantaneous matrix `M` over learned concept latents.

**Faithfulness rule (decided):** this is a **port of the authors' released code**, not a from-scratch reimplementation. Default every behavior to the reference code; deviate only where (a) the user explicitly chose the paper over the code, or (b) our setup forces it. **Every deviation is enumerated in §3 and must stay documented in the code + plan.**

**Reference (source of truth):** cloned locally at `reference/temp-inst-sae/` (gitignored). The ported model is `examples/linear_idol_model.py` (`LinearIDOL`); training/defaults are `examples/main.py`; their window builder is `examples/utils.py`.

**Method in one paragraph.** A bias-free, untied **linear** autoencoder encodes activations into latents and reconstructs them. The latents carry a causal SEM: an instantaneous matrix `M` (within a token) and per-lag time-delayed matrices `B_1..B_τ` (from earlier tokens). Training minimizes the reference's six losses: reconstruction MSE (on the current token), the SEM residual independence penalty `ε̂ = z_t − (M z_t + Σ_τ B_τ z_{t−τ})`, and L1 sparsity on `B`/`M`, with **TopK** enforcing latent sparsity.

---

## 2. Inputs — the integration seam (reused, not reinvented)

| Input | Source | Notes |
|---|---|---|
| Base model | `clts.tl_model.build_hooked_transformer(model_dir, device)` | HF Llama → TransformerLens `HookedTransformer`. Default `model/grid-L4-H6` (4 layers, `d_model=384`, vocab 1836). |
| Tokenizer | `util.condensed_tokenizer.CondensedTokenizer.from_remap_path(data_dir/"old_to_new.json")` | reduced GPT-2 vocab, size 1836, `eos=bos=pad=1835`. |
| Data | `util.bio_sampler.BioSampler(data_dir/"people.json", fields=("birthday",))` | renders byte-identical bioS bios; `sampler.sample(rng)["text"]`. |
| Activations | `model.run_with_cache(tokens, names_filter=lambda n: n==hook)` at `blocks.{L}.hook_resid_post` | default layer `L=2`. RMS-norm `eps=1e-5` (TL default); `--eps 1e-6` to match the checkpoint exactly. |

---

## 3. Deviations ledger (authoritative)

**Matches the reference code exactly:** bias-free untied `F_enc`/`F_dec`; `Bs` zero-init, `F_enc`/`F_dec`/`M` xavier; `forward` returns the 6-tuple `(mse_Xt, mse_Zt, indep, sparse_Bs, sparse_M, sparse_Zt)`; **reconstruction MSE on the last window position only**; `Zt = M·z_t + Σ_lag B_lag·z_{t−lag}`; `ε̂ = z_t − Zt`; `lap = mean|ε̂|`, `gau = trace(cov(ε̂))`; loss `= mse_Xt + l_mse_Zt·mse_Zt + l_ind·indep + l_spB·sparse_Bs + l_spM·sparse_M + l_spZ·sparse_Zt` (`main.py:113`); defaults `lr=0.01, wd=1e-4, z_dim=3072, l_ind=0.1, topk=100`, `w`/`mse_Zt` off; **feature-first** window layout `[batch, x_dim, τ+1]`, last index = current token.

**Deliberate deviations toward the PAPER (user's call):**
| # | We do (paper) | Their code does | Why |
|---|---|---|---|
| P1 | `M = tril(M, −1)` (strictly lower, zero diagonal) | `tril(M, +1)` | Paper §3/§4.3: the instantaneous graph is an acyclic DAG (no self-loops). |
| P2 | **TopK on the encoded latents** `Zp` (all timesteps), **kept on at eval** | TopK on the predicted `Zt`, disabled at eval | Paper App. A.5.1 ("sparsity in the hidden feature activations") + cited method [8]=BatchTopK (TopK on encoder outputs). Makes it a genuine sparse SAE so recon/L0/CE-recovered are meaningful. |
| P3 | **`l_spZ = 0`** (no L1 on latents/`Zt`) | `l_spZ = 0.1` | Paper Eq. 8 penalizes only `B_τ` and `M`; latent sparsity is left to TopK (P2). Configurable: `--l-spZ 0.1` restores the code's term. |
| P4 | **`l_spB = l_spM = 0.01`** (paper's tuned β) | `0.1` | Paper Eq. 9 + sensitivity study (Table 6) select β=0.01 for `B`/`M` sparsity; their code default is `0.1`. Configurable: `--l-spB`/`--l-spM`. |

**Deviations forced by our setup:**
| # | We do | Their code does | Why |
|---|---|---|---|
| S1 | activations via TransformerLens `run_with_cache` | nnsight + `dictionary_learning.ActivationBuffer` | we use TL `HookedTransformer`; no nnsight stack. |
| S2 | **span-the-bio** windows: within a bio, one zero-padded window per token position; bios processed in parallel, never concatenated | slide over a continuous, boundary-free stream | bios are short, independent docs. **Merging ruled out** (mentor confirmed: bios in parallel, no cross-bio windows). |
| S3 | `τ = longest bio − 1` (auto) | fixed `τ=20` | bios are ~12–20 tokens; a 21-wide window wouldn't fit. |
| S4 | fixed-step training loop | token-budget `while n_tokens < total` | no streaming buffer. |
| S5 | additive `save_to_dir`/`load_from_dir` (safetensors+yaml) + `aggB()` | `torch.save(state_dict)` only | downstream analysis; no effect on training math. |
| S6 | `noise_mode` default `lap` | argparse default `lap` | unchanged (matches). |

---

## 4. Package layout — `sae_CRL/`

```
sae_CRL/
  __init__.py          # empty package marker
  storage.py           # storage_root() resolver (env SAE_CRL_STORAGE_ROOT -> PSC -> repo-root sae_CRL_storage/)
  sae_crl.py           # SAE_CRL(nn.Module) + topk_latents — the LinearIDOL port
  windows.py           # span_windows / windows_for_batch / pack_bios / build_bio_corpus
  evalSAE_CRL.py       # capture_resid_post + recon_metrics + structure_metrics + ce_recovered
  trainSAE_CRL.py      # driver: derive_tau, train_step, setup, train_one_run, sweep
  tests/               # storage, model, windows, train
```
Reused verbatim: `clts.tl_model.build_hooked_transformer`, `util.condensed_tokenizer.CondensedTokenizer`, `util.bio_sampler.BioSampler`. Artifacts: `storage_root()/sae_CRL_runs/<model-name>/{sweep-<id>|standalone}/<trial>/final/`.

---

## 5. Model — `SAE_CRL(nn.Module)` (`sae_crl.py`)

Port of `LinearIDOL`. `x_dim` = `d_model` (384 at resid_post), `z_dim` = latent dim (3072), `tau` = lags.

**Parameters:** `F_enc [x_dim,z_dim]`, `F_dec [z_dim,x_dim]`, `M [z_dim,z_dim]` (all xavier); `Bs = ParameterList(τ × [z_dim,z_dim])` (**zeros**). No bias.

**`forward(Xp) -> 6-tuple`**, `Xp [batch, x_dim, τ+1]` (feature-first, last column = current token):
1. `Zp = einsum('hd,bdt->bht', F_enc.T, Xp)` → `[batch, z_dim, τ+1]`.
2. **P2:** if `topk>0`, `Zp = topk_latents(Zp, topk)` (keep top-k by `|·|` over `z_dim` per `(batch, t)`; applied to all timesteps; **not** disabled at eval).
3. `recons = einsum('dh,bht->bdt', F_dec.T, Zp)`; `loss_mse_Xt = mse(recons[:,:,-1], Xp[:,:,-1])` (last position only).
4. **P1:** `M = tril(self.M, -1)`. `Zt = M·Zp[:,:,τ] + Σ_{lag=1..τ} B_lag·Zp[:,:,τ-lag]`; accumulate `loss_sparse_Bs = Σ_lag l1(B_lag)`.
5. `loss_mse_Zt = mse(Zt, Zp[:,:,τ])`; `ε̂ = Zp[:,:,τ] − Zt`; `loss_indep = trace(cov(ε̂))` (`gau`) or `mean|ε̂|` (`lap`); `loss_sparse_M = l1(M)`; `loss_sparse_Zt = l1(Zt)`.

**Additive (S5):** `aggB()` = `max_τ |B_τ|` → `[z_dim,z_dim]`; `save_to_dir(out_dir, model_name, hook_name, layer)` (safetensors `{F_enc,F_dec,M,B_0..B_{τ-1}}` + `config.yaml`); `load_from_dir`.

---

## 6. Windowing — `windows.py` (span-the-bio, S2)

- `build_bio_corpus(sampler, tokenizer, n_bios, max_bio_len, seed)` → one bio per row, EOS-prefixed, right-padded → `tokens [n_bios, max_bio_len]`, `valid_len [n_bios]`.
- `span_windows(acts, valid_len, tau)`: for **every** token position `t ∈ [0, valid_len)`, build the window `acts[t−τ .. t]`, **zero-left-padded** before the bio start, transposed to feature-first `[x_dim, τ+1]` (current token = last column). → `[valid_len, x_dim, τ+1]`. So a bio of length `L` yields `L` windows; the first tokens have mostly-zero lookback; **no window crosses a bio boundary**.
- `windows_for_batch(acts, valid_lens, tau)`: concatenate `span_windows` over a batch of bios → `[total_windows, x_dim, τ+1]`.
- `τ = derive_tau(valid_len, "auto", tau_cap) = max(valid_len) − 1` (S3), optionally capped.

---

## 7. Training — `trainSAE_CRL.py` (mirrors `main.py`'s per-step loss assembly)

- Module globals + `setup(args)`: `storage_root()` write-probe; `build_hooked_transformer`; `CondensedTokenizer`; `BioSampler`; capture a small held-out eval corpus (`seed+1`).
- `derive_tau`, `train_step(sae, opt, windows, l_ind, l_spB, l_spM, l_spZ, l_mse_Zt)` → assembles `loss = mse_Xt + l_mse_Zt·mse_Zt + l_ind·indep + l_spB·sp_B + l_spM·sp_M + l_spZ·sp_Zt`, backward, Adam step.
- `train_one_run`: build train token corpus (`seed=0`); derive `τ`; build `SAE_CRL`; loop epochs → batches of `batch_bios` → `capture_resid_post` → `windows_for_batch` → SGD over windows in sub-batches of `batch_windows`. Periodic + final eval; `save_to_dir`; wandb logging (`project="interpLM4"`).
- `build_sweep_config` (z_dim × topk), `_patch_signal_for_worker_threads`, `parse_args`, `main` (branches on `--sweep`).

---

## 8. Eval — `evalSAE_CRL.py`

- `recon_metrics`: reconstruction MSE / explained variance on the **last** window position; current-token **L0** (= `topk` under P2).
- `structure_metrics`: `B_τ`/`M` L1 sparsity + above-threshold relation counts.
- `ce_recovered`: splice the sparse reconstruction `decode(topk(encode(resid_post)))` at the hook via `run_with_hooks`; `(ce_zero − ce_sae)/(ce_zero − ce_orig)`. Comparable to the SAE/CLT eval.

---

## 9. Defaults & out-of-scope

**Defaults** (single source of truth: `DEFAULTS` in `trainSAE_CRL.py`): model `grid-L4-H6`; hook `blocks.{L}.hook_resid_post`, `L=2`; `z_dim=3072`; `τ=auto` (longest bio−1) with `--tau-cap`; `topk=100`; `noise_mode=lap`; `l_ind=0.1`, **`l_spB=l_spM=0.01` (P4)**, **`l_spZ=0` (P3)**, `mse_Zt=off`, `w=off`; `lr=0.01`, `wd=1e-4`; `epochs=10`; `n_bios=50_000`; `max_bio_len=48`; `batch_bios=256`; `batch_windows=4096`; `eps=1e-5`.

**Memory note:** `B` holds `τ` matrices of `z_dim²`. With τ auto-spanning the largest bio (~25–30) at `z_dim=3072`, that's ~250M params (~1 GB fp32, ~3× with Adam). Fine on A100/L40; on a memory-constrained Mac drop `z_dim` (e.g. 1024–1536) or set `--tau-cap`.

**Out of scope this round:** relation-extraction dashboards (the model already exposes `aggB()` + stores `B`/`M`), synthetic MCC/identifiability validation, SAEBench comparison, circuit-tracer attribution.

---

## 10. References

- Song et al., NeurIPS 2025, arXiv:2509.23323v2.
- Reference code (cloned, source of truth): `reference/temp-inst-sae/examples/{linear_idol_model.py,main.py,utils.py}`.
- Implementation plan (complete code, TDD): `docs/superpowers/plans/2026-06-01-sae-CRL-framework.md`.
- Repo seam/templates: `clts/{tl_model,trainCLT,evalCLT,clt,storage}.py`, `util/{condensed_tokenizer,bio_sampler}.py`.
