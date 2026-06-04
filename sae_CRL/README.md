# SAE-CRL pipeline

Causal-representation SAE for `grid-L*-H*` Llama models trained on bioS data. A faithful
port of the temp-inst-sae **LinearIDOL** method (`reference/temp-inst-sae/`): it learns a
bias-free dictionary over one residual-stream site **and** a causal graph among the learned
concepts — an instantaneous DAG `M` plus per-lag time-delayed matrices `B_1..B_τ`.

- **Design contract / deviations ledger:** [docs/superpowers/plans/2026-06-01-sae-CRL-framework.md](../docs/superpowers/plans/2026-06-01-sae-CRL-framework.md)
- **Spec:** [docs/superpowers/specs/2026-06-01-sae-crl-framework-design.md](../docs/superpowers/specs/2026-06-01-sae-crl-framework-design.md)

---

## What it computes

For a chosen residual hook `blocks.<layer>.hook_resid_post` (dimension `d_model`):

```
acts x_t  ──F_enc──▶  z_t (latents, TopK-sparse)  ──F_dec──▶  x̂_t   (reconstruction)
                                │
                                ├─ instantaneous:  M · z_t           (M = tril, strictly lower → acyclic DAG)
                                └─ time-delayed:   Σ_lag B_lag · z_{t-lag}
                          predicted latent  Zt = M·z_t + Σ B_lag·z_{t-lag}
                          residual          Et = z_t − Zt   (driven toward independent noise)
```

`forward` returns the 6-tuple `(mse_Xt, mse_Zt, indep, sparse_Bs, sparse_M, sparse_Zt)`;
the training loss is `mse_Xt + l_mse_Zt·mse_Zt + l_ind·indep + l_spB·sparse_Bs + l_spM·sparse_M + l_spZ·sparse_Zt`.

**Key design facts**
- **Single-site / per-layer.** One model = one residual hook (`x_dim = d_model`). "CRL on N layers" means N independent models. It is *not* a cross-layer model (that's the CLT).
- **`x_dim` auto-tracks the model** (`= model.cfg.d_model`); `tau` auto-tracks the **data** (`= longest bio − 1`). Neither is hand-set.
- **The dataset must be the one the model was trained on** — activations are captured by running the model on its bios, with that dataset's tokenizer.

### Deviations from the reference (summary — full table in the plan)
| | what we do | why |
|---|---|---|
| P1 | `M = tril(M, −1)` (strictly lower, zero diagonal) | acyclic instantaneous DAG (paper §3/§4.3) |
| P2 | TopK on the **encoded latents**, kept on at eval | genuine sparse SAE; recon/L0/CE-recovered meaningful |
| P3 | `l_spZ = 0` (no L1 on latents) | sparsity left to TopK (paper Eq. 8) |
| P4 | `l_spB = l_spM = 0.01` | paper's tuned β (Table 6) |
| S1–S6 | TL `run_with_cache`; span-the-bio windows; auto-τ; fixed-step loop; safetensors+yaml save; `lap` noise | forced by this repo's setup |

---

## Components

| File | Responsibility |
|---|---|
| [sae_crl.py](sae_crl.py) | `SAE_CRL(nn.Module)` — params, `forward`+losses, `aggB`, `save_to_dir`/`load_from_dir`; `topk_latents` |
| [windows.py](windows.py) | span-the-bio windows (`span_windows`, `windows_for_batch`) + corpus (`pack_bios`, `build_bio_corpus`) |
| [evalSAE_CRL.py](evalSAE_CRL.py) | `capture_resid_post`, `recon_metrics`, `structure_metrics`, `ce_recovered` |
| [trainSAE_CRL.py](trainSAE_CRL.py) | driver: `derive_tau`, `train_step`, `setup`, `train_one_run`, `build_sweep_config`, `main` |
| [storage.py](storage.py) | `storage_root()` resolver |
| [tests/](tests/) | unit tests (17) |

---

## Quickstart

Run everything in the parent venv (the one that runs `clts/trainCLT.py` — has `torch`,
`transformer_lens`, `safetensors`, `pyyaml`, `wandb`).

**Unit tests**
```bash
../.venv/bin/python -m pytest sae_CRL/tests/ -v        # 17 tests
```

**One standalone run (local smoke; tiny knobs)**
```bash
WANDB_MODE=offline python sae_CRL/trainSAE_CRL.py \
  --model-dir model/grid-L4-H6 --data-dir data/bioS_N-Bd_final_grid \
  --z-dim 256 --topk 16 --n-bios 64 --epochs 1 --batch-bios 16 --layer 2
```

**A sweep, locally** (β × topk grid; needs online wandb)
```bash
python sae_CRL/trainSAE_CRL.py --sweep \
  --model-dir model/grid-L4-H6 --data-dir data/bioS_N-Bd_final_grid --layer 2 \
  --sweep-beta 0.001,0.01,0.1 --sweep-topk 25,100
```

---

## Running sweeps on PSC

Two files: an **orchestrator** you run by hand, and the **SLURM job** it submits.

- [scripts/sweeps_crl.sh](../scripts/sweeps_crl.sh) — submit / monitor one parallel job per layer.
- [scripts/sweep_crl_psc.sbatch](../scripts/sweep_crl_psc.sbatch) — the GPU job (env setup, preflight, the sweep).

```bash
wandb login                          # once, on the login node
./scripts/sweeps_crl.sh submit       # one job per layer in $LAYERS
./scripts/sweeps_crl.sh watch        # status table: state / elapsed / trials (n/N) per layer
./scripts/sweeps_crl.sh cancel
```

**Everything tunable lives in one `EDIT HERE` block** at the top of `sweeps_crl.sh`:

| knob | default | meaning |
|---|---|---|
| `MODEL_NAME` | `grid-L4-H6` | model dir name (under `$GRID` in the .sbatch) |
| `DATASET` | `bioS_N-Bd_final_grid` | **must** be the dataset the model was trained on (preflight enforces this) |
| `LAYERS` | `0 1` | space list; 0-indexed `blocks.<n>.hook_resid_post` |
| `BETAS` | `0.001,0.01,0.1` | sweep axis — `l_spB = l_spM` (graph sparsity) |
| `TOPKS` | `25,100` | sweep axis — latent L0 |
| `Z_DIM` | `3072` | dictionary size (8× d_model=384) |
| `EPOCHS` / `N_BIOS` / `LR` / `NOISE_MODE` | `10 / 50000 / 0.01 / lap` | fixed training knobs |
| `TIME` | `12:00:00` | walltime per layer-job |

Trials/layer = `|BETAS| × |TOPKS|` (auto-computed). One layer-job registers one wandb sweep
`sae_CRL_sweep_<model>_L<layer>`; override per-run, e.g. `BETAS="0.01" TOPKS="50" LAYERS="0 1 2" ./scripts/sweeps_crl.sh submit`.

---

## Outputs

`storage_root()` resolves to `$SAE_CRL_STORAGE_ROOT`, else PSC `/jet/home/friedmae/data_storage/LM4_Results`,
else repo-root `sae_CRL_storage/` (gitignored). Each trial saves:

```
<storage_root>/sae_CRL_runs/<model_name>/<sweep-<id> | standalone>/L<layer>_z<z>_tau<τ>_k<topk>_lr<lr>_ep<ep>_n<n>[_b<beta>]/final/
    ├── sae_crl.safetensors   # F_enc, F_dec, M, B_0..B_{τ-1}
    └── config.yaml           # x_dim, z_dim, tau, w, noise_mode, topk_sparsity, hook_name, layer, model_name
```

Reload with `SAE_CRL.load_from_dir(path)` (returns a CPU model — `.to(device)` before use).
`aggB()` gives a lag-agnostic `[z_dim, z_dim]` edge map (max |B_lag| over lags).

---

## wandb metrics

**`train/*`** (every 30 steps): `loss`, `mse_Xt` (recon), `indep` (noise term), `sp_B`, `sp_M`, `sp_Zt`.

**`eval/*`** (every epoch, held-out bios) and **`final_eval/*`** (once, in run summary):

| metric | meaning |
|---|---|
| `recon_mse`, `explained_var` | reconstruction quality at the current token |
| `l0` | avg active latents/token (sanity: ≈ `topk`) |
| `sparse_M`, `n_M_above` | magnitude & **edge count of the instantaneous DAG `M`** |
| `sparse_B`, `n_B_above` | magnitude & **edge count of the time-delayed `B_τ`** |
| `ce_recovered` (+ `ce_orig`/`ce_sae`/`ce_zero`) | next-token CE preserved when the sparse SAE recon replaces resid_post |

The sweep optimizes **`final_eval/ce_recovered`** (maximize). Reading a β sweep as "different
causal interpretations": as **β rises**, `n_M_above`/`n_B_above` fall (sparser, more readable
graph) while `ce_recovered`/`explained_var` may drop — look for the β that keeps recovery high
while `M`/`B` stay sparse. `ce_recovered` can read >1 or <0 on tiny/underfit runs; compare
*relative* ordering across trials.

---

## Key CLI flags (`trainSAE_CRL.py`)

`--model-dir` `--data-dir` (required) · `--layer` · `--hook-template` (default `blocks.{layer}.hook_resid_post`)
· `--z-dim` · `--topk` · `--tau`/`--tau-cap` (default `auto`) · `--noise-mode {lap,gau}`
· `--l-ind` `--l-spB` `--l-spM` `--l-spZ` · `--mse-Zt` · `--lr` `--wd` `--epochs` `--n-bios`
· `--max-bio-len` `--batch-bios` `--batch-windows` · `--eps` · `--sweep` `--sweep-beta` `--sweep-topk`.
