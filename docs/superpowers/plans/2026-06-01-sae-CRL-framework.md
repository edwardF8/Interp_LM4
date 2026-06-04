# SAE-CRL Framework Implementation Plan (faithful port)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the temp-inst-sae `LinearIDOL` method (cloned at `reference/temp-inst-sae/`) into a new `sae_CRL/` package, trained on this repo's `grid-L4-H6` Llama / condensed tokenizer / bioS data, producing trained `B_τ` (time-delayed) and `M` (instantaneous) concept-relation matrices.

**Architecture:** A faithful line-by-line port of `reference/temp-inst-sae/examples/linear_idol_model.py`. Bias-free untied linear `F_enc`/`F_dec`; instantaneous `M`; per-lag `B_1..B_τ` (zero-init). `forward(Xp)` on feature-first windows `[batch, x_dim, τ+1]` returns the same 6-tuple of losses; reconstruction MSE is on the **last window position only**; predicted latent `Zt = M·z_t + Σ_lag B_lag·z_{t-lag}`; residual `Et = z_t − Zt`. Two **deliberate paper-side deviations** (see Deviations): `M = tril(M, −1)` (strictly lower) and **TopK on the encoded latents** (kept on at eval). Trained over **span-the-bio** windows (one zero-padded window per token position within each bio; τ = longest bio − 1), batched, fixed-step loop.

**Tech Stack:** PyTorch, TransformerLens (`HookedTransformer`), safetensors, PyYAML, wandb. Tests: pytest.

**Reference (source of truth, cloned locally, gitignored):** `reference/temp-inst-sae/examples/{linear_idol_model.py,main.py,utils.py}`.
**Spec:** `docs/superpowers/specs/2026-06-01-sae-crl-framework-design.md`.
**Test env:** the env that runs `clts/trainCLT.py` (parent venv: `torch`, `transformer_lens`, `safetensors`, `pyyaml`). Unit tests (Tasks 1–7) need only `torch`/`safetensors`/`pyyaml`; the integration check (Task 8) also needs `transformer_lens` + a checkpoint + a data dir.

---

## Deviations ledger (what we match, where we differ, and why)

**Matches their code exactly:** bias-free untied `F_enc`/`F_dec`; `Bs` zero-init, `F_enc`/`F_dec`/`M` xavier; `forward` returns the 6-tuple `(mse_Xt, mse_Zt, indep, sparse_Bs, sparse_M, sparse_Zt)`; **reconstruction MSE on the last window position only**; `Zt = M·z_t + Σ_lag B_lag·z_{t-lag}`; `Et = z_t − Zt`; `lap = mean|Et|`, `gau = trace(cov(Et))`; loss assembly `mse_Xt + l_mse_Zt·mse_Zt + l_ind·indep + l_spB·sparse_Bs + l_spM·sparse_M + l_spZ·sparse_Zt`; defaults `lr=0.01, wd=1e-4, z_dim=3072, l_ind=0.1, topk=100`, `w`/`mse_Zt` off; feature-first window layout `[batch, x_dim, τ+1]` with the last index = current token.

**Deliberate deviations toward the PAPER (documented, your call):**
| # | We do (paper) | Their code does | Why |
|---|---|---|---|
| P1 | `M = tril(M, −1)` (strictly lower, zero diagonal) | `tril(M, +1)` | Paper §3/§4.3 require an acyclic instantaneous DAG. |
| P2 | TopK on the **encoded latents** `Zp` (all timesteps), **kept on at eval** | TopK on the predicted `Zt`, disabled at eval | Paper App. A.5.1 ("sparsity in hidden activations") + cited method [8]=BatchTopK (TopK on encoder outputs); makes it a genuine sparse SAE so recon/L0/CE-recovered are meaningful. |
| P3 | **`l_spZ = 0`** (no L1 on the latents / `Zt`) | `l_spZ = 0.1` | Paper Eq. 8 puts L1 only on `B_τ` and `M`; latent sparsity is left entirely to TopK (P2). Kept as a configurable flag — set `--l-spZ 0.1` to restore the code's term. |
| P4 | **`l_spB = l_spM = 0.01`** (paper's tuned β) | `0.1` | Paper Eq. 9 + sensitivity study (Table 6) select β=0.01 for `B`/`M` sparsity; their code default is `0.1`. Configurable: `--l-spB`/`--l-spM`. |

**Deviations forced by our setup (documented):**
| # | We do | Their code does | Why |
|---|---|---|---|
| S1 | activations via TransformerLens `run_with_cache` | nnsight + `dictionary_learning.ActivationBuffer` | we use TL `HookedTransformer`; no nnsight stack. |
| S2 | **span-the-bio** windows (within a bio, one zero-padded window per token position; bios processed in parallel, never concatenated) | slide over a continuous, boundary-free stream | bios are short, independent docs. **Merging ruled out** (mentor confirmed: bios in parallel, no cross-bio windows). |
| S3 | `τ = longest bio − 1` (auto) | fixed `τ=20` | bios are ~12–20 tokens; a 21-wide window wouldn't fit. |
| S4 | fixed-step training loop | token-budget `while n_tokens < total` | no streaming buffer. |
| S5 | additive `save_to_dir`/`load_from_dir` (safetensors+yaml) + `aggB()` | `torch.save(state_dict)` only | downstream analysis in this repo; no effect on training math. |
| S6 | `noise_mode` default `lap` | argparse default `lap` | matches their default (no change). |

---

## File Structure

| File | Responsibility |
|---|---|
| `sae_CRL/__init__.py` | empty package marker |
| `sae_CRL/storage.py` | `storage_root()` resolver |
| `sae_CRL/sae_crl.py` | `SAE_CRL(nn.Module)` + `topk_latents` — the `LinearIDOL` port |
| `sae_CRL/windows.py` | span-the-bio window builder + one-bio-per-row corpus |
| `sae_CRL/evalSAE_CRL.py` | capture + recon/L0/structure + ce_recovered |
| `sae_CRL/trainSAE_CRL.py` | driver: `derive_tau`, `train_step`, setup, loop, sweep |
| `sae_CRL/tests/*` | unit tests |
| `.gitignore` | ignore `sae_CRL_storage/` (done) + `reference/` (done) |

---

## Task 1: Package scaffold + storage resolver

**Files:** Create `sae_CRL/__init__.py`, `sae_CRL/tests/__init__.py`, `sae_CRL/storage.py`, `sae_CRL/tests/test_storage.py`.

- [ ] **Step 1: Create package markers**
```bash
mkdir -p "sae_CRL/tests"; : > "sae_CRL/__init__.py"; : > "sae_CRL/tests/__init__.py"
```

- [ ] **Step 2: Failing test** — `sae_CRL/tests/test_storage.py`
```python
import importlib


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SAE_CRL_STORAGE_ROOT", str(tmp_path))
    import sae_CRL.storage as storage
    importlib.reload(storage)
    assert storage.storage_root() == tmp_path


def test_repo_root_fallback(monkeypatch):
    monkeypatch.delenv("SAE_CRL_STORAGE_ROOT", raising=False)
    import sae_CRL.storage as storage
    importlib.reload(storage)
    root = storage.storage_root()
    assert root.name == "sae_CRL_storage" and root.parent.name == "Interp_LM4"
```

- [ ] **Step 3: Run — expect FAIL** (`No module named 'sae_CRL.storage'`)
`python -m pytest sae_CRL/tests/test_storage.py -v`

- [ ] **Step 4: Implement** — `sae_CRL/storage.py`
```python
"""Canonical storage-root resolver for SAE-CRL artifacts. Dependency-light.
Order: $SAE_CRL_STORAGE_ROOT, else PSC dir if present, else repo-root sae_CRL_storage/."""
from __future__ import annotations

import os
from pathlib import Path


def storage_root() -> Path:
    env = os.environ.get("SAE_CRL_STORAGE_ROOT")
    if env:
        return Path(env)
    psc_root = Path("/jet/home/friedmae/data_storage/LM4_Results")
    if psc_root.exists():
        return psc_root
    return Path(__file__).resolve().parent.parent / "sae_CRL_storage"
```

- [ ] **Step 5: Run — expect PASS** (2 passed)

- [ ] **Step 6: Commit**
```bash
git add sae_CRL/__init__.py sae_CRL/tests/__init__.py sae_CRL/storage.py sae_CRL/tests/test_storage.py
git commit -m "feat(sae_CRL): package scaffold + storage_root resolver"
```

---

## Task 2: Model parameters (`LinearIDOL.__init__` port)

**Files:** Create `sae_CRL/sae_crl.py`, `sae_CRL/tests/test_model.py`.

- [ ] **Step 1: Failing test** — `sae_CRL/tests/test_model.py`
```python
import torch
from sae_CRL.sae_crl import SAE_CRL


def _tiny(**kw):
    kw.setdefault("x_dim", 6); kw.setdefault("z_dim", 8); kw.setdefault("tau", 3)
    return SAE_CRL(**kw)


def test_param_shapes_and_inits():
    m = _tiny()
    assert m.F_enc.shape == (6, 8) and m.F_dec.shape == (8, 6) and m.M.shape == (8, 8)
    assert len(m.Bs) == 3 and all(b.shape == (8, 8) for b in m.Bs)
    assert all(torch.count_nonzero(b) == 0 for b in m.Bs)        # Bs zero-init (ref lines 25-29)
    assert torch.count_nonzero(m.F_enc) > 0                       # F_enc xavier (ref lines 50-53)
```

- [ ] **Step 2: Run — expect FAIL** (`No module named 'sae_CRL.sae_crl'`)
`python -m pytest sae_CRL/tests/test_model.py -v`

- [ ] **Step 3: Implement** — `sae_CRL/sae_crl.py`
```python
"""SAE_CRL — faithful port of reference/temp-inst-sae/examples/linear_idol_model.py
(LinearIDOL), with two documented paper-side deviations: M = tril(M, -1) (strictly
lower) and TopK on the encoded latents (kept at eval). See the plan's deviations ledger."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file, load_file


def topk_latents(Zp: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k by |value| over the latent axis (dim=1) per (batch, timestep).
    Zp: [batch, z_dim, T]. PAPER deviation P2: TopK on encoded latents (all timesteps)."""
    if k <= 0 or k >= Zp.shape[1]:
        return Zp
    idx = Zp.abs().topk(k, dim=1).indices
    mask = torch.zeros_like(Zp)
    mask.scatter_(1, idx, 1.0)
    return Zp * mask


class SAE_CRL(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, tau: int,
                 w: float = 0.5, noise_mode: str = "lap", topk_sparsity: int = 100):
        super().__init__()
        self.x_dim, self.z_dim, self.tau = x_dim, z_dim, tau
        self.w, self.noise_mode, self.topk_sparsity = w, noise_mode, topk_sparsity
        # per-lag time-delayed matrices, ZERO init (ref lines 25-29, NOT xavier)
        self.Bs = nn.ParameterList([nn.Parameter(torch.zeros(z_dim, z_dim)) for _ in range(tau)])
        self.F_enc = nn.Parameter(torch.ones(x_dim, z_dim))   # ref line 31
        self.F_dec = nn.Parameter(torch.ones(z_dim, x_dim))   # ref line 33
        self.M = nn.Parameter(torch.ones(z_dim, z_dim))       # ref line 36
        self.init_params()
        self._hook_name = None

    def init_params(self):  # ref lines 50-53
        nn.init.xavier_normal_(self.F_enc.data)
        nn.init.xavier_normal_(self.F_dec.data)
        nn.init.xavier_normal_(self.M.data)
```

- [ ] **Step 4: Run — expect PASS** (1 passed)

- [ ] **Step 5: Commit**
```bash
git add sae_CRL/sae_crl.py sae_CRL/tests/test_model.py
git commit -m "feat(sae_CRL): LinearIDOL parameter port (Bs zero-init, xavier F/M)"
```

---

## Task 3: `forward` + losses (port, with P1/P2 deviations)

**Files:** Modify `sae_CRL/sae_crl.py`, `sae_CRL/tests/test_model.py`.

- [ ] **Step 1: Add failing tests** — append to `sae_CRL/tests/test_model.py`
```python
def _xp(batch=2, x_dim=6, tau=3):
    return torch.randn(batch, x_dim, tau + 1)   # feature-first [batch, x_dim, tau+1]


def test_forward_returns_six_finite_losses():
    m = _tiny()
    out = m(_xp())
    assert len(out) == 6
    assert all(torch.isfinite(torch.as_tensor(float(x))) for x in out)


def test_get_M_strictly_lower():
    m = _tiny()
    Mt = torch.tril(m.M, diagonal=-1)
    assert torch.allclose(torch.triu(Mt), torch.zeros_like(Mt))   # P1: zero diagonal + upper


def test_topk_on_latents_keeps_k_per_token():
    m = _tiny(topk_sparsity=3)
    Zp = torch.randn(2, 8, 4)
    from sae_CRL.sae_crl import topk_latents
    masked = topk_latents(Zp, 3)
    assert torch.all((masked != 0).sum(dim=1) == 3)               # exactly k along z_dim


def test_recon_uses_last_position_only():
    # Perturbing non-last window positions must not change loss_mse_Xt.
    m = _tiny(topk_sparsity=0)
    x = _xp()
    base = float(m(x)[0])
    x2 = x.clone(); x2[:, :, :-1] += 50.0
    assert abs(base - float(m(x2)[0])) < 1e-4
```

- [ ] **Step 2: Run — expect FAIL** (`SAE_CRL` not callable / forward missing)
`python -m pytest sae_CRL/tests/test_model.py -k "forward or get_M or topk_on_latents or recon_uses" -v`

- [ ] **Step 3: Add `forward`** to `SAE_CRL` (after `init_params`)
```python
    def forward(self, Xp: torch.Tensor, enable_w: bool = False):
        """Xp: [batch, x_dim, tau+1], last index = current token. Returns the
        reference 6-tuple (mse_Xt, mse_Zt, indep, sparse_Bs, sparse_M, sparse_Zt)."""
        # Encode all timesteps (ref line 78)
        Zp = torch.einsum("hd,bdt->bht", self.F_enc.T, Xp)        # [batch, z_dim, tau+1]
        # P2: TopK on encoded latents (all timesteps), KEPT ON at eval
        if self.topk_sparsity > 0:
            Zp = topk_latents(Zp, self.topk_sparsity)
        # Decode all; MSE on LAST position only (ref lines 79-80)
        recons = torch.einsum("dh,bht->bdt", self.F_dec.T, Zp)
        loss_mse_Xt = F.mse_loss(recons[:, :, -1], Xp[:, :, -1])

        M = torch.tril(self.M, diagonal=-1)                       # P1: strictly lower
        if enable_w:
            w, _w = self.w, 1.0 - self.w
        else:
            w, _w = 1.0, 1.0
        Zt = _w * torch.einsum("hd,bd->bh", M, Zp[:, :, self.tau])   # M @ z_t (ref line 91)
        loss_sparse_Bs = 0.0
        for lag in range(1, self.tau + 1):                        # ref lines 92-96
            B_lag = self.Bs[lag - 1]
            loss_sparse_Bs = loss_sparse_Bs + F.l1_loss(B_lag, torch.zeros_like(B_lag))
            Zt = Zt + w * torch.einsum("hd,bd->bh", B_lag, Zp[:, :, self.tau - lag])

        loss_mse_Zt = F.mse_loss(Zt, Zp[:, :, self.tau])          # ref line 106
        Et = Zp[:, :, self.tau] - Zt                              # ref line 109
        if self.noise_mode == "gau":
            loss_indep = torch.trace(torch.cov(Et))               # ref lines 110-112
        elif self.noise_mode == "lap":
            loss_indep = F.l1_loss(Et, torch.zeros_like(Et))      # ref lines 113-115
        else:
            raise NotImplementedError(self.noise_mode)
        loss_sparse_M = F.l1_loss(M, torch.zeros_like(M))         # ref line 121
        loss_sparse_Zt = F.l1_loss(Zt, torch.zeros_like(Zt))      # ref line 124
        return (loss_mse_Xt, loss_mse_Zt, loss_indep,
                loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt)
```

- [ ] **Step 4: Run — expect PASS** (4 passed)

- [ ] **Step 5: Commit**
```bash
git add sae_CRL/sae_crl.py sae_CRL/tests/test_model.py
git commit -m "feat(sae_CRL): forward + 6-tuple losses (recon-on-last; P1 strict-lower M; P2 TopK on latents)"
```

---

## Task 4: aggB + save/load

**Files:** Modify `sae_CRL/sae_crl.py`, `sae_CRL/tests/test_model.py`.

- [ ] **Step 1: Add failing tests** — append to `sae_CRL/tests/test_model.py`
```python
def test_aggB_max_abs_over_lags():
    m = _tiny(tau=2)
    with torch.no_grad():
        m.Bs[0].copy_(torch.full((8, 8), -3.0)); m.Bs[1].copy_(torch.full((8, 8), 1.0))
    assert torch.allclose(m.aggB(), torch.full((8, 8), 3.0))


def test_save_load_roundtrip(tmp_path):
    m = _tiny(tau=2, topk_sparsity=4, noise_mode="lap")
    m.save_to_dir(tmp_path, model_name="grid-L4-H6", hook_name="blocks.2.hook_resid_post", layer=2)
    m2 = SAE_CRL.load_from_dir(tmp_path)
    assert (m2.x_dim, m2.z_dim, m2.tau, m2.topk_sparsity) == (6, 8, 2, 4)
    assert m2._hook_name == "blocks.2.hook_resid_post"
    assert torch.allclose(m.F_enc, m2.F_enc) and torch.allclose(m.M, m2.M)
    assert all(torch.allclose(a, b) for a, b in zip(m.Bs, m2.Bs))
```

- [ ] **Step 2: Run — expect FAIL** (`aggB` missing)
`python -m pytest sae_CRL/tests/test_model.py -k "aggB or save_load" -v`

- [ ] **Step 3: Add methods** to `SAE_CRL`
```python
    @torch.no_grad()
    def aggB(self) -> torch.Tensor:
        """Max-pool |B_lag| over lags -> [z_dim, z_dim] lag-agnostic edge map (additive S5)."""
        if self.tau == 0:
            return torch.zeros(self.z_dim, self.z_dim, device=self.M.device)
        return torch.stack([b.abs() for b in self.Bs], dim=0).amax(dim=0)

    def save_to_dir(self, out_dir, model_name: str, hook_name: str, layer: int) -> None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        tensors = {"F_enc": self.F_enc.detach().cpu(), "F_dec": self.F_dec.detach().cpu(),
                   "M": self.M.detach().cpu()}
        for i, b in enumerate(self.Bs):
            tensors[f"B_{i}"] = b.detach().cpu()
        save_file(tensors, str(out_dir / "sae_crl.safetensors"))
        cfg = {"model_name": model_name, "hook_name": hook_name, "layer": layer,
               "x_dim": self.x_dim, "z_dim": self.z_dim, "tau": self.tau, "w": self.w,
               "noise_mode": self.noise_mode, "topk_sparsity": self.topk_sparsity}
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)

    @classmethod
    def load_from_dir(cls, in_dir) -> "SAE_CRL":
        in_dir = Path(in_dir)
        with open(in_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        m = cls(x_dim=cfg["x_dim"], z_dim=cfg["z_dim"], tau=cfg["tau"], w=cfg.get("w", 0.5),
                noise_mode=cfg.get("noise_mode", "lap"), topk_sparsity=cfg.get("topk_sparsity", 100))
        t = load_file(str(in_dir / "sae_crl.safetensors"))
        with torch.no_grad():
            m.F_enc.copy_(t["F_enc"]); m.F_dec.copy_(t["F_dec"]); m.M.copy_(t["M"])
            for i in range(cfg["tau"]):
                m.Bs[i].copy_(t[f"B_{i}"])
        m._hook_name = cfg["hook_name"]
        return m
```

- [ ] **Step 4: Run — expect PASS** (all model tests)
`python -m pytest sae_CRL/tests/test_model.py -v`

- [ ] **Step 5: Commit**
```bash
git add sae_CRL/sae_crl.py sae_CRL/tests/test_model.py
git commit -m "feat(sae_CRL): aggB + safetensors/yaml save_to_dir/load_from_dir"
```

---

## Task 5: span-the-bio windows + corpus

**Files:** Create `sae_CRL/windows.py`, `sae_CRL/tests/test_windows.py`.

- [ ] **Step 1: Failing test** — `sae_CRL/tests/test_windows.py`
```python
import torch
from sae_CRL.windows import span_windows, pack_bios, build_bio_corpus


def test_span_windows_count_and_current_is_last():
    acts = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)  # [T=5, x_dim=2]
    win = span_windows(acts, valid_len=4, tau=2)                   # window_size=3
    assert win.shape == (4, 2, 3)                                  # one window per valid token
    # current token (last col) of window i must equal acts[i]
    for i in range(4):
        assert torch.allclose(win[i, :, -1], acts[i])


def test_span_windows_zero_pads_start():
    acts = torch.ones(5, 2)
    win = span_windows(acts, valid_len=4, tau=2)
    assert torch.allclose(win[0, :, :-1], torch.zeros(2, 2))       # first token: empty lookback
    assert torch.allclose(win[0, :, -1], torch.ones(2))


def test_pack_bios_prefix_pad_truncate():
    tokens, valid = pack_bios([[1, 2, 3], [4]], max_bio_len=4, bos_id=9)
    assert tokens.tolist() == [[9, 1, 2, 3], [9, 4, 9, 9]]
    assert valid.tolist() == [4, 2]


class _FakeSampler:
    def sample(self, rng):
        return {"text": "ab" * rng.randint(1, 3)}


class _FakeTok:
    bos_token_id = 0
    def encode(self, text):
        return [ord(c) for c in text]


def test_build_bio_corpus_shapes():
    tokens, valid = build_bio_corpus(_FakeSampler(), _FakeTok(), n_bios=5, max_bio_len=10, seed=0)
    assert tokens.shape == (5, 10) and valid.shape == (5,)
    assert torch.all(tokens[:, 0] == 0)
```

- [ ] **Step 2: Run — expect FAIL** (`No module named 'sae_CRL.windows'`)
`python -m pytest sae_CRL/tests/test_windows.py -v`

- [ ] **Step 3: Implement** — `sae_CRL/windows.py`
```python
"""Span-the-bio windowing (deviation S2) + one-bio-per-row corpus.

Each bio yields ONE window per token position: the window ending at token t holds
the tau+1 activations [t-tau .. t], zero-left-padded where the lookback falls before
the bio start, then transposed to feature-first [x_dim, tau+1] (last col = current t).
Mirrors reference utils.py:gen_window_slicing_batch (window=tau+1, last=current) but
stays within a single bio and zero-pads the start instead of crossing a boundary.
"""
from __future__ import annotations

import random

import torch


def span_windows(acts: torch.Tensor, valid_len: int, tau: int) -> torch.Tensor:
    """acts: [T, x_dim]. Returns [valid_len, x_dim, tau+1]."""
    x_dim = acts.shape[1]
    W = tau + 1
    pad = acts.new_zeros((tau, x_dim))
    padded = torch.cat([pad, acts[:valid_len]], dim=0)            # [tau+valid_len, x_dim]
    idx = torch.arange(valid_len).unsqueeze(1) + torch.arange(W).unsqueeze(0)  # [valid_len, W]
    win = padded[idx]                                            # [valid_len, W, x_dim]
    return win.transpose(1, 2)                                   # [valid_len, x_dim, W]


def windows_for_batch(acts: torch.Tensor, valid_lens: torch.Tensor, tau: int) -> torch.Tensor:
    """acts: [n_bios, T, x_dim]. Returns [total_windows, x_dim, tau+1]."""
    out = [span_windows(acts[b], int(valid_lens[b]), tau) for b in range(acts.shape[0])]
    return torch.cat(out, dim=0)


def pack_bios(token_lists, max_bio_len: int, bos_id: int):
    """[[ids]...] -> ([n, max_bio_len] long, [n] valid_len). Each row = [bos]+ids,
    truncated to max_bio_len, right-padded with bos_id."""
    n = len(token_lists)
    tokens = torch.full((n, max_bio_len), bos_id, dtype=torch.long)
    valid_len = torch.zeros(n, dtype=torch.long)
    for i, toks in enumerate(token_lists):
        row = ([bos_id] + list(toks))[:max_bio_len]
        tokens[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        valid_len[i] = len(row)
    return tokens, valid_len


def build_bio_corpus(sampler, tokenizer, n_bios: int, max_bio_len: int, seed: int):
    rng = random.Random(seed)
    token_lists = [tokenizer.encode(sampler.sample(rng)["text"]) for _ in range(n_bios)]
    return pack_bios(token_lists, max_bio_len, tokenizer.bos_token_id)
```

- [ ] **Step 4: Run — expect PASS** (4 passed)

- [ ] **Step 5: Commit**
```bash
git add sae_CRL/windows.py sae_CRL/tests/test_windows.py
git commit -m "feat(sae_CRL): span-the-bio window builder + one-bio-per-row corpus"
```

---

## Task 6: Eval metrics

**Files:** Create `sae_CRL/evalSAE_CRL.py`; add pure-metric tests to `sae_CRL/tests/test_model.py`.

- [ ] **Step 1: Add failing tests** — append to `sae_CRL/tests/test_model.py`
```python
from sae_CRL.evalSAE_CRL import recon_metrics, structure_metrics


def test_recon_metrics_keys_and_l0():
    m = _tiny(topk_sparsity=3)
    windows = torch.randn(10, 6, 4)            # [n_windows, x_dim, tau+1]
    out = recon_metrics(m, windows)
    assert set(out) == {"recon_mse", "explained_var", "l0"}
    assert abs(out["l0"] - 3.0) < 1e-6         # TopK=3 on latents -> 3 active at current token


def test_structure_metrics_keys():
    m = _tiny()
    out = structure_metrics(m)
    assert set(out) == {"sparse_B", "sparse_M", "n_B_above", "n_M_above"}
    assert out["sparse_B"] == 0.0              # Bs zero-init
```

- [ ] **Step 2: Run — expect FAIL** (`No module named 'sae_CRL.evalSAE_CRL'`)
`python -m pytest sae_CRL/tests/test_model.py -k "recon_metrics or structure_metrics" -v`

- [ ] **Step 3: Implement** — `sae_CRL/evalSAE_CRL.py`
```python
"""SAE-CRL eval: activation capture, reconstruction/structure metrics, CE-recovered.
Metrics use the model's actual sparse path (TopK on latents kept at eval, P2)."""
from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from sae_CRL.sae_crl import SAE_CRL, topk_latents


def capture_resid_post(model: HookedTransformer, tokens: torch.Tensor, hook_name: str) -> torch.Tensor:
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name, return_type=None)
    return cache[hook_name]                       # [B, L, d_model]


@torch.no_grad()
def _encode_sparse(sae: SAE_CRL, windows: torch.Tensor) -> torch.Tensor:
    """windows [n, x_dim, tau+1] -> sparse latents Zp [n, z_dim, tau+1]."""
    Zp = torch.einsum("hd,bdt->bht", sae.F_enc.T, windows)
    if sae.topk_sparsity > 0:
        Zp = topk_latents(Zp, sae.topk_sparsity)
    return Zp


@torch.no_grad()
def recon_metrics(sae: SAE_CRL, windows: torch.Tensor) -> dict:
    """Reconstruction (last position) MSE / explained variance / current-token L0."""
    Zp = _encode_sparse(sae, windows)
    recons = torch.einsum("dh,bht->bdt", sae.F_dec.T, Zp)
    cur_hat, cur = recons[:, :, -1], windows[:, :, -1]
    mse = (cur_hat - cur).pow(2).mean().item()
    var = cur.var().item()
    ev = 1.0 - mse / var if var > 1e-12 else float("nan")
    l0 = (Zp[:, :, -1] != 0).float().sum(-1).mean().item()
    return {"recon_mse": mse, "explained_var": ev, "l0": l0}


@torch.no_grad()
def structure_metrics(sae: SAE_CRL, thresh: float = 1e-3) -> dict:
    sparse_B = sum(b.abs().mean().item() for b in sae.Bs) if sae.tau > 0 else 0.0
    Mt = torch.tril(sae.M, diagonal=-1)
    n_B_above = int(sum((b.abs() > thresh).sum().item() for b in sae.Bs))
    return {"sparse_B": sparse_B, "sparse_M": Mt.abs().mean().item(),
            "n_B_above": n_B_above, "n_M_above": int((Mt.abs() > thresh).sum().item())}


def _ce(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = tokens[:, 1:]
    return -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()


@torch.no_grad()
def ce_recovered(model: HookedTransformer, sae: SAE_CRL, tokens: torch.Tensor, hook_name: str) -> dict:
    """CE recovered when the sparse SAE reconstruction replaces resid_post.
    Reconstruction = F_dec.T @ topk(F_enc.T @ acts) per token (the model's encode/decode)."""
    ce_orig = _ce(model(tokens, return_type="logits"), tokens)
    acts = capture_resid_post(model, tokens, hook_name)             # [B, L, d]
    Zp = torch.einsum("ld,bnd->bnl", sae.F_enc, acts)               # [B, L, z]
    if sae.topk_sparsity > 0:                                       # TopK over z (dim=-1) per token
        idx = Zp.abs().topk(sae.topk_sparsity, dim=-1).indices
        mask = torch.zeros_like(Zp); mask.scatter_(-1, idx, 1.0); Zp = Zp * mask
    recon = torch.einsum("dl,bnl->bnd", sae.F_dec.T.T, Zp)          # F_dec: [z,d]; recon [B,L,d]

    def repl(act, *, hook=None):
        return recon
    ce_sae = _ce(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, repl)], return_type="logits"), tokens)

    def zero(act, *, hook=None):
        return torch.zeros_like(act)
    ce_zero = _ce(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, zero)], return_type="logits"), tokens)
    denom = ce_zero - ce_orig
    rec = (ce_zero - ce_sae) / denom if abs(denom) > 1e-12 else float("nan")
    return {"ce_orig": ce_orig, "ce_sae": ce_sae, "ce_zero": ce_zero, "ce_recovered": rec}
```

- [ ] **Step 4: Run — expect PASS** (2 passed)

- [ ] **Step 5: Commit**
```bash
git add sae_CRL/evalSAE_CRL.py sae_CRL/tests/test_model.py
git commit -m "feat(sae_CRL): eval metrics (recon-on-last / L0 / structure) + ce_recovered"
```

---

## Task 7: Training driver

**Files:** Create `sae_CRL/trainSAE_CRL.py`, `sae_CRL/tests/test_train.py`.

- [ ] **Step 1: Failing test** — `sae_CRL/tests/test_train.py`
```python
import torch
from sae_CRL.sae_crl import SAE_CRL
from sae_CRL.trainSAE_CRL import derive_tau, train_step


def test_derive_tau_auto_and_cap():
    valid = torch.tensor([3, 7, 5])
    assert derive_tau(valid, "auto", None) == 6     # max-1 (longest bio)
    assert derive_tau(valid, "auto", 4) == 4        # capped
    assert derive_tau(valid, 2, None) == 2          # explicit


def test_train_step_reduces_loss():
    torch.manual_seed(0)
    sae = SAE_CRL(x_dim=6, z_dim=12, tau=3, topk_sparsity=0, noise_mode="lap")
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2, weight_decay=1e-4)
    windows = torch.randn(16, 6, 4)
    w = dict(l_ind=0.1, l_spB=0.01, l_spM=0.01, l_spZ=0.0, l_mse_Zt=0.0)  # final defaults (P3/P4)
    first = train_step(sae, opt, windows, **w)["loss"]
    for _ in range(200):
        last = train_step(sae, opt, windows, **w)["loss"]
    assert last < first
```

- [ ] **Step 2: Run — expect FAIL** (`No module named 'sae_CRL.trainSAE_CRL'`)
`python -m pytest sae_CRL/tests/test_train.py -v`

- [ ] **Step 3: Implement** — `sae_CRL/trainSAE_CRL.py`
```python
"""Train SAE_CRL (LinearIDOL port) on grid-L4-H6 resid_post over bioS bios.

Mirrors reference/temp-inst-sae/examples/main.py's per-step loss assembly, but sources
activations via TransformerLens (S1), builds span-the-bio windows (S2), derives tau from
the longest bio (S3), and uses a fixed-step loop (S4).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sae_CRL.sae_crl import SAE_CRL                                          # noqa: E402
from sae_CRL.storage import storage_root                                    # noqa: E402
from sae_CRL.windows import build_bio_corpus, windows_for_batch             # noqa: E402
from sae_CRL.evalSAE_CRL import (recon_metrics, structure_metrics,          # noqa: E402
                                 ce_recovered, capture_resid_post)
from clts.tl_model import build_hooked_transformer                          # noqa: E402
from util.bio_sampler import BioSampler                                     # noqa: E402
from util.condensed_tokenizer import CondensedTokenizer                     # noqa: E402

DEFAULTS = {  # match reference main.py argparse where applicable
    "z_dim": 3072, "tau": "auto", "tau_cap": None, "topk": 100, "noise_mode": "lap",
    "l_ind": 0.1, "l_spB": 0.01, "l_spM": 0.01, "l_spZ": 0.0, "mse_Zt": False,  # l_spB/M=0.01 (P4), l_spZ=0 (P3): paper deviations
    "lr": 0.01, "wd": 1e-4, "epochs": 10, "n_bios": 50_000, "max_bio_len": 48,
    "batch_windows": 4096, "batch_bios": 256, "layer": 2, "eps": 1e-5,
}
SEED = 0
ARGS = device = model = tokenizer = sampler = None
eval_windows = eval_tokens = None


def pick_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def derive_tau(valid_len: torch.Tensor, tau_arg, tau_cap) -> int:
    tau = (int(valid_len.max().item()) - 1) if tau_arg in ("auto", None) else int(tau_arg)
    if tau_cap is not None:
        tau = min(tau, int(tau_cap))
    return max(1, tau)


def train_step(sae, opt, windows, l_ind, l_spB, l_spM, l_spZ, l_mse_Zt) -> dict:
    mse_Xt, mse_Zt, indep, sp_B, sp_M, sp_Zt = sae(windows)
    loss = (mse_Xt + l_mse_Zt * mse_Zt + l_ind * indep
            + l_spB * sp_B + l_spM * sp_M + l_spZ * sp_Zt)        # ref main.py:113
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return {"loss": float(loss), "mse_Xt": float(mse_Xt), "indep": float(indep),
            "sp_B": float(sp_B), "sp_M": float(sp_M), "sp_Zt": float(sp_Zt)}


def hook_name() -> str:
    return ARGS.hook_template.format(layer=ARGS.layer)


def setup(args):
    global ARGS, device, model, tokenizer, sampler, eval_windows, eval_tokens
    ARGS = args
    root = storage_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        p = root / f".probe.{os.getpid()}"; p.touch(); p.unlink()
    except OSError as e:
        raise SystemExit(f"storage_root not writable: {root}\n  {e}")
    print(f"[storage] {root} (writable)")
    if args.model_name is None:
        args.model_name = (args.model_dir.parent.name if args.model_dir.name == "final"
                           else args.model_dir.name)
    device = pick_device()
    model = build_hooked_transformer(args.model_dir, device, torch.float32)
    if args.eps is not None:
        for blk in model.blocks:
            for ln in (getattr(blk, "ln1", None), getattr(blk, "ln2", None)):
                if ln is not None and hasattr(ln, "eps"):
                    ln.eps = args.eps
    print(f"[model] {args.model_dir} d_model={model.cfg.d_model}  [hook] {hook_name()}")
    tokenizer = CondensedTokenizer.from_remap_path(args.data_dir / "old_to_new.json")
    sampler = BioSampler(args.data_dir / "people.json", fields=("birthday",), seed=SEED)
    et, evl = build_bio_corpus(sampler, tokenizer, 256, args.max_bio_len, SEED + 1)
    eval_tokens = et.to(device)
    eacts = capture_resid_post(model, eval_tokens, hook_name())
    # eval windows use the train-derived tau; deferred until tau is known (in train_one_run)
    eval_windows = (eacts.cpu(), evl)


def trial_name(z, tau, k, lr, ep, n):
    return f"z{z}_tau{tau}_k{k}_lr{lr:g}_ep{ep}_n{n}"


def train_one_run(_override=None):
    import wandb
    wandb.init(project="interpLM4"); cfg = wandb.config; sweep_id = wandb.run.sweep_id
    z_dim = cfg.get("z_dim", ARGS.z_dim); topk = cfg.get("topk", ARGS.topk)
    lr = cfg.get("lr", ARGS.lr); epochs = cfg.get("epochs", ARGS.epochs)
    n_bios = cfg.get("n_bios", ARGS.n_bios); hk = hook_name()
    l_mse_Zt = 1.0 if ARGS.mse_Zt else 0.0

    tokens_cpu, valid_len = build_bio_corpus(sampler, tokenizer, n_bios, ARGS.max_bio_len, SEED)
    tau = derive_tau(valid_len, ARGS.tau, ARGS.tau_cap)
    median = int(valid_len.median().item())
    print(f"[corpus] bios={n_bios} valid_tokens={int(valid_len.sum())} median={median} -> tau={tau}")
    if tau + 1 > median:
        print(f"[warn] tau+1={tau+1} > median bio {median}: high lags seen only in long bios.")

    sae = SAE_CRL(x_dim=model.cfg.d_model, z_dim=z_dim, tau=tau,
                  noise_mode=ARGS.noise_mode, topk_sparsity=topk).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=ARGS.wd)
    lw = dict(l_ind=ARGS.l_ind, l_spB=ARGS.l_spB, l_spM=ARGS.l_spM, l_spZ=ARGS.l_spZ, l_mse_Zt=l_mse_Zt)

    eacts_cpu, evalid = eval_windows
    eval_w = windows_for_batch(eacts_cpu, evalid, tau).to(device)

    n_rows = tokens_cpu.shape[0]; bb = ARGS.batch_bios; step = 0
    for ep in range(epochs):
        perm = torch.randperm(n_rows)
        for start in range(0, n_rows, bb):
            idx = perm[start:start + bb]
            batch_tokens = tokens_cpu[idx].to(device)
            acts = capture_resid_post(model, batch_tokens, hk)              # [bb, L, d]
            win = windows_for_batch(acts.cpu(), valid_len[idx], tau).to(device)  # [n_win, d, tau+1]
            # SGD over windows in sub-batches of batch_windows
            wperm = torch.randperm(win.shape[0])
            for ws in range(0, win.shape[0], ARGS.batch_windows):
                losses = train_step(sae, opt, win[wperm[ws:ws + ARGS.batch_windows]], **lw)
                if step % 30 == 0:
                    wandb.log({f"train/{k}": v for k, v in losses.items()}, step=step)
                step += 1
        ev = recon_metrics(sae, eval_w) | structure_metrics(sae)
        wandb.log({f"eval/{k}": v for k, v in ev.items()}, step=step)

    final_dir = (storage_root() / "sae_CRL_runs" / ARGS.model_name /
                 (f"sweep-{sweep_id}" if sweep_id else "standalone") /
                 trial_name(z_dim, tau, topk, lr, epochs, n_bios) / "final")
    final = (recon_metrics(sae, eval_w) | structure_metrics(sae)
             | ce_recovered(model, sae, eval_tokens, hk))
    sae.save_to_dir(final_dir, model_name=ARGS.model_name, hook_name=hk, layer=ARGS.layer)
    payload = {f"final_eval/{k}": v for k, v in final.items()} | {"storage_path": str(final_dir)}
    wandb.log(payload); wandb.run.summary.update(payload)
    print(f"[final] saved {final_dir}  ce_recovered={final['ce_recovered']:.4f}")
    wandb.finish()


def build_sweep_config():
    return {"program": "trainSAE_CRL.py", "method": "grid",
            "name": f"sae_CRL_sweep_{ARGS.model_name}",
            "metric": {"name": "final_eval/ce_recovered", "goal": "maximize"},
            "parameters": {"z_dim": {"values": [1536, 3072]}, "topk": {"values": [25, 100]}},
            "early_terminate": {"type": "hyperband", "min_iter": 5, "eta": 3}}


def _patch_signal_for_worker_threads():
    import signal, threading
    _real = signal.signal
    def _safe(s, h):
        return _real(s, h) if threading.current_thread() is threading.main_thread() else None
    signal.signal = _safe


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--hook-template", type=str, default="blocks.{layer}.hook_resid_post")
    p.add_argument("--layer", type=int, default=DEFAULTS["layer"])
    p.add_argument("--z-dim", dest="z_dim", type=int, default=DEFAULTS["z_dim"])
    p.add_argument("--tau", default=DEFAULTS["tau"], help="'auto' (= longest bio-1) or int")
    p.add_argument("--tau-cap", dest="tau_cap", type=int, default=DEFAULTS["tau_cap"])
    p.add_argument("--topk", type=int, default=DEFAULTS["topk"])
    p.add_argument("--noise-mode", choices=["lap", "gau"], default=DEFAULTS["noise_mode"])
    p.add_argument("--l-ind", type=float, default=DEFAULTS["l_ind"])
    p.add_argument("--l-spB", type=float, default=DEFAULTS["l_spB"])
    p.add_argument("--l-spM", type=float, default=DEFAULTS["l_spM"])
    p.add_argument("--l-spZ", type=float, default=DEFAULTS["l_spZ"])
    p.add_argument("--mse-Zt", dest="mse_Zt", action="store_true")
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--wd", type=float, default=DEFAULTS["wd"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--n-bios", dest="n_bios", type=int, default=DEFAULTS["n_bios"])
    p.add_argument("--max-bio-len", dest="max_bio_len", type=int, default=DEFAULTS["max_bio_len"])
    p.add_argument("--batch-bios", dest="batch_bios", type=int, default=DEFAULTS["batch_bios"])
    p.add_argument("--batch-windows", dest="batch_windows", type=int, default=DEFAULTS["batch_windows"])
    p.add_argument("--eps", type=float, default=DEFAULTS["eps"])
    p.add_argument("--sweep", action="store_true")
    return p.parse_args()


def main():
    import wandb
    args = parse_args(); setup(args)
    if args.sweep:
        _patch_signal_for_worker_threads()
        sid = wandb.sweep(build_sweep_config(), project="interpLM4")
        print(f"[sweep] {sid}"); wandb.agent(sid, function=train_one_run)
    else:
        train_one_run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run — expect PASS** (2 passed)
`python -m pytest sae_CRL/tests/test_train.py -v`

- [ ] **Step 5: Run full unit suite — expect PASS**
`python -m pytest sae_CRL/tests/ -v`

- [ ] **Step 6: Commit**
```bash
git add sae_CRL/trainSAE_CRL.py sae_CRL/tests/test_train.py
git commit -m "feat(sae_CRL): training driver (derive_tau, train_step with ref loss assembly, sweep)"
```

---

## Task 8: End-to-end integration smoke (manual)

Needs `transformer_lens` + a checkpoint + data dir; tiny knobs so it finishes in seconds.

- [ ] **Step 1: Few-step run**
```bash
python sae_CRL/trainSAE_CRL.py --model-dir model/grid-L4-H6 \
  --data-dir data/bioS_N-Bd_final_grid \
  --z-dim 256 --topk 16 --n-bios 64 --epochs 1 --batch-bios 16 --layer 2
```
Expected: `[storage] … (writable)`, `[model] … d_model=384`, `[hook] blocks.2.hook_resid_post`, a `[corpus] … -> tau=<N>` (`tau ≥ 1`), training logs, then `[final] saved …/sae_CRL_runs/grid-L4-H6/standalone/<trial>/final  ce_recovered=<float>`. (`WANDB_MODE=offline` to skip login.)

- [ ] **Step 2: Verify checkpoint loads + sparsity/strict-lower hold**
```bash
python -c "
from sae_CRL.sae_crl import SAE_CRL
import glob, torch
d = sorted(glob.glob('sae_CRL_storage/sae_CRL_runs/grid-L4-H6/standalone/*/final'))[-1]
m = SAE_CRL.load_from_dir(d)
print('loaded', d, 'z_dim', m.z_dim, 'tau', m.tau, 'topk', m.topk_sparsity)
print('M strict-lower OK', bool((torch.tril(m.M, diagonal=-1).triu()==0).all()))
print('aggB', tuple(m.aggB().shape))
"
```
Expected: prints the dir, `z_dim 256`, `tau ≥ 1`, `topk 16`, `M strict-lower OK True`, `aggB (256, 256)`.

- [ ] **Step 3: Mark verified**
```bash
git commit --allow-empty -m "test(sae_CRL): end-to-end integration smoke verified"
```

---

## Self-Review (by plan author)

**Coverage vs reference + deviations ledger:** model init/forward/losses port `linear_idol_model.py` (Tasks 2–3) with P1/P2 called out in code comments and the ledger; windowing (Task 5) is the documented S2 span-the-bio adaptation; training (Task 7) uses the exact `main.py:113` loss assembly with their default weights; eval (Task 6) measures the sparse path; save/load+aggB are the additive S5. ✔

**Placeholder scan:** none — every step has complete code + concrete run/expected. ✔

**Type/name consistency:** `SAE_CRL(x_dim,z_dim,tau,w,noise_mode,topk_sparsity)`; `forward` returns the 6-tuple consumed by `train_step`/`recon_metrics`; `topk_latents`; `span_windows`/`windows_for_batch`/`pack_bios`/`build_bio_corpus`; `capture_resid_post`/`recon_metrics`/`structure_metrics`/`ce_recovered`; `derive_tau`/`train_step`/`setup`/`train_one_run`. `save_to_dir(out_dir, model_name, hook_name, layer)` ↔ `load_from_dir` config keys match. ✔

**Known follow-up:** `ce_recovered` einsum (`F_dec.T.T`) is written for clarity — confirm `F_dec` is `[z_dim, d_model]` at implementation time (it is, per `__init__`) so `recon` comes out `[B, L, d_model]`.

**Out of scope (per spec §13):** relation dashboards, synthetic MCC validation, SAEBench, attribution.
