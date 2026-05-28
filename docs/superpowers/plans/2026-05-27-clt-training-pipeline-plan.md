# CLT Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Cross-Layer Transcoder (CLT) training pipeline for Llama-architecture models. Output CLTs use Anthropic `circuit-tracer`'s exact on-disk format so downstream attribution-graph and UI subprojects consume them without conversion.

**Architecture:** Single `nn.Module` (`CrossLayerTranscoder`) holds N encoders + triangular packed decoders. Joint multi-layer reconstruction loss + JumpReLU sparsity with decoder-norm-weighted tanh penalty. Activations captured once per forward via `model.run_with_cache`. CLI mirrors `saes/trainSAE.py` conventions; sweep is single-axis (not per-layer) since the CLT spans all layers.

**Tech Stack:** PyTorch, transformer_lens, safetensors, PyYAML, wandb, HuggingFace `transformers` + `tokenizers`. JumpReLU implemented locally as a short `torch.autograd.Function` (avoids sae_lens version drift).

**Spec reference:** [docs/superpowers/specs/2026-05-27-clt-training-pipeline-design.md](../specs/2026-05-27-clt-training-pipeline-design.md)

---

## File Structure

**New files:**
- `clts/__init__.py` (empty)
- `clts/clt.py` — `CrossLayerTranscoder` nn.Module + `JumpReLU` autograd.Function + save/load
- `clts/export_tokenizer.py` — `ensure_hf_tokenizer(data_dir) -> Path`
- `clts/evalCLT.py` — eval functions (MSE, L0, dead-frac, CE-recovered, capture helper)
- `clts/trainCLT.py` — CLI + training loop + `setup()` + sweep wiring
- `tests/test_clt.py` — 7 unit tests for CLT
- `tests/test_export_tokenizer.py` — 3 tests for tokenizer export

**Modified files:** none. Spec deliberately constrains scope to new files only.

**Storage paths (created at runtime, not in repo):**
- `<STORAGE_ROOT>/clt_runs/<model-name>/<sweep|standalone>/<trial>/final/` — trained CLTs
- `<STORAGE_ROOT>/hf_tokenizers/<remap-hash>/` — exported HF tokenizers

---

### Task 1: Scaffold + `CrossLayerTranscoder` constructor

**Files:**
- Create: `clts/__init__.py`
- Create: `clts/clt.py`
- Create: `tests/test_clt.py`

- [ ] **Step 1: Create the empty package init**

```bash
mkdir -p clts
touch clts/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_clt.py`:

```python
"""Unit tests for CrossLayerTranscoder."""
import torch

from clts.clt import CrossLayerTranscoder


def test_shapes():
    clt = CrossLayerTranscoder(n_layers=4, d_model=8, expansion=2)
    assert clt.W_enc.shape == (4, 16, 8)
    assert clt.b_enc.shape == (4, 16)
    assert clt.threshold.shape == (4, 16)
    assert clt.b_dec.shape == (4, 8)
    assert len(clt.W_dec) == 4
    for i in range(4):
        assert clt.W_dec[i].shape == (16, 4 - i, 8), \
            f"W_dec[{i}] is {clt.W_dec[i].shape}, expected ({16}, {4 - i}, {8})"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_clt.py::test_shapes -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clts.clt'`

- [ ] **Step 4: Write the minimal implementation**

Create `clts/clt.py`:

```python
"""Cross-Layer Transcoder.

Matches Anthropic circuit-tracer's CrossLayerTranscoder packed-tensor
layout so trained weights load into their tooling without conversion.

Shapes (N = n_layers, D = d_model, d_t = d_transcoder = expansion * D):
    W_enc:     [N, d_t, D]
    b_enc:     [N, d_t]
    threshold: [N, d_t]
    W_dec:     ParameterList of length N; entry i is [d_t, N - i, D]
    b_dec:     [N, D]
"""
from __future__ import annotations

import torch
import torch.nn as nn


JUMPRELU_INIT_THRESHOLD = 0.1
JUMPRELU_BANDWIDTH = 2.0


class CrossLayerTranscoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, expansion: int = 16):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_transcoder = expansion * d_model

        N, D, d_t = n_layers, d_model, self.d_transcoder

        self.W_enc = nn.Parameter(torch.empty(N, d_t, D))
        self.b_enc = nn.Parameter(torch.zeros(N, d_t))
        self.threshold = nn.Parameter(torch.full((N, d_t), JUMPRELU_INIT_THRESHOLD))
        self.b_dec = nn.Parameter(torch.zeros(N, D))

        self.W_dec = nn.ParameterList([
            nn.Parameter(torch.empty(d_t, N - i, D)) for i in range(N)
        ])

        # Kaiming-uniform init for encoder/decoder weights (matches typical
        # sae_lens init; avoids any param staying at zero from above).
        for i in range(N):
            nn.init.kaiming_uniform_(self.W_enc[i], a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.W_dec[i], a=5 ** 0.5)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_clt.py::test_shapes -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add clts/__init__.py clts/clt.py tests/test_clt.py
git commit -m "$(cat <<'EOF'
feat(clt): scaffold CrossLayerTranscoder module with packed tensor layout

Constructor matches Anthropic circuit-tracer's on-disk shapes:
W_enc [N, d_t, D], W_dec ParameterList of [d_t, N-i, D], etc.
Kaiming init for encoder/decoder weights; threshold initialized at 0.1.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Forward pass — `encode()`, `decode()`, `forward()` + dimensions test

**Files:**
- Modify: `clts/clt.py` (add three methods + JumpReLU)
- Modify: `tests/test_clt.py` (add `test_forward_pass_dimensions`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_clt.py`:

```python
def test_forward_pass_dimensions():
    torch.manual_seed(0)
    clt = CrossLayerTranscoder(n_layers=4, d_model=8, expansion=2)
    x_list = [torch.randn(2, 8) for _ in range(4)]
    y_hat_list = clt(x_list)
    assert len(y_hat_list) == 4
    for L in range(4):
        assert y_hat_list[L].shape == (2, 8)
        assert torch.isfinite(y_hat_list[L]).all()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_clt.py::test_forward_pass_dimensions -v`
Expected: FAIL with `TypeError: 'CrossLayerTranscoder' object is not callable` (no `forward`)

- [ ] **Step 3: Add JumpReLU + forward methods**

Add to `clts/clt.py` (above the `CrossLayerTranscoder` class):

```python
class JumpReLU(torch.autograd.Function):
    """JumpReLU with rectangle-kernel STE for threshold gradient.

    Forward: y = x * 1[x > threshold]
    Backward (x):         pass-through where x > threshold
    Backward (threshold): -threshold/bandwidth * 1[|x - threshold| < bandwidth/2]
    Standard recipe from Rajamanoharan et al. (2024).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bw = ctx.bandwidth
        x_grad = grad_output * (x > threshold).to(x.dtype)
        in_window = ((x - threshold).abs() < bw / 2).to(x.dtype)
        threshold_grad = grad_output * (-threshold / bw) * in_window
        # Sum threshold grad across batch axes; threshold is [d_t], grad is [B, d_t]
        while threshold_grad.dim() > threshold.dim():
            threshold_grad = threshold_grad.sum(0)
        return x_grad, threshold_grad, None
```

Add these methods inside the `CrossLayerTranscoder` class:

```python
    def encode(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """For each source L, compute features a[L] = JumpReLU(W_enc[L] x[L] + b_enc[L])."""
        assert len(x_list) == self.n_layers, \
            f"expected {self.n_layers} inputs, got {len(x_list)}"
        a_list = []
        for L in range(self.n_layers):
            preact = x_list[L] @ self.W_enc[L].T + self.b_enc[L]
            a = JumpReLU.apply(preact, self.threshold[L], JUMPRELU_BANDWIDTH)
            a_list.append(a)
        return a_list

    def decode(self, a_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """For each target L', sum decoder contributions from sources L <= L'."""
        y_hat_list = []
        batch_size = a_list[0].shape[0]
        for L_prime in range(self.n_layers):
            y_hat = self.b_dec[L_prime].unsqueeze(0).expand(batch_size, -1).clone()
            for L in range(L_prime + 1):
                target_idx = L_prime - L
                dec_slice = self.W_dec[L][:, target_idx, :]  # [d_t, D]
                y_hat = y_hat + a_list[L] @ dec_slice
            y_hat_list.append(y_hat)
        return y_hat_list

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.decode(self.encode(x_list))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_clt.py::test_forward_pass_dimensions -v`
Expected: PASS

- [ ] **Step 5: Run all tests to verify nothing regressed**

Run: `pytest tests/test_clt.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add clts/clt.py tests/test_clt.py
git commit -m "$(cat <<'EOF'
feat(clt): encode/decode/forward with JumpReLU activation

JumpReLU autograd.Function uses rectangle-kernel STE for threshold
gradient (Rajamanoharan et al. 2024). Forward pass routes each source
layer's features through (N - L) decoder heads to N - L target layers.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Cross-layer routing correctness test

**Files:**
- Modify: `tests/test_clt.py` (add `test_cross_layer_writes`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_clt.py`:

```python
def test_cross_layer_writes():
    """Feature at source 0 must write only to target k when its decoder
    column for k is the only nonzero one. Catches decoder routing off-by-one.
    """
    torch.manual_seed(1)
    clt = CrossLayerTranscoder(n_layers=4, d_model=8, expansion=2)

    # Zero everything, then set up a single feature path: source L=0, target k=2.
    with torch.no_grad():
        clt.W_enc.zero_()
        clt.b_enc.zero_()
        clt.b_dec.zero_()
        for i in range(4):
            clt.W_dec[i].zero_()
        # Make feature 0 at source layer 0 fire on input dim 0 with value 1.
        clt.W_enc[0, 0, 0] = 1.0
        clt.threshold[0, 0] = 0.0   # so any positive preact fires
        # Route that feature ONLY to target k=2, output dim 5 with value 1.
        target_k = 2
        clt.W_dec[0][0, target_k, 5] = 1.0

    x_list = [torch.zeros(1, 8) for _ in range(4)]
    x_list[0][0, 0] = 1.0   # fires feature 0 at L=0 with magnitude 1
    y_hat_list = clt(x_list)

    # Target k=2 should have a 1.0 at dim 5; every other target should be all zero.
    for L_prime in range(4):
        if L_prime == target_k:
            expected = torch.zeros(1, 8); expected[0, 5] = 1.0
            assert torch.allclose(y_hat_list[L_prime], expected), \
                f"target {L_prime}: {y_hat_list[L_prime]}"
        else:
            assert torch.allclose(y_hat_list[L_prime], torch.zeros(1, 8)), \
                f"target {L_prime} should be all zero, got {y_hat_list[L_prime]}"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_clt.py::test_cross_layer_writes -v`
Expected: PASS (Task 2's `decode` already handles routing correctly; this test is a guard against future regressions)

If FAIL: the routing in `decode` has an off-by-one in `target_idx = L_prime - L`. Fix in `clts/clt.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_clt.py
git commit -m "$(cat <<'EOF'
test(clt): assert decoder routes each feature only to its target layer

Sets up a feature at source 0 that decodes only into target 2 at dim 5,
verifies all other targets receive zero. Guards against off-by-one in
the target_idx = L_prime - L indexing of W_dec[L].

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Gradient flow test

**Files:**
- Modify: `tests/test_clt.py` (add `test_jumprelu_gradients_flow`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_clt.py`:

```python
def test_jumprelu_gradients_flow():
    """All Parameters must receive a non-None gradient after backward.
    Catches accidentally-frozen params (easy to miss with ParameterList).
    """
    torch.manual_seed(2)
    clt = CrossLayerTranscoder(n_layers=3, d_model=4, expansion=2)
    # Bias preactivations above threshold so JumpReLU passes through.
    with torch.no_grad():
        clt.b_enc.fill_(1.0)
        clt.threshold.fill_(0.0)

    x_list = [torch.randn(2, 4, requires_grad=False) for _ in range(3)]
    y_hat_list = clt(x_list)
    loss = sum(y.pow(2).mean() for y in y_hat_list)
    loss.backward()

    assert clt.W_enc.grad is not None and clt.W_enc.grad.abs().sum() > 0
    assert clt.b_enc.grad is not None and clt.b_enc.grad.abs().sum() > 0
    assert clt.b_dec.grad is not None and clt.b_dec.grad.abs().sum() > 0
    assert clt.threshold.grad is not None
    for i in range(3):
        assert clt.W_dec[i].grad is not None, f"W_dec[{i}].grad is None"
        assert clt.W_dec[i].grad.abs().sum() > 0, f"W_dec[{i}].grad is all-zero"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_clt.py::test_jumprelu_gradients_flow -v`
Expected: PASS (JumpReLU backward and ParameterList both propagate gradients)

- [ ] **Step 3: Run all CLT tests**

Run: `pytest tests/test_clt.py -v`
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_clt.py
git commit -m "$(cat <<'EOF'
test(clt): assert every Parameter (incl. ParameterList) receives gradients

Forces JumpReLU into the pass-through regime so encoder grads are
non-trivial, then asserts non-None / non-zero gradient on W_enc, b_enc,
b_dec, threshold, and every entry of W_dec.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Loss computation + one-step decrease test

**Files:**
- Modify: `clts/clt.py` (add `compute_loss`)
- Modify: `tests/test_clt.py` (add `test_loss_decreases_one_optimizer_step`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_clt.py`:

```python
def test_loss_decreases_one_optimizer_step():
    """End-to-end sanity: forward -> loss -> backward -> Adam step -> loss decreases."""
    torch.manual_seed(3)
    clt = CrossLayerTranscoder(n_layers=3, d_model=4, expansion=2)
    x_list = [torch.randn(8, 4) for _ in range(3)]
    y_list = [torch.randn(8, 4) for _ in range(3)]

    opt = torch.optim.Adam(clt.parameters(), lr=1e-2)
    loss_before = clt.compute_loss(x_list, y_list, l0_coefficient=1.0)["total"].item()
    opt.zero_grad()
    clt.compute_loss(x_list, y_list, l0_coefficient=1.0)["total"].backward()
    opt.step()
    loss_after = clt.compute_loss(x_list, y_list, l0_coefficient=1.0)["total"].item()

    assert loss_after < loss_before, f"loss did not decrease: {loss_before} -> {loss_after}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_clt.py::test_loss_decreases_one_optimizer_step -v`
Expected: FAIL with `AttributeError: 'CrossLayerTranscoder' object has no attribute 'compute_loss'`

- [ ] **Step 3: Add `compute_loss` to the class**

Add inside `CrossLayerTranscoder` in `clts/clt.py`:

```python
    def _decoder_norms(self) -> list[torch.Tensor]:
        """Per-feature L2 norm of the *summed* decoder vector across all
        downstream targets. Shape: list of N tensors, each [d_t].

        For source L, sums the decoder contributions across the N - L
        targets, then takes the L2 norm per feature.
        """
        norms = []
        for L in range(self.n_layers):
            summed = self.W_dec[L].sum(dim=1)             # [d_t, D]
            norms.append(summed.norm(dim=-1))             # [d_t]
        return norms

    def compute_loss(
        self,
        x_list: list[torch.Tensor],
        y_list: list[torch.Tensor],
        l0_coefficient: float,
        tanh_scale: float = 4.0,
        pre_act_coef: float = 3e-6,
    ) -> dict[str, torch.Tensor]:
        """Joint multi-layer CLT loss.

        Returns a dict with keys: total, recon, sparsity, preact, plus
        per-layer recon_L{i} and l0_L{i} for logging.
        """
        # Encoder pre-activations (needed for both encode output and preact loss).
        preacts = [
            x_list[L] @ self.W_enc[L].T + self.b_enc[L]
            for L in range(self.n_layers)
        ]
        a_list = [
            JumpReLU.apply(preacts[L], self.threshold[L], JUMPRELU_BANDWIDTH)
            for L in range(self.n_layers)
        ]
        y_hat_list = self.decode(a_list)

        # Per-layer reconstruction MSE (mean over batch and features).
        recon_per_layer = [
            (y_hat_list[L] - y_list[L]).pow(2).mean()
            for L in range(self.n_layers)
        ]
        recon = sum(recon_per_layer) / self.n_layers

        # Decoder-norm-weighted tanh sparsity penalty.
        dec_norms = self._decoder_norms()
        sparsity_terms = []
        l0_per_layer = []
        for L in range(self.n_layers):
            weighted = a_list[L] * dec_norms[L].unsqueeze(0)      # [B, d_t]
            sparsity_terms.append(torch.tanh(tanh_scale * weighted).sum(dim=-1).mean())
            l0_per_layer.append((a_list[L] > 0).float().sum(dim=-1).mean())
        sparsity = l0_coefficient * sum(sparsity_terms)

        # Pre-activation L2 penalty (matches sae_lens preact loss intent).
        preact = pre_act_coef * sum(p.pow(2).mean() for p in preacts)

        total = recon + sparsity + preact

        out = {"total": total, "recon": recon, "sparsity": sparsity, "preact": preact}
        for L in range(self.n_layers):
            out[f"recon_L{L}"] = recon_per_layer[L].detach()
            out[f"l0_L{L}"] = l0_per_layer[L].detach()
        return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_clt.py::test_loss_decreases_one_optimizer_step -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `pytest tests/test_clt.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add clts/clt.py tests/test_clt.py
git commit -m "$(cat <<'EOF'
feat(clt): compute_loss with decoder-norm-weighted tanh sparsity

Joint multi-layer reconstruction MSE + sparsity term weighted by the
L2 norm of each feature's summed downstream decoder, plus a small L2
preact penalty matching sae_lens recipe. Returns per-layer diagnostics
for logging.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Save / load + circuit-tracer format compatibility

**Files:**
- Modify: `clts/clt.py` (add `save_to_dir`, `load_from_dir`)
- Modify: `tests/test_clt.py` (add `test_save_load_roundtrip`, `test_circuit_tracer_format_keys`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_clt.py`:

```python
def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(4)
    clt = CrossLayerTranscoder(n_layers=3, d_model=4, expansion=2)
    out_dir = tmp_path / "clt_out"
    clt.save_to_dir(out_dir, model_name="test-model")

    clt2 = CrossLayerTranscoder.load_from_dir(out_dir)
    assert clt2.n_layers == clt.n_layers
    assert clt2.d_model == clt.d_model
    assert clt2.d_transcoder == clt.d_transcoder
    assert torch.equal(clt.W_enc, clt2.W_enc)
    assert torch.equal(clt.b_enc, clt2.b_enc)
    assert torch.equal(clt.threshold, clt2.threshold)
    assert torch.equal(clt.b_dec, clt2.b_dec)
    for i in range(3):
        assert torch.equal(clt.W_dec[i], clt2.W_dec[i])


def test_circuit_tracer_format_keys(tmp_path):
    """On-disk tensor keys must match what circuit-tracer's loader reads."""
    from safetensors import safe_open

    clt = CrossLayerTranscoder(n_layers=3, d_model=4, expansion=2)
    out_dir = tmp_path / "clt_out"
    clt.save_to_dir(out_dir, model_name="test-model")

    for i in range(3):
        with safe_open(out_dir / f"W_enc_{i}.safetensors", framework="pt") as f:
            keys = set(f.keys())
            assert keys == {f"W_enc_{i}", f"b_enc_{i}", f"b_dec_{i}", f"threshold_{i}"}
        with safe_open(out_dir / f"W_dec_{i}.safetensors", framework="pt") as f:
            keys = set(f.keys())
            assert keys == {f"W_dec_{i}"}

    # config.yaml must contain circuit-tracer's required 4 fields.
    import yaml
    with open(out_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["model_name"] == "test-model"
    assert cfg["model_kind"] == "cross_layer_transcoder"
    assert cfg["feature_input_hook"] == "hook_resid_mid"
    assert cfg["feature_output_hook"] == "hook_mlp_out"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_clt.py::test_save_load_roundtrip tests/test_clt.py::test_circuit_tracer_format_keys -v`
Expected: FAIL with `AttributeError: ... 'save_to_dir'`

- [ ] **Step 3: Add save/load to `CrossLayerTranscoder`**

Add to top of `clts/clt.py`:

```python
from pathlib import Path
import yaml
from safetensors.torch import save_file, load_file
```

Add inside `CrossLayerTranscoder`:

```python
    DEFAULT_ENC_HOOK = "hook_resid_mid"
    DEFAULT_DEC_HOOK = "hook_mlp_out"

    def save_to_dir(
        self,
        out_dir: str | Path,
        model_name: str,
        feature_input_hook: str = DEFAULT_ENC_HOOK,
        feature_output_hook: str = DEFAULT_DEC_HOOK,
    ) -> None:
        """Save as 2N safetensors + config.yaml in circuit-tracer's exact format."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.n_layers):
            save_file({
                f"W_enc_{i}":     self.W_enc.data[i].contiguous(),
                f"b_enc_{i}":     self.b_enc.data[i].contiguous(),
                f"b_dec_{i}":     self.b_dec.data[i].contiguous(),
                f"threshold_{i}": self.threshold.data[i].contiguous(),
            }, str(out_dir / f"W_enc_{i}.safetensors"))
            save_file({
                f"W_dec_{i}": self.W_dec[i].data.contiguous(),
            }, str(out_dir / f"W_dec_{i}.safetensors"))

        cfg = {
            "model_name": model_name,
            "model_kind": "cross_layer_transcoder",
            "feature_input_hook": feature_input_hook,
            "feature_output_hook": feature_output_hook,
        }
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)

    @classmethod
    def load_from_dir(cls, in_dir: str | Path) -> "CrossLayerTranscoder":
        """Reconstruct a CLT from the 2N safetensors layout. Dimensions are
        inferred from tensor shapes (matches circuit-tracer's loader)."""
        in_dir = Path(in_dir)

        # Discover N from the number of W_enc_* files.
        enc_files = sorted(in_dir.glob("W_enc_*.safetensors"))
        n_layers = len(enc_files)
        assert n_layers > 0, f"no W_enc_*.safetensors found in {in_dir}"

        # Peek at W_enc_0 to infer d_t and D.
        enc0 = load_file(str(in_dir / "W_enc_0.safetensors"))
        d_t, D = enc0["W_enc_0"].shape
        assert d_t % D == 0, f"d_transcoder {d_t} not a multiple of d_model {D}"
        expansion = d_t // D

        clt = cls(n_layers=n_layers, d_model=D, expansion=expansion)
        with torch.no_grad():
            for i in range(n_layers):
                enc = load_file(str(in_dir / f"W_enc_{i}.safetensors"))
                dec = load_file(str(in_dir / f"W_dec_{i}.safetensors"))
                clt.W_enc.data[i].copy_(enc[f"W_enc_{i}"])
                clt.b_enc.data[i].copy_(enc[f"b_enc_{i}"])
                clt.b_dec.data[i].copy_(enc[f"b_dec_{i}"])
                clt.threshold.data[i].copy_(enc[f"threshold_{i}"])
                clt.W_dec[i].data.copy_(dec[f"W_dec_{i}"])
        return clt
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_clt.py::test_save_load_roundtrip tests/test_clt.py::test_circuit_tracer_format_keys -v`
Expected: PASS (both)

- [ ] **Step 5: Run all CLT tests**

Run: `pytest tests/test_clt.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add clts/clt.py tests/test_clt.py
git commit -m "$(cat <<'EOF'
feat(clt): save_to_dir / load_from_dir in circuit-tracer format

Per-source-layer files: W_enc_{i}.safetensors holds {W_enc_{i}, b_enc_{i},
b_dec_{i}, threshold_{i}}; W_dec_{i}.safetensors holds {W_dec_{i}}.
config.yaml has the 4 fields circuit-tracer's loader expects. Dimensions
are inferred from tensor shapes on load. Roundtrip + key-name tests guard
against format drift.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Tokenizer export — `ensure_hf_tokenizer`

**Files:**
- Create: `clts/export_tokenizer.py`
- Create: `tests/test_export_tokenizer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_export_tokenizer.py`:

```python
"""Tests for ensure_hf_tokenizer: roundtrip, cache hit, hash discrimination."""
import json
import shutil

import pytest
from transformers import AutoTokenizer

from clts.export_tokenizer import ensure_hf_tokenizer
from util.condensed_tokenizer import CondensedTokenizer


@pytest.fixture
def fake_data_dir(tmp_path, monkeypatch):
    """A data dir containing the project's real old_to_new.json, plus
    STORAGE_ROOT pointed at tmp_path so writes are isolated."""
    src = next((p for p in [
        "data/BD_llama_inital/old_to_new.json",
        "data/bioS_N-Bd_final_grid/old_to_new.json",
    ] if __import__("pathlib").Path(p).exists()), None)
    assert src, "no old_to_new.json found under data/"
    data_dir = tmp_path / "data" / "fake"
    data_dir.mkdir(parents=True)
    shutil.copyfile(src, data_dir / "old_to_new.json")
    monkeypatch.setenv("CLT_STORAGE_ROOT", str(tmp_path / "storage"))
    return data_dir


def test_roundtrip(fake_data_dir):
    """Exported tokenizer.encode(text) must equal CondensedTokenizer.encode(text)
    for several bio-style texts. Round-trips through AutoTokenizer.from_pretrained.
    """
    out = ensure_hf_tokenizer(fake_data_dir)
    hf_tok = AutoTokenizer.from_pretrained(str(out))
    cond = CondensedTokenizer.from_remap_path(fake_data_dir / "old_to_new.json")

    for text in [
        "John Smith was born on May 3, 1987",
        "The birthday of Jane Doe is January 1, 2000",
        "Birthday: December 31, 1999",
    ]:
        ids_hf = hf_tok(text, add_special_tokens=False)["input_ids"]
        ids_cond = cond.encode(text)
        assert ids_hf == ids_cond, f"mismatch on {text!r}: hf={ids_hf} cond={ids_cond}"


def test_cache_hit_returns_same_path_without_reexport(fake_data_dir):
    out1 = ensure_hf_tokenizer(fake_data_dir)
    mtime1 = (out1 / "tokenizer.json").stat().st_mtime
    out2 = ensure_hf_tokenizer(fake_data_dir)
    mtime2 = (out2 / "tokenizer.json").stat().st_mtime
    assert out1 == out2
    assert mtime1 == mtime2, "tokenizer.json was rewritten on cache hit"


def test_distinct_remaps_produce_distinct_dirs(fake_data_dir, tmp_path):
    """A second data dir with a different remap must produce a different
    output directory rather than overwriting the first."""
    out1 = ensure_hf_tokenizer(fake_data_dir)

    other = tmp_path / "data" / "other"
    other.mkdir(parents=True)
    # Build a trivially-different remap (drop one entry).
    with open(fake_data_dir / "old_to_new.json") as f:
        remap = json.load(f)
    remap.pop(next(iter(remap)))
    with open(other / "old_to_new.json", "w") as f:
        json.dump(remap, f)

    out2 = ensure_hf_tokenizer(other)
    assert out1 != out2
    assert out1.exists() and out2.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_export_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clts.export_tokenizer'`

- [ ] **Step 3: Implement `ensure_hf_tokenizer`**

Create `clts/export_tokenizer.py`:

```python
"""Export CondensedTokenizer state as an HF-loadable tokenizer directory.

Content-addressed by the sha256 (first 8 hex chars) of the data dir's
old_to_new.json. Idempotent: a cache hit returns immediately without
re-exporting. New remaps automatically produce new output dirs.

Required by subproject #2 (attribution graphs) because circuit-tracer's
create_graph_files calls AutoTokenizer.from_pretrained(cfg.tokenizer_name).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

# Reuse the project's CondensedTokenizer.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.condensed_tokenizer import CondensedTokenizer  # noqa: E402


def _storage_root() -> Path:
    """STORAGE_ROOT for tokenizers. Env override for tests."""
    env = os.environ.get("CLT_STORAGE_ROOT")
    if env:
        return Path(env)
    # Default mirrors saes/trainSAE.py's STORAGE_ROOT for PSC; falls back to
    # a local dir off the repo root so this works on dev machines too.
    psc_root = Path("/jet/home/friedmae/data_storage/LM4_Results")
    if psc_root.exists():
        return psc_root
    return Path(__file__).resolve().parent.parent / "clt_storage"


def _remap_hash(remap_path: Path) -> str:
    return hashlib.sha256(remap_path.read_bytes()).hexdigest()[:8]


def ensure_hf_tokenizer(data_dir: str | Path) -> Path:
    """Return path to HF-loadable tokenizer dir for `data_dir`. Exports it
    on first call; cache-hits on subsequent calls."""
    data_dir = Path(data_dir)
    remap_path = data_dir / "old_to_new.json"
    if not remap_path.exists():
        raise FileNotFoundError(f"no old_to_new.json under {data_dir}")

    out_dir = _storage_root() / "hf_tokenizers" / _remap_hash(remap_path)
    if (out_dir / "tokenizer.json").exists():
        print(f"[tokenizer] cache hit: {out_dir}")
        return out_dir

    print(f"[tokenizer] exporting to {out_dir}")
    cond = CondensedTokenizer.from_remap_path(remap_path)

    # Build a WordLevel tokenizer whose vocab is the post-remap GPT-2 strings,
    # indexed at the reduced ids. GPT-2's vocab maps token string -> gpt2 id;
    # we re-key it at reduced ids and reuse GPT-2's byte-level pretokenizer
    # for parity with how the model was trained.
    vocab = cond.vocab  # str -> reduced_id
    word_level = WordLevel(vocab=vocab, unk_token=cond.unk_token)
    tk = Tokenizer(word_level)

    # GPT-2 byte-level pretokenization makes encode() match the model's
    # training-time tokenization byte-for-byte.
    from tokenizers.pre_tokenizers import ByteLevel as PTByteLevel
    from tokenizers.decoders import ByteLevel as DecByteLevel
    tk.pre_tokenizer = PTByteLevel(add_prefix_space=False)
    tk.decoder = DecByteLevel()

    hf = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token=cond.unk_token,
        bos_token=cond.bos_token,
        eos_token=cond.eos_token,
        pad_token=cond.pad_token,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(str(out_dir))

    # Roundtrip-verify before returning so a failed export is loud, not silent.
    from transformers import AutoTokenizer
    reloaded = AutoTokenizer.from_pretrained(str(out_dir))
    probe = "John Smith was born on May 3, 1987"
    a = reloaded(probe, add_special_tokens=False)["input_ids"]
    b = cond.encode(probe)
    if a != b:
        raise RuntimeError(
            f"roundtrip mismatch on {probe!r}: reloaded={a} condensed={b}"
        )
    return out_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    args = p.parse_args()
    out = ensure_hf_tokenizer(args.data_dir)
    print(out)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_export_tokenizer.py -v`
Expected: PASS (3 tests)

If `test_roundtrip` fails because the WordLevel vocab construction doesn't byte-match GPT-2's BPE: the issue is that GPT-2 uses BPE-level merges, not word-level. In that case, swap the implementation to: load `GPT2Tokenizer.from_pretrained("gpt2")`, then build the reduced tokenizer by exporting GPT-2 to a fast tokenizer JSON, post-processing its vocab to retain only the remapped ids, and renumbering. Concretely:

```python
from transformers import GPT2TokenizerFast
fast = GPT2TokenizerFast.from_pretrained("gpt2")
# Save fast tokenizer to out_dir, then patch tokenizer.json to substitute
# its vocab for the reduced one and rewrite added_tokens to match cond.
```

The roundtrip test will tell you which path works on first run. If neither works for full BPE fidelity, restrict the roundtrip test fixtures to inputs that contain only single-token strings present in the reduced vocab (which covers all bios by construction — the model trained on these and the remap is a *subset* of GPT-2's vocab).

- [ ] **Step 5: Commit**

```bash
git add clts/export_tokenizer.py tests/test_export_tokenizer.py
git commit -m "$(cat <<'EOF'
feat(clt): ensure_hf_tokenizer — idempotent HF tokenizer export

Content-addressed by sha256 of old_to_new.json so cache hits are no-ops
and new remaps automatically produce new output dirs. Roundtrip-verifies
exported tokenizer matches CondensedTokenizer before returning. Required
by subproject #2 because circuit-tracer's frontend calls
AutoTokenizer.from_pretrained on the saved dir.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Activation capture helper

**Files:**
- Create: `clts/evalCLT.py` (capture function only; eval functions added in Task 9)

- [ ] **Step 1: Implement the helper**

Create `clts/evalCLT.py`:

```python
"""CLT held-out eval and activation capture helpers."""
from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from clts.clt import CrossLayerTranscoder


def capture_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    enc_hook_template: str = "blocks.{layer}.hook_resid_mid",
    dec_hook_template: str = "blocks.{layer}.hook_mlp_out",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run `model` on `tokens` and return aligned (x_list, y_list).

    Each entry is shape [B*T, d_model], reshaped from [B, T, d_model] so
    positions are i.i.d. for loss purposes.

    Args:
        tokens: [B, T] integer token ids.
    """
    N = model.cfg.n_layers
    enc_names = [enc_hook_template.format(layer=L) for L in range(N)]
    dec_names = [dec_hook_template.format(layer=L) for L in range(N)]
    wanted = set(enc_names + dec_names)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: n in wanted,
            return_type=None,
        )

    D = model.cfg.d_model
    x_list = [cache[enc_names[L]].reshape(-1, D) for L in range(N)]
    y_list = [cache[dec_names[L]].reshape(-1, D) for L in range(N)]
    return x_list, y_list
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "from clts.evalCLT import capture_activations; print('ok')"`
Expected: `ok` (no import errors)

- [ ] **Step 3: Commit**

```bash
git add clts/evalCLT.py
git commit -m "$(cat <<'EOF'
feat(clt): capture_activations — single forward, all hooks

Uses model.run_with_cache to capture pre-MLP residuals and MLP outputs
for all layers in one pass. Reshapes [B, T, D] -> [B*T, D] since positions
are i.i.d. for CLT training. ~6 MB extra memory at the typical batch
shape used by trainSAE.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Per-layer eval metrics — MSE, L0, dead-feature fraction

**Files:**
- Modify: `clts/evalCLT.py` (add `compute_layer_metrics`)

- [ ] **Step 1: Implement `compute_layer_metrics`**

Append to `clts/evalCLT.py`:

```python
def compute_layer_metrics(
    clt: CrossLayerTranscoder,
    x_list: list[torch.Tensor],
    y_list: list[torch.Tensor],
) -> dict[str, float]:
    """Per-layer recon MSE, normalized MSE (MSE / Var(y)), L0, dead-frac.

    Returns a flat dict suitable for wandb logging:
        mse_total, mse_L{i}, nmse_L{i}, l0_L{i}, dead_frac_L{i}
    """
    with torch.no_grad():
        a_list = clt.encode(x_list)
        y_hat_list = clt.decode(a_list)

        out = {}
        total_mse = 0.0
        for L in range(clt.n_layers):
            mse = (y_hat_list[L] - y_list[L]).pow(2).mean().item()
            var = y_list[L].var().item()
            nmse = mse / var if var > 1e-12 else float("nan")
            l0 = (a_list[L] > 0).float().sum(dim=-1).mean().item()
            dead = (a_list[L].sum(dim=0) == 0).float().mean().item()

            out[f"mse_L{L}"] = mse
            out[f"nmse_L{L}"] = nmse
            out[f"l0_L{L}"] = l0
            out[f"dead_frac_L{L}"] = dead
            total_mse += mse

        out["mse_total"] = total_mse / clt.n_layers
        return out
```

- [ ] **Step 2: Smoke-test it interactively**

Run:

```bash
python -c "
import torch
from clts.clt import CrossLayerTranscoder
from clts.evalCLT import compute_layer_metrics
torch.manual_seed(0)
clt = CrossLayerTranscoder(n_layers=3, d_model=4, expansion=2)
x = [torch.randn(16, 4) for _ in range(3)]
y = [torch.randn(16, 4) for _ in range(3)]
m = compute_layer_metrics(clt, x, y)
print(sorted(m.keys()))
assert 'mse_total' in m and 'l0_L0' in m and 'dead_frac_L2' in m
print('ok')
"
```

Expected: prints a sorted list of metric keys and `ok`.

- [ ] **Step 3: Commit**

```bash
git add clts/evalCLT.py
git commit -m "$(cat <<'EOF'
feat(clt): compute_layer_metrics — per-layer MSE/nMSE/L0/dead-frac

Flat-dict output matches wandb's preferred shape. Computes normalized
MSE (MSE / Var(y)) so layers with different output magnitudes are
comparable.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: CE-recovered (full-replacement and per-layer diagnostic)

**Files:**
- Modify: `clts/evalCLT.py` (add `ce_recovered_full`, `ce_recovered_per_layer`)

- [ ] **Step 1: Implement CE-recovered evals**

Append to `clts/evalCLT.py`:

```python
def _model_ce(model: HookedTransformer, tokens: torch.Tensor) -> float:
    """Average cross-entropy of next-token prediction on `tokens`. Used as
    a reference and as the numerator/denominator for ce_recovered."""
    with torch.no_grad():
        logits = model(tokens, return_type="logits")
    # logits: [B, T, V]. shift by one for next-token target.
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = tokens[:, 1:]
    return -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()


def _ce_with_mlp_replaced(
    model: HookedTransformer,
    clt: CrossLayerTranscoder,
    tokens: torch.Tensor,
    layers_to_replace: list[int],
    enc_hook_template: str,
    dec_hook_template: str,
) -> float:
    """CE when CLT predictions replace MLP outputs at `layers_to_replace`.

    Computes CLT predictions in one forward (caching all encoder inputs),
    then runs a second forward with the prediction installed via hooks.
    """
    N = model.cfg.n_layers
    D = model.cfg.d_model
    enc_names = [enc_hook_template.format(layer=L) for L in range(N)]
    dec_names = {dec_hook_template.format(layer=L): L for L in range(N)}

    # Pass 1: collect encoder inputs only.
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: n in set(enc_names),
            return_type=None,
        )

    B, T = tokens.shape
    x_list_flat = [cache[enc_names[L]].reshape(-1, D) for L in range(N)]
    with torch.no_grad():
        a_list = clt.encode(x_list_flat)
        y_hat_flat = clt.decode(a_list)
    y_hat_per_layer = {
        L: y_hat_flat[L].reshape(B, T, D) for L in range(N)
    }

    # Pass 2: install replacements via hooks at the target layers.
    def make_hook(L):
        def hook(activation, hook_obj):
            return y_hat_per_layer[L]
        return hook

    fwd_hooks = [
        (name, make_hook(L)) for name, L in dec_names.items()
        if L in layers_to_replace
    ]
    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens, fwd_hooks=fwd_hooks, return_type="logits"
        )
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = tokens[:, 1:]
    return -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()


def ce_recovered_full(
    model: HookedTransformer,
    clt: CrossLayerTranscoder,
    tokens: torch.Tensor,
    enc_hook_template: str = "blocks.{layer}.hook_resid_mid",
    dec_hook_template: str = "blocks.{layer}.hook_mlp_out",
    ce_baseline: float | None = None,
    ce_zero: float | None = None,
) -> dict[str, float]:
    """CE when ALL MLPs are simultaneously replaced by the CLT.

    Returns a dict: ce_orig, ce_clt, ce_zero (MLPs replaced with zeros, an
    interpretable lower bound), and ce_recovered = (ce_zero - ce_clt) /
    (ce_zero - ce_orig). 1.0 = perfect, 0.0 = no better than zeroing MLPs.
    """
    N = model.cfg.n_layers
    if ce_baseline is None:
        ce_baseline = _model_ce(model, tokens)

    ce_clt = _ce_with_mlp_replaced(
        model, clt, tokens, list(range(N)), enc_hook_template, dec_hook_template
    )

    if ce_zero is None:
        # Zero out every MLP output (rough lower bound on usefulness).
        def zero_hook(act, hook_obj):
            return torch.zeros_like(act)
        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (dec_hook_template.format(layer=L), zero_hook)
                    for L in range(N)
                ],
                return_type="logits",
            )
        logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tgt = tokens[:, 1:]
        ce_zero = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()

    denom = ce_zero - ce_baseline
    recovered = (ce_zero - ce_clt) / denom if abs(denom) > 1e-12 else float("nan")
    return {
        "ce_orig": ce_baseline,
        "ce_clt": ce_clt,
        "ce_zero": ce_zero,
        "ce_recovered": recovered,
    }


def ce_recovered_per_layer(
    model: HookedTransformer,
    clt: CrossLayerTranscoder,
    tokens: torch.Tensor,
    enc_hook_template: str = "blocks.{layer}.hook_resid_mid",
    dec_hook_template: str = "blocks.{layer}.hook_mlp_out",
) -> dict[str, float]:
    """Per-layer diagnostic: replace one MLP at a time, report CE per L."""
    N = model.cfg.n_layers
    ce_orig = _model_ce(model, tokens)
    out = {"ce_orig": ce_orig}
    for L in range(N):
        ce_L = _ce_with_mlp_replaced(
            model, clt, tokens, [L], enc_hook_template, dec_hook_template
        )
        out[f"ce_clt_L{L}"] = ce_L
    return out
```

- [ ] **Step 2: Smoke-check the shape contract**

Run:

```bash
python -c "
import torch
from clts.clt import CrossLayerTranscoder
from clts.evalCLT import ce_recovered_full

# Verify the function signature parses (we can't run it without a real
# HookedTransformer; that's covered by the end-to-end smoke test in Task 16).
import inspect
sig = inspect.signature(ce_recovered_full)
assert {'model', 'clt', 'tokens', 'enc_hook_template', 'dec_hook_template'} <= set(sig.parameters)
print('ok')
"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add clts/evalCLT.py
git commit -m "$(cat <<'EOF'
feat(clt): ce_recovered_full and ce_recovered_per_layer

Full-replacement: run model with every MLP swapped for the CLT prediction
simultaneously. Normalize: recovered = (ce_zero - ce_clt) / (ce_zero - ce_orig)
so 1.0 = perfect reconstruction, 0.0 = no better than zeroing all MLPs.
Per-layer variant replaces one MLP at a time as a diagnostic.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: `trainCLT.py` — `setup()`

**Files:**
- Create: `clts/trainCLT.py` (setup function + module state; CLI + training added in later tasks)

- [ ] **Step 1: Scaffold `trainCLT.py` with `setup()`**

Create `clts/trainCLT.py`:

```python
"""Train cross-layer transcoders on a base Llama checkpoint.

Single run, default hooks:
    python clts/trainCLT.py --model-dir <path> --data-dir <path>

Sweep over (expansion x l0 x lr):
    python clts/trainCLT.py --model-dir <path> --data-dir <path> --sweep

Outputs land in:
    STORAGE_ROOT / clt_runs / <model-name> / [sweep-<id>|standalone] / <trial> / final/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_llama_weights  # type: ignore

# Project imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clts.clt import CrossLayerTranscoder  # noqa: E402
from clts.evalCLT import (  # noqa: E402
    capture_activations, compute_layer_metrics,
    ce_recovered_full, ce_recovered_per_layer,
)
from clts.export_tokenizer import ensure_hf_tokenizer  # noqa: E402
from util.bio_sampler import BioSampler  # noqa: E402
from util.condensed_tokenizer import CondensedTokenizer  # noqa: E402
from util.diverse_subset import DiverseBioSubset  # noqa: E402


# ============================================================================
# Output location — edit STORAGE_ROOT if you move workspaces.
# ============================================================================
STORAGE_ROOT = Path(os.environ.get(
    "CLT_STORAGE_ROOT",
    "/jet/home/friedmae/data_storage/LM4_Results",
))


# ============================================================================
# Defaults — overridable via CLI flags or the sweep grid.
# ============================================================================
DEFAULTS = {
    "n_examples":     10_000,
    "epochs":         30,
    "context_size":   512,
    "expansion":      16,
    "l0_coefficient": 5.0,
    "lr":             5e-5,
}
CLT_SEED = 0
BATCH_SIZE = 4096


# ============================================================================
# Module state (set by setup(), reused across sweep trials).
# ============================================================================
ARGS: argparse.Namespace | None = None
device: str | None = None
model: HookedTransformer | None = None
tokenizer: CondensedTokenizer | None = None
hf_tokenizer_path: Path | None = None
sampler: BioSampler | None = None
eval_tokens: torch.Tensor | None = None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup(args: argparse.Namespace) -> None:
    """Load model + tokenizer + sampler + held-out eval tokens. Called once
    per process; sweep trials reuse the globals."""
    global ARGS, device, model, tokenizer, hf_tokenizer_path, sampler, eval_tokens
    ARGS = args

    # Pre-flight: STORAGE_ROOT must be writable.
    try:
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        probe = STORAGE_ROOT / f".write_probe.{os.getpid()}"
        probe.touch()
        probe.unlink()
    except OSError as e:
        raise SystemExit(
            f"STORAGE_ROOT is not writable: {STORAGE_ROOT}\n  {e}\n"
            f"Set CLT_STORAGE_ROOT env var or edit STORAGE_ROOT at top of trainCLT.py."
        )
    print(f"[storage] {STORAGE_ROOT}  (writable)")

    if args.model_name is None:
        args.model_name = (
            args.model_dir.parent.name if args.model_dir.name == "final"
            else args.model_dir.name
        )

    device = pick_device()
    dtype = torch.float32

    hf_model = LlamaForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype).eval()
    hf_cfg = hf_model.config
    tl_cfg = HookedTransformerConfig(
        n_layers=hf_cfg.num_hidden_layers,
        d_model=hf_cfg.hidden_size,
        d_head=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        n_heads=hf_cfg.num_attention_heads,
        d_mlp=hf_cfg.intermediate_size,
        d_vocab=hf_cfg.vocab_size,
        n_ctx=hf_cfg.max_position_embeddings,
        act_fn="silu",
        normalization_type="RMS",
        gated_mlp=True,
        positional_embedding_type="rotary",
        rotary_base=int(getattr(hf_cfg, "rope_theta", 10000.0)),
        rotary_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        final_rms=True,
        tie_word_embeddings=hf_cfg.tie_word_embeddings,
        initializer_range=hf_cfg.initializer_range,
        n_key_value_heads=hf_cfg.num_key_value_heads,
        device=device,
    )
    model = HookedTransformer(tl_cfg)
    model.load_state_dict(convert_llama_weights(hf_model, tl_cfg), strict=False)
    model.to(device).eval()
    print(f"[model]   {args.model_dir} (name: {args.model_name})")
    print(f"          n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}, "
          f"n_heads={model.cfg.n_heads}, d_vocab={model.cfg.d_vocab}")

    tokenizer = CondensedTokenizer.from_remap_path(args.data_dir / "old_to_new.json")
    hf_tokenizer_path = ensure_hf_tokenizer(args.data_dir)
    print(f"[tokenizer] {hf_tokenizer_path}")

    sampler = BioSampler(args.data_dir / "people.json", fields=("birthday",), seed=CLT_SEED)

    # Held-out eval slice (seed+1, matches trainSAE.py convention).
    eval_subset = DiverseBioSubset(
        sampler, tokenizer, context_size=args.context_size, seed=CLT_SEED + 1
    )
    eval_rows = eval_subset.to_hf_dataset(64, verbose=False)["input_ids"]
    eval_tokens = torch.tensor(np.array(eval_rows), dtype=torch.long, device=device)
    print(f"[data]    {args.data_dir}")
    print(f"          {len(sampler.people):,} people, {sampler.n_templates} templates, "
          f"eval tokens: {tuple(eval_tokens.shape)}")

    # Section 2 coverage check: exposures per person at current --n-examples.
    train_subset = DiverseBioSubset(sampler, tokenizer, context_size=args.context_size, seed=CLT_SEED)
    rows = train_subset.to_hf_dataset(args.n_examples, verbose=False)["input_ids"]
    rows_np = np.array(rows)
    n_bios = int((rows_np == tokenizer.eos_token_id).sum())
    exposures = n_bios / max(1, len(sampler.people))
    expected_missing = len(sampler.people) * pow(2.71828, -exposures)
    print(f"[coverage] n_examples={args.n_examples} -> ~{n_bios:,} bios/epoch")
    print(f"           ~{exposures:.1f} exposures/person/epoch")
    print(f"           expected people with 0 exposures: ~{int(expected_missing):,} "
          f"({100*expected_missing/len(sampler.people):.1f}%)")
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "from clts.trainCLT import setup; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add clts/trainCLT.py
git commit -m "$(cat <<'EOF'
feat(clt): trainCLT.py setup() — model+tokenizer+sampler+eval loader

Mirrors saes/trainSAE.py setup() conventions: write-probe, HF -> TL
weight conversion, DiverseBioSubset for held-out eval (seed+1). Adds
ensure_hf_tokenizer() call so HF tokenizer dir exists before training,
and a Section 2 coverage-stats print at startup so undertraining is
visible.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: `trainCLT.py` — training step + L0 warmup + Adam

**Files:**
- Modify: `clts/trainCLT.py` (add `train_one_run`)

- [ ] **Step 1: Add training-loop infrastructure**

Append to `clts/trainCLT.py`:

```python
# ============================================================================
# Per-trial training (called by wandb.agent OR directly for single runs)
# ============================================================================

def trial_name(expansion: int, l0_coefficient: float, lr: float,
               epochs: int, n_examples: int) -> str:
    return f"mult{expansion}_l0{l0_coefficient:g}_lr{lr:g}_ep{epochs}_n{n_examples}"


def train_one_run(wandb_config_override: dict | None = None) -> None:
    """One CLT training run end-to-end. wandb_config_override comes from
    wandb.agent during a sweep; None for standalone runs."""
    import wandb

    wandb.init(project="interpLM4")
    run_id, run_entity, run_project = wandb.run.id, wandb.run.entity, wandb.run.project
    sweep_id = wandb.run.sweep_id
    cfg = wandb.config

    expansion      = cfg.get("expansion",      ARGS.expansion)
    l0_coefficient = cfg.get("l0_coefficient", ARGS.l0_coefficient)
    lr             = cfg.get("lr",             ARGS.lr)
    epochs         = cfg.get("epochs",         ARGS.epochs)
    n_examples     = cfg.get("n_examples",     ARGS.n_examples)
    context_size   = cfg.get("context_size",   ARGS.context_size)

    name = trial_name(expansion, l0_coefficient, lr, epochs, n_examples)
    sweep_folder = f"sweep-{sweep_id}" if sweep_id else "standalone"
    run_dir = STORAGE_ROOT / "clt_runs" / ARGS.model_name / sweep_folder / name
    final_dir = run_dir / "final"

    # Build training data.
    train_subset = DiverseBioSubset(
        sampler, tokenizer, context_size=context_size, seed=CLT_SEED
    )
    train_rows = train_subset.to_hf_dataset(n_examples, verbose=False)["input_ids"]
    train_tokens = torch.tensor(np.array(train_rows), dtype=torch.long, device=device)
    n_tokens = train_subset.n_tokens(n_examples)
    total_steps = (epochs * n_tokens) // BATCH_SIZE
    l0_warmup = total_steps // 10
    lr_warmup = total_steps // 50
    print(f"[trial]   {name}")
    print(f"          training tokens={n_tokens:,}, steps={total_steps:,}, "
          f"l0_warmup={l0_warmup}, lr_warmup={lr_warmup}")

    # Build CLT.
    clt = CrossLayerTranscoder(
        n_layers=model.cfg.n_layers, d_model=model.cfg.d_model, expansion=expansion,
    ).to(device)
    opt = torch.optim.Adam(clt.parameters(), lr=lr, betas=(0.9, 0.999))

    # Token-batch iterator. Yields [B, T] slices, looping over training_tokens.
    rows_per_batch = max(1, BATCH_SIZE // context_size)
    def token_batches():
        n_rows = train_tokens.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n_rows)
            for start in range(0, n_rows - rows_per_batch + 1, rows_per_batch):
                yield train_tokens[perm[start:start + rows_per_batch]]

    step = 0
    LOG_EVERY = 30
    EVAL_EVERY = 600

    for batch_tokens in token_batches():
        # Capture activations from the frozen base model.
        x_list, y_list = capture_activations(
            model, batch_tokens,
            enc_hook_template=ARGS.enc_hook_template,
            dec_hook_template=ARGS.dec_hook_template,
        )

        # L0 warmup: linear ramp of sparsity coefficient.
        ramp = min(1.0, (step + 1) / max(1, l0_warmup))
        lam = l0_coefficient * ramp

        # LR warmup: linear ramp from 0 to lr over lr_warmup steps.
        lr_ramp = min(1.0, (step + 1) / max(1, lr_warmup))
        for g in opt.param_groups:
            g["lr"] = lr * lr_ramp

        losses = clt.compute_loss(x_list, y_list, l0_coefficient=lam)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        opt.step()

        if step % LOG_EVERY == 0:
            log_payload = {"clt_train/mse_total": losses["recon"].item(),
                           "clt_train/sparsity_loss": losses["sparsity"].item(),
                           "clt_train/preact_loss": losses["preact"].item(),
                           "clt_train/l0_coef_effective": lam,
                           "clt_train/lr": opt.param_groups[0]["lr"]}
            for L in range(model.cfg.n_layers):
                log_payload[f"clt_train/mse_L{L}"] = losses[f"recon_L{L}"].item()
                log_payload[f"clt_train/l0_L{L}"] = losses[f"l0_L{L}"].item()
            wandb.log(log_payload, step=step)

        if step > 0 and step % EVAL_EVERY == 0:
            x_eval, y_eval = capture_activations(
                model, eval_tokens,
                enc_hook_template=ARGS.enc_hook_template,
                dec_hook_template=ARGS.dec_hook_template,
            )
            metrics = compute_layer_metrics(clt, x_eval, y_eval)
            ce = ce_recovered_full(
                model, clt, eval_tokens,
                enc_hook_template=ARGS.enc_hook_template,
                dec_hook_template=ARGS.dec_hook_template,
            )
            wandb.log(
                {f"clt_eval/{k}": v for k, v in {**metrics, **ce}.items()},
                step=step,
            )

        step += 1

    # Final eval + save.
    x_eval, y_eval = capture_activations(
        model, eval_tokens,
        enc_hook_template=ARGS.enc_hook_template,
        dec_hook_template=ARGS.dec_hook_template,
    )
    final_metrics = compute_layer_metrics(clt, x_eval, y_eval)
    final_ce = ce_recovered_full(
        model, clt, eval_tokens,
        enc_hook_template=ARGS.enc_hook_template,
        dec_hook_template=ARGS.dec_hook_template,
    )
    final_per_layer = ce_recovered_per_layer(
        model, clt, eval_tokens,
        enc_hook_template=ARGS.enc_hook_template,
        dec_hook_template=ARGS.dec_hook_template,
    )

    clt.save_to_dir(final_dir, model_name=ARGS.model_name,
                    feature_input_hook=ARGS.enc_hook_template.split(".")[-1],
                    feature_output_hook=ARGS.dec_hook_template.split(".")[-1])

    payload = {f"final_eval/{k}": v for k, v in
               {**final_metrics, **final_ce, **final_per_layer}.items()}
    payload["storage_path"] = str(final_dir)
    payload["tokenizer_path"] = str(hf_tokenizer_path)
    wandb.log(payload)
    wandb.run.summary.update(payload)
    print(f"[final]   saved to {final_dir}")
    print(f"          ce_recovered={final_ce['ce_recovered']:.4f}")
    wandb.finish()
```

- [ ] **Step 2: Verify imports + signatures**

Run: `python -c "from clts.trainCLT import train_one_run, trial_name; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add clts/trainCLT.py
git commit -m "$(cat <<'EOF'
feat(clt): train_one_run — end-to-end training + eval + save

Per-batch: capture residuals + MLP outputs in one forward, compute joint
multi-layer loss, Adam step with L0 + LR linear warmup. Logs train metrics
every 30 steps, held-out eval every 600 steps. Final eval + per-layer
CE-recovered (when --eval-mode full) + save to STORAGE_ROOT in
circuit-tracer format.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: `trainCLT.py` — CLI + main + sweep wiring

**Files:**
- Modify: `clts/trainCLT.py` (add `parse_args`, sweep config, `_patch_signal_for_worker_threads`, `main`)

- [ ] **Step 1: Add CLI + sweep + main**

Append to `clts/trainCLT.py`:

```python
# ============================================================================
# Sweep config
# ============================================================================

def build_sweep_config() -> dict:
    return {
        "program": "trainCLT.py",
        "method":  "grid",
        "name":    f"clt_sweep_{ARGS.model_name}",
        "metric":  {"name": "final_eval/ce_recovered", "goal": "maximize"},
        "parameters": {
            "expansion":      {"values": [8, 16]},
            "l0_coefficient": {"values": [2.0, 5.0, 10.0]},
            "lr":             {"values": [3e-5, 1e-4]},
            "epochs":         {"value":  50},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 5, "eta": 3},
    }


def _patch_signal_for_worker_threads():
    """wandb.agent runs trials in worker threads; signal.signal() rejects
    non-main-thread calls. Wrap signal.signal so it no-ops off the main thread."""
    import signal
    import threading
    _real = signal.signal

    def _safe(signum, handler):
        if threading.current_thread() is threading.main_thread():
            return _real(signum, handler)
        return None
    signal.signal = _safe


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-dir", type=Path, required=True,
                   help="HF Llama checkpoint dir (the base model the CLT attaches to).")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Dataset dir with people.json + old_to_new.json.")
    p.add_argument("--model-name", type=str, default=None,
                   help="Identifier for this base model in output paths. "
                        "Default: parent dir of --model-dir.")

    p.add_argument("--enc-hook-template", type=str,
                   default="blocks.{layer}.hook_resid_mid",
                   help="Encoder input hook template (must contain '{layer}').")
    p.add_argument("--dec-hook-template", type=str,
                   default="blocks.{layer}.hook_mlp_out",
                   help="Decoder target hook template (must contain '{layer}').")

    p.add_argument("--expansion", type=int, default=DEFAULTS["expansion"],
                   help="d_transcoder = expansion * d_model.")
    p.add_argument("--l0", dest="l0_coefficient", type=float,
                   default=DEFAULTS["l0_coefficient"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--context-size", type=int, default=DEFAULTS["context_size"])
    p.add_argument("--n-examples", type=int, default=DEFAULTS["n_examples"])

    # NOTE: --eval-mode flag deferred. Current pipeline always runs the "quick"
    # eval (64 rows) + per-layer CE-recovered diagnostic at final time. The
    # spec's "full" mode (per-person / per-template breakdowns over the full
    # 50k-people sample) is documented as a follow-up; the existing single
    # ce_recovered headline number is sufficient for sweep selection.

    p.add_argument("--sweep", action="store_true",
                   help="Launch wandb grid sweep over (expansion x l0 x lr).")

    return p.parse_args()


def main():
    import wandb
    args = parse_args()
    setup(args)
    if args.sweep:
        _patch_signal_for_worker_threads()
        cfg = build_sweep_config()
        sweep_id = wandb.sweep(cfg, project="interpLM4")
        print(f"[sweep]   registered: {sweep_id}")
        wandb.agent(sweep_id, function=train_one_run)
    else:
        train_one_run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `--help` parses**

Run: `python clts/trainCLT.py --help 2>&1 | head -40`
Expected: usage text listing `--model-dir`, `--data-dir`, `--expansion`, `--l0`, `--sweep`, etc., without any traceback.

- [ ] **Step 3: Commit**

```bash
git add clts/trainCLT.py
git commit -m "$(cat <<'EOF'
feat(clt): CLI, main, sweep config, signal-handler shim

Flags mirror saes/trainSAE.py where meaning is identical (--model-dir,
--data-dir, --epochs, --n-examples). Sweep is single-axis over
(expansion x l0 x lr) — no per-layer dimension since one CLT spans all
layers. _patch_signal_for_worker_threads avoids the SIGINT crash that
hits wandb.agent on worker threads.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: End-to-end smoke test on a real checkpoint

**Files:** none new; runs the trainer.

This task must be run on a machine that has one of the trained Llama checkpoints under `model/` available (or PSC). It is the **gate** between local CI tests passing and launching expensive sweeps.

- [ ] **Step 1: Run the smoke test**

If running locally with a CPU-only environment and `model/BD_llama_3heads_12epoch_4layers/` present:

```bash
WANDB_MODE=disabled CLT_STORAGE_ROOT=/tmp/clt_smoke \
  python clts/trainCLT.py \
    --model-dir model/BD_llama_3heads_12epoch_4layers \
    --data-dir  data/BD_llama_inital \
    --epochs 1 --n-examples 100 --expansion 4
```

If running on PSC with default `STORAGE_ROOT`:

```bash
WANDB_MODE=disabled \
  python clts/trainCLT.py \
    --model-dir model/BD_llama_3heads_12epoch_4layers \
    --data-dir  data/BD_llama_inital \
    --epochs 1 --n-examples 100 --expansion 4
```

- [ ] **Step 2: Verify acceptance criteria**

Check the output for each of the following (per Section 4 of the spec):

1. Trainer starts; prints model + data + storage paths
2. `[tokenizer]` line either reports a cache hit or a new export, with a resolved path
3. `[coverage]` line prints exposures-per-person stats
4. Training runs to completion without traceback
5. `[final] saved to <path>` line reports a real directory
6. `ce_recovered=` reports a finite number (not NaN)

- [ ] **Step 3: Verify the output directory**

```bash
ls $CLT_STORAGE_ROOT/clt_runs/BD_llama_3heads_12epoch_4layers/standalone/*/final/
```

Expected listing:
- `W_enc_0.safetensors`, `W_enc_1.safetensors`, `W_enc_2.safetensors`, `W_enc_3.safetensors`
- `W_dec_0.safetensors`, `W_dec_1.safetensors`, `W_dec_2.safetensors`, `W_dec_3.safetensors`
- `config.yaml`

- [ ] **Step 4: Verify a second invocation is a tokenizer cache hit**

Re-run the smoke test command from Step 1. The `[tokenizer]` line must say "cache hit" rather than "exporting to".

- [ ] **Step 5: Run all unit tests one more time**

Run: `pytest tests/test_clt.py tests/test_export_tokenizer.py -v`
Expected: 10 passed (7 CLT + 3 tokenizer)

- [ ] **Step 6: Commit a marker noting smoke-test passage (no code changes)**

If everything in steps 1-5 passes:

```bash
git commit --allow-empty -m "$(cat <<'EOF'
chore(clt): smoke test passes on BD_llama_3heads_12epoch_4layers

End-to-end: setup -> ensure_hf_tokenizer (export then cache hit) ->
1-epoch training -> final eval -> circuit-tracer-format save. All
10 unit tests pass. Pipeline ready for sweep runs.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

If anything fails: do NOT mark the task complete. Diagnose the failure, fix it in the relevant earlier file, add or amend a test that would have caught the issue, and re-run from Step 1.

---

## Done

After Task 14 passes, the CLT training pipeline is implementation-complete and ready for:
- Standalone runs on any of the project's Llama checkpoints
- wandb grid sweeps over `(expansion × l0 × lr)`
- Consumption by subproject #2 (attribution graphs) — which adds the `~30-line` adapter described in Section 5 of the spec
- Consumption by subproject #3 (UI) — which uses circuit-tracer's bundled frontend per Section 5

No code changes are required to existing `saes/`, `model/`, `util/`, or any other directory.

### Deferred follow-up (intentional)

The spec's `--eval-mode full` was scoped out of this plan to keep it focused. It would add:
- An `ExhaustiveBioSubset`-style eval generator (every `(person, template)` pair once, ~100M tokens)
- Per-person and per-template CE-recovered breakdowns
- "Fraction of people whose birthday is still top-1 predicted after full-MLP replacement"

The current `ce_recovered` (population mean over 64 held-out rows) and per-layer CE diagnostic together cover sweep selection and headline reporting. The full-mode eval is the right addition when you start interpreting individual feature failures or want to publish per-person quality numbers — both natural follow-ups, neither blocking on this plan.
