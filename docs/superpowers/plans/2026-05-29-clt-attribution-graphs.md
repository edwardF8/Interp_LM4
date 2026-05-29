# CLT Attribution Graphs & Circuit Exploration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Anthropic-style attribution graphs from the project's trained Cross-Layer Transcoders and explore the resulting circuits in `circuit-tracer`'s bundled viewer, starting with birthday-recall on `grid-L4-H6`.

**Architecture:** A thin adapter wires the custom-sized local Llama checkpoint + trained CLT into `circuit-tracer`'s `TransformerLensReplacementModel`; drivers call the library's unmodified `attribute()` / `create_graph_files()` / `serve()`. Feature dashboards are precomputed (PSC) and served locally (Mac). Fidelity is enforced by numeric equivalence checks, not by reimplementing the algorithm.

**Tech Stack:** Python 3.11, PyTorch, `transformer_lens`, `circuit-tracer` (decoderesearch fork v0.4.1, in a dedicated venv pinning `transformers<=4.57.3`), `safetensors`, `pydantic`, `pytest`.

**Spec:** [docs/superpowers/specs/2026-05-29-clt-attribution-graphs-design.md](../specs/2026-05-29-clt-attribution-graphs-design.md)

---

## File structure

| File | Responsibility |
|---|---|
| `clts/circuit_env/requirements.txt` | Pinned deps for the dedicated venv (committed). |
| `clts/circuit_env/README.md` | How to create the venv + the import gate. |
| `clts/feature_index.py` | `cantor_pair` / `cantor_unpair` — the viewer's feature-index encoding. |
| `clts/tl_model.py` | `build_tl_config` + `build_hooked_transformer` (extracted from `trainCLT`). |
| `clts/trainCLT.py` | **Modify** `setup()` to call `build_hooked_transformer` (behavior-preserving). |
| `clts/load_replacement_model.py` | The adapter → configured `TransformerLensReplacementModel`. |
| `clts/build_attribution_graph.py` | Driver: prompt → `attribute()` → `to_pt()` → `create_graph_files()` + error-share report. |
| `clts/gen_feature_dashboards.py` | PSC batch generator → per-feature `feature_models.Model` JSON. |
| `clts/serve_ui.py` | Thin wrapper over `circuit_tracer.frontend.local_server.serve`. |
| `clts/validate_graph.py` | Intervention-based graph validation (feature ablation vs attributed effect). |
| `tests/test_attribution.py` | All fidelity + format tests. |
| `scripts/sync_from_psc.sh` | **Modify**: add a `clt_features` bundle target. |
| `.gitignore` | **Modify**: ignore the venv + local graph outputs. |

**Conventions for every task:** run commands from the repo root `/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4`. Tests that need `circuit_tracer` or the real model run under the dedicated venv: `clts/.venv-ct/bin/python -m pytest ...`. Pure tests run under either interpreter. Mark model/circuit-tracer tests with `@pytest.mark.integration` so the fast suite can skip them.

---

## Task 0: Dedicated venv + circuit-tracer install + import gate

**Files:**
- Create: `clts/circuit_env/requirements.txt`
- Create: `clts/circuit_env/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Write the requirements file**

Create `clts/circuit_env/requirements.txt`:

```
# Dedicated venv for CLT attribution graphs (circuit-tracer).
# Isolated from the training/SAE env so its transformers pin cannot
# downgrade the rest of the project. Install circuit-tracer from a local
# clone (see README.md), which pulls these as resolved deps; pins below
# are the ones we depend on directly.
torch>=2.0.0
transformer-lens>=2.16.0
transformers>=4.56.0,<=4.57.3
safetensors>=0.5.0
pydantic>=2.0.0
pyyaml>=6.0
numpy>=1.24.0
pytest>=8.0.0
```

- [ ] **Step 2: Write the README with creation steps + import gate**

Create `clts/circuit_env/README.md`:

````markdown
# circuit-tracer venv

Isolated environment for building CLT attribution graphs. Never installed into
the training/SAE env (circuit-tracer pins `transformers<=4.57.3`).

## Create (Mac and PSC both)

```bash
python3.11 -m venv clts/.venv-ct
clts/.venv-ct/bin/pip install -U pip
clts/.venv-ct/bin/pip install -r clts/circuit_env/requirements.txt

# circuit-tracer from the decoderesearch fork:
git clone https://github.com/decoderesearch/circuit-tracer.git /tmp/circuit-tracer
clts/.venv-ct/bin/pip install /tmp/circuit-tracer
```

## Import gate (run after install; must print OK)

```bash
clts/.venv-ct/bin/python - <<'PY'
from transformer_lens.loading_from_pretrained import convert_llama_weights
from circuit_tracer.replacement_model.replacement_model_transformerlens import TransformerLensReplacementModel
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt
from circuit_tracer import attribute
from circuit_tracer.utils.create_graph_files import create_graph_files
from circuit_tracer.frontend.local_server import serve
from circuit_tracer.frontend.feature_models import Model as FeatureModel
print("OK")
PY
```
````

- [ ] **Step 3: Ignore the venv and local outputs**

Add to `.gitignore`:

```
clts/.venv-ct/
clts/clt_graphs/
clts/clt_features/
```

- [ ] **Step 4: Create the venv and run the import gate**

Run the three blocks from the README. Expected final output: `OK`.
If `convert_llama_weights` import fails, stop — the TL version is incompatible and the adapter cannot be built as designed.

- [ ] **Step 5: Commit**

```bash
git add clts/circuit_env/requirements.txt clts/circuit_env/README.md .gitignore
git commit -m "build(clt): dedicated circuit-tracer venv setup + import gate"
```

---

## Task 1: `clts/feature_index.py` — cantor pairing

**Files:**
- Create: `clts/feature_index.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_attribution.py`:

```python
"""Fidelity + format tests for CLT attribution graphs."""
from __future__ import annotations

import pytest


def test_cantor_roundtrip_matches_frontend():
    from clts.feature_index import cantor_pair, cantor_unpair

    # Frontend Node.feature_node: feature = (l+f)(l+f+1)//2 + f
    # Frontend cantorUnpair(z) -> [layer, feat]
    for layer in range(4):
        for feat in (0, 1, 5, 383, 6143):
            z = cantor_pair(layer, feat)
            assert cantor_unpair(z) == (layer, feat)

    # Spot-check the exact integer the frontend computes for (2, 100):
    assert cantor_pair(2, 100) == (2 + 100) * (2 + 100 + 1) // 2 + 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_attribution.py::test_cantor_roundtrip_matches_frontend -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clts.feature_index'`

- [ ] **Step 3: Write the implementation**

Create `clts/feature_index.py`:

```python
"""Feature-index encoding shared with circuit-tracer's frontend.

A CLT feature node is identified in the viewer by a single integer that is the
Cantor pairing of (layer, feature_index). circuit-tracer's `Node.feature_node`
encodes it as `(l+f)(l+f+1)//2 + f`; the frontend's `cantorUnpair` inverts it.
Feature-dashboard files are named `<cantor_pair(layer, feat)>.json`.
"""
from __future__ import annotations

import math


def cantor_pair(layer: int, feat_idx: int) -> int:
    s = layer + feat_idx
    return s * (s + 1) // 2 + feat_idx


def cantor_unpair(z: int) -> tuple[int, int]:
    w = (math.isqrt(8 * z + 1) - 1) // 2
    t = (w * w + w) // 2
    feat_idx = z - t
    layer = w - feat_idx
    return layer, feat_idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_attribution.py::test_cantor_roundtrip_matches_frontend -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clts/feature_index.py tests/test_attribution.py
git commit -m "feat(clt): cantor feature-index helpers matching circuit-tracer frontend"
```

---

## Task 2: `clts/tl_model.py` + refactor `trainCLT.setup()`

**Files:**
- Create: `clts/tl_model.py`
- Modify: `clts/trainCLT.py:108-137` (the inline model build inside `setup()`)
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing test (TL-vs-HF logit equivalence)**

Add to `tests/test_attribution.py`:

```python
MODEL_DIR = "model/grid-L4-H6"
CLT_DIR = "clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final"


@pytest.mark.integration
def test_build_hooked_transformer_matches_hf():
    import torch
    from transformers import LlamaForCausalLM
    from clts.tl_model import build_hooked_transformer

    tl = build_hooked_transformer(MODEL_DIR, device="cpu")
    hf = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
    ids = torch.tensor([[1835, 5, 10, 20, 30, 40]])  # ids < vocab (1836)
    with torch.no_grad():
        tl_logits = tl(ids, return_type="logits")
        hf_logits = hf(ids).logits
    assert tl_logits.shape == hf_logits.shape
    assert torch.allclose(tl_logits, hf_logits, atol=2e-3, rtol=2e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_build_hooked_transformer_matches_hf -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clts.tl_model'`

- [ ] **Step 3: Write `clts/tl_model.py`**

Create `clts/tl_model.py` (the config block is lifted verbatim from `trainCLT.py:112-131`):

```python
"""Build a TransformerLens HookedTransformer from a local HF Llama checkpoint.

Factored out of trainCLT.py so trainCLT, the replacement-model adapter, and the
feature-dashboard generator all build the model identically.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import LlamaForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_llama_weights  # type: ignore


def build_tl_config(hf_cfg, device: str,
                    dtype: torch.dtype = torch.float32) -> HookedTransformerConfig:
    """Translate an HF LlamaConfig into a HookedTransformerConfig."""
    return HookedTransformerConfig(
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
        dtype=dtype,
        device=device,
    )


def build_hooked_transformer(model_dir: str | Path, device: str,
                             dtype: torch.dtype = torch.float32,
                             tokenizer=None) -> HookedTransformer:
    """Load a local HF Llama checkpoint as a HookedTransformer."""
    hf_model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).eval()
    cfg = build_tl_config(hf_model.config, device, dtype)
    model = HookedTransformer(cfg, tokenizer=tokenizer)
    model.load_state_dict(convert_llama_weights(hf_model, cfg), strict=False)
    model.to(device).eval()
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_build_hooked_transformer_matches_hf -v`
Expected: PASS

- [ ] **Step 5: Refactor `trainCLT.setup()` to use the helper**

In `clts/trainCLT.py`, replace the inline build (currently `clts/trainCLT.py:108-134`, from `dtype = torch.float32` through `model.to(device).eval()` and its print) with:

```python
    dtype = torch.float32
    from clts.tl_model import build_hooked_transformer
    model = build_hooked_transformer(args.model_dir, device, dtype)
    print(f"[model]   {args.model_dir} (name: {args.model_name})")
    print(f"          n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}, "
          f"n_heads={model.cfg.n_heads}, d_vocab={model.cfg.d_vocab}")
```

Remove the now-unused imports at the top of `trainCLT.py` if they are no longer referenced elsewhere in the file: `LlamaForCausalLM`, `HookedTransformerConfig`, `convert_llama_weights`. Keep `HookedTransformer` only if still referenced (it is used as a type in the module globals annotation `model: HookedTransformer | None`); keep that import.

- [ ] **Step 6: Verify trainCLT still imports and the smoke path is intact**

Run: `python -c "import ast; ast.parse(open('clts/trainCLT.py').read()); print('parse OK')"`
Expected: `parse OK`

Run (under the training env, the one that has `sae_lens`/`wandb`): the trainCLT smoke command from the spec for 1 step is optional here; at minimum confirm the import graph:
Run: `clts/.venv-ct/bin/python -c "import sys; sys.path.insert(0,'.'); from clts.tl_model import build_hooked_transformer; print('import OK')"`
Expected: `import OK`

- [ ] **Step 7: Commit**

```bash
git add clts/tl_model.py clts/trainCLT.py tests/test_attribution.py
git commit -m "refactor(clt): extract HookedTransformer build into tl_model; reuse in trainCLT"
```

---

## Task 3: `clts/load_replacement_model.py` — the adapter

**Files:**
- Create: `clts/load_replacement_model.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing test — Check A (CLT compute equivalence) + Check B (logits match HF)**

Add to `tests/test_attribution.py`:

```python
SCAN_NAME = "grid-L4-H6"
DATA_DIR = "data/bioS_N-Bd_final_grid"


@pytest.mark.integration
def test_adapter_clt_compute_equivalence():
    """Check A: circuit-tracer's loaded CLT reconstructs identically to ours
    on real resid_mid activations from the assembled model."""
    import torch
    from clts.clt import CrossLayerTranscoder
    from clts.evalCLT import capture_activations
    from clts.export_tokenizer import ensure_hf_tokenizer
    from clts.load_replacement_model import load_replacement_model

    hf_tok = ensure_hf_tokenizer(DATA_DIR)
    model = load_replacement_model(MODEL_DIR, CLT_DIR, hf_tok, SCAN_NAME, device="cpu")
    ours = CrossLayerTranscoder.load_from_dir(CLT_DIR)

    ids = torch.tensor([[1835, 5, 10, 20, 30, 40, 50, 60]])
    x_list, _ = capture_activations(model, ids)            # list[N] of [B*T, D]
    N = ours.n_layers
    # ours: list-of-layers; theirs: stacked [N, n_pos, D]
    a_ours = ours.encode(x_list)                           # list[N] of [n, d_t]
    x_stacked = torch.stack(x_list)                        # [N, n, D]
    a_theirs = model.transcoders.encode(x_stacked)         # [N, n, d_t]
    for L in range(N):
        assert torch.allclose(a_ours[L], a_theirs[L], atol=1e-4), f"encode layer {L}"

    yhat_ours = torch.stack(ours.decode(a_ours))           # [N, n, D]
    recon_theirs = model.transcoders.decode(a_theirs.to_sparse())  # [N, n, D]
    assert torch.allclose(yhat_ours, recon_theirs, atol=1e-4)


@pytest.mark.integration
def test_adapter_logits_match_hf():
    """Check B: the assembled replacement model's logits equal the HF model's
    (proves weight loading + config are correct after configuration)."""
    import torch
    from transformers import LlamaForCausalLM
    from clts.export_tokenizer import ensure_hf_tokenizer
    from clts.load_replacement_model import load_replacement_model

    hf_tok = ensure_hf_tokenizer(DATA_DIR)
    model = load_replacement_model(MODEL_DIR, CLT_DIR, hf_tok, SCAN_NAME, device="cpu")
    hf = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
    ids = torch.tensor([[1835, 7, 11, 21, 31]])
    with torch.no_grad():
        tl_logits = model(ids, return_type="logits")
        hf_logits = hf(ids).logits
    assert torch.allclose(tl_logits, hf_logits, atol=2e-3, rtol=2e-3)


@pytest.mark.integration
def test_adapter_sets_cfg_metadata():
    from clts.export_tokenizer import ensure_hf_tokenizer
    from clts.load_replacement_model import load_replacement_model

    hf_tok = ensure_hf_tokenizer(DATA_DIR)
    model = load_replacement_model(MODEL_DIR, CLT_DIR, hf_tok, SCAN_NAME, device="cpu")
    assert model.cfg.model_name == SCAN_NAME
    assert str(hf_tok) == model.cfg.tokenizer_name
    assert model.scan_name == SCAN_NAME
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py -k adapter -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clts.load_replacement_model'`

- [ ] **Step 3: Write the adapter**

Create `clts/load_replacement_model.py`:

```python
"""Adapter: wire a local custom-sized Llama + a trained CLT into circuit-tracer.

circuit-tracer's ReplacementModel.from_pretrained derives the config from a
known HF alias, which does not exist for our custom dims (4 layers, d=384,
vocab=1836). We instead build the HookedTransformerConfig ourselves (tl_model),
construct the replacement subclass directly, load our weights BEFORE
configuration (which renames mlp/unembed keys), and load the CLT with
circuit-tracer's own loader so its attribution methods are available.
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, LlamaForCausalLM
from transformer_lens.loading_from_pretrained import convert_llama_weights  # type: ignore
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clts.tl_model import build_tl_config  # noqa: E402


def _read_hooks(clt_dir: Path) -> tuple[str, str]:
    cfg_path = clt_dir / "config.yaml"
    if not cfg_path.exists():
        return "hook_resid_mid", "hook_mlp_out"
    cfg = yaml.safe_load(cfg_path.read_text())
    return (cfg.get("feature_input_hook", "hook_resid_mid"),
            cfg.get("feature_output_hook", "hook_mlp_out"))


def load_replacement_model(
    model_dir: str | Path,
    clt_dir: str | Path,
    hf_tokenizer_dir: str | Path,
    scan_name: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TransformerLensReplacementModel:
    """Return a configured TransformerLensReplacementModel ready for attribute().

    Args:
        model_dir: local HF Llama checkpoint dir (the CLT's base model).
        clt_dir: dir of trained CLT safetensors + config.yaml.
        hf_tokenizer_dir: AutoTokenizer.from_pretrained-loadable dir
            (clts/export_tokenizer.ensure_hf_tokenizer).
        scan_name: stable CLT identifier; must equal the feature-dashboard
            subfolder name so the viewer finds dashboards.
    """
    model_dir, clt_dir = Path(model_dir), Path(clt_dir)

    hf_model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(hf_tokenizer_dir))

    cfg = build_tl_config(hf_model.config, device, dtype)
    cfg.model_name = scan_name                      # invariant 2: non-None
    cfg.tokenizer_name = str(hf_tokenizer_dir)      # invariant 3: graph-file step needs it
    cfg.dtype = dtype

    # invariant 1: load weights BEFORE _configure_replacement_model renames keys
    model = TransformerLensReplacementModel(cfg, tokenizer=tokenizer)
    model.load_state_dict(convert_llama_weights(hf_model, cfg), strict=False)
    model = model.to(device=device, dtype=dtype)

    enc_hook, dec_hook = _read_hooks(clt_dir)        # invariant 4: trained hooks
    clt = load_clt(
        str(clt_dir),
        feature_input_hook=enc_hook,
        feature_output_hook=dec_hook,
        scan_name=scan_name,                         # invariant 7
        device=torch.device(device),
        dtype=dtype,                                 # invariant 5: fp32, not bf16
        lazy_decoder=False,
        lazy_encoder=False,
    )
    model._configure_replacement_model(clt)
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py -k adapter -v`
Expected: PASS (3 tests)

If `TransformerLensReplacementModel(cfg, tokenizer=tokenizer)` raises on the `tokenizer` kwarg, change construction to:
```python
    model = TransformerLensReplacementModel(cfg)
    model.set_tokenizer(tokenizer)
```
and re-run.

- [ ] **Step 5: Commit**

```bash
git add clts/load_replacement_model.py tests/test_attribution.py
git commit -m "feat(clt): replacement-model adapter + fidelity equivalence checks A/B"
```

---

## Task 4: `clts/build_attribution_graph.py` — graph driver + error-share report

**Files:**
- Create: `clts/build_attribution_graph.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing test (feature nodes invert in-range; report shape)**

Add to `tests/test_attribution.py`:

```python
@pytest.mark.integration
def test_build_graph_birthday_recall(tmp_path):
    from clts.build_attribution_graph import build_graph
    from clts.feature_index import cantor_unpair

    out = build_graph(
        model_dir=MODEL_DIR, clt_dir=CLT_DIR, data_dir=DATA_DIR,
        scan_name=SCAN_NAME, prompt=None, device="cpu",
        graph_dir=str(tmp_path), slug="test-bday",
        max_feature_nodes=512,
    )
    # Graph + frontend files written
    assert (tmp_path / "test-bday.json").exists()
    assert (tmp_path / "graph-metadata.json").exists()
    assert out["pt_path"]

    # Every CLT feature node inverts to in-range (layer, feat)
    import torch
    from clts.clt import CrossLayerTranscoder
    clt = CrossLayerTranscoder.load_from_dir(CLT_DIR)
    graph = out["graph"]
    for row in graph.active_features[graph.selected_features].tolist():
        layer, _pos, feat = row
        z = (layer + feat) * (layer + feat + 1) // 2 + feat
        assert cantor_unpair(z) == (layer, feat)
        assert 0 <= layer < clt.n_layers and 0 <= feat < clt.d_transcoder

    # Report is well-formed
    r = out["report"]
    assert 0.0 <= r["error_influence_share"] <= 1.0
    assert r["target_logit_prob"] >= 0.0
    assert r["n_feature_nodes_pruned"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_build_graph_birthday_recall -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clts.build_attribution_graph'`

- [ ] **Step 3: Write the driver**

Create `clts/build_attribution_graph.py`:

```python
"""Build a CLT attribution graph for a prompt and write viewer files.

Uses circuit-tracer's attribute() UNMODIFIED (the canonical algorithm). Adds a
per-graph fidelity report: target-logit probability, the error-node influence
share, and the pruned feature-node count, so an under-explained graph (large
error nodes — a weak-CLT symptom) is surfaced rather than shipped silently.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clts.export_tokenizer import ensure_hf_tokenizer  # noqa: E402
from clts.load_replacement_model import load_replacement_model  # noqa: E402
from util.bio_sampler import BioSampler  # noqa: E402

STORAGE_ROOT = Path(__file__).resolve().parent / "clt_graphs"


def default_birthday_prompt(data_dir: str | Path) -> str:
    """A '<Name> was born on' prompt for a real person in people.json."""
    sampler = BioSampler(Path(data_dir) / "people.json", fields=("birthday",), seed=0)
    p = sampler.people[0]
    name = f"{p['first_name']} {p['last_name']}"
    return f"{name} was born on"


def _error_influence_share(graph) -> float:
    """Fraction of total node influence carried by MLP-reconstruction error
    nodes. Node order: [features, errors, tokens, logits]."""
    from circuit_tracer.graph import prune_graph

    node_mask, edge_mask, cumulative_scores = (el.cpu() for el in prune_graph(graph))
    n_features = len(graph.selected_features)
    layers = graph.cfg.n_layers
    error_end = n_features + graph.n_pos * layers
    token_end = error_end + len(graph.input_tokens)

    scores = cumulative_scores.clamp(min=0)
    total = scores[:token_end].sum().item()           # features + errors + tokens
    err = scores[n_features:error_end].sum().item()
    n_pruned = int(node_mask[:n_features].sum().item())
    return {
        "error_influence_share": (err / total) if total > 1e-12 else float("nan"),
        "n_feature_nodes_pruned": n_pruned,
    }


def build_graph(model_dir, clt_dir, data_dir, scan_name, graph_dir, slug,
                prompt=None, target=None, device="cpu",
                max_n_logits=10, desired_logit_prob=0.95,
                max_feature_nodes=4096, batch_size=256, verbose=True):
    from circuit_tracer import attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files

    hf_tok = ensure_hf_tokenizer(data_dir)
    model = load_replacement_model(model_dir, clt_dir, hf_tok, scan_name, device=device)

    if prompt is None:
        prompt = default_birthday_prompt(data_dir)

    graph = attribute(
        prompt=prompt, model=model,
        attribution_targets=([target] if target else None),
        max_n_logits=max_n_logits, desired_logit_prob=desired_logit_prob,
        batch_size=batch_size, max_feature_nodes=max_feature_nodes,
        offload=None, verbose=verbose,
    )

    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    pt_path = graph_dir / f"{slug}.pt"
    graph.to_pt(str(pt_path))
    create_graph_files(graph_or_path=graph, slug=slug, output_path=str(graph_dir),
                       scan_name=scan_name, node_threshold=0.8, edge_threshold=0.98)

    top_token = model.tokenizer.decode(graph.logit_tokens()[0].item())
    report = {
        "prompt": prompt,
        "scan_name": scan_name,
        "top_logit_token": top_token,
        "target_logit_prob": float(graph.logit_probabilities[0].item()),
        **_error_influence_share(graph),
    }
    (graph_dir / f"{slug}.report.json").write_text(json.dumps(report, indent=2))
    if verbose:
        print(json.dumps(report, indent=2))
    return {"graph": graph, "pt_path": str(pt_path), "report": report}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--clt-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scan-name", default=None,
                    help="Defaults to the CLT config.yaml model_name.")
    ap.add_argument("--prompt", default=None, help="Defaults to a birthday-recall prompt.")
    ap.add_argument("--target", default=None, help="Target token string; default auto-selects.")
    ap.add_argument("--slug", default="bday-recall")
    ap.add_argument("--graph-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-feature-nodes", type=int, default=4096)
    args = ap.parse_args()

    import yaml
    scan = args.scan_name or yaml.safe_load(
        (Path(args.clt_dir) / "config.yaml").read_text())["model_name"]
    graph_dir = args.graph_dir or str(STORAGE_ROOT / scan / args.slug)

    build_graph(args.model_dir, args.clt_dir, args.data_dir, scan, graph_dir,
                args.slug, prompt=args.prompt, target=args.target,
                device=args.device, max_feature_nodes=args.max_feature_nodes)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_build_graph_birthday_recall -v`
Expected: PASS. If `attribute()` errors on CPU for any op, re-run with `device="mps"` is NOT advised (sparse-op gaps); instead reduce `max_feature_nodes` — the failure to watch for is OOM/time, not correctness.

- [ ] **Step 5: Commit**

```bash
git add clts/build_attribution_graph.py tests/test_attribution.py
git commit -m "feat(clt): attribution-graph driver + per-graph error-share report"
```

---

## Task 5: `clts/serve_ui.py` — local viewer

**Files:**
- Create: `clts/serve_ui.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing test (server starts and stops, serves a feature file)**

Add to `tests/test_attribution.py`:

```python
@pytest.mark.integration
def test_serve_ui_starts_and_serves_features(tmp_path):
    import json
    import urllib.request
    from clts.serve_ui import start_server

    # minimal feature dir: features/<scan>/<idx>.json
    feats = tmp_path / "features"
    (feats / SCAN_NAME).mkdir(parents=True)
    (feats / SCAN_NAME / "5.json").write_text(json.dumps({"index": 5}))
    (tmp_path / "graphs").mkdir()

    server = start_server(graph_dir=str(tmp_path / "graphs"),
                          features_dir=str(feats), port=8047)
    try:
        with urllib.request.urlopen("http://localhost:8047/features/"
                                    f"{SCAN_NAME}/5.json", timeout=5) as r:
            body = json.loads(r.read())
        assert body["index"] == 5
    finally:
        server.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_serve_ui_starts_and_serves_features -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clts.serve_ui'`

- [ ] **Step 3: Write the server wrapper**

Create `clts/serve_ui.py`:

```python
"""Serve circuit-tracer's bundled viewer over local graph + feature files."""
from __future__ import annotations

import argparse
import time


def start_server(graph_dir: str, features_dir: str | None = None, port: int = 8032):
    """Start the viewer server in a background thread; returns a handle with .stop()."""
    from circuit_tracer.frontend.local_server import serve
    return serve(data_dir=graph_dir, port=port, features_dir=features_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--port", type=int, default=8032)
    args = ap.parse_args()

    server = start_server(args.graph_dir, args.features_dir, args.port)
    print(f"Serving at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_serve_ui_starts_and_serves_features -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clts/serve_ui.py tests/test_attribution.py
git commit -m "feat(clt): local viewer server wrapper"
```

---

## Task 6: `clts/gen_feature_dashboards.py` — PSC feature-dashboard generator

Decomposed into helpers, each tested. The generator runs in the dedicated venv (it imports `feature_models.Model` for guaranteed schema match) and reuses `saes.sae_explorer.build_index_corpus` for the corpus.

**Files:**
- Create: `clts/gen_feature_dashboards.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing test for `decoder_logit_effects`**

Add to `tests/test_attribution.py`:

```python
def test_decoder_logit_effects_shape_and_values():
    import torch
    from clts.gen_feature_dashboards import decoder_logit_effects

    class FakeCLT:
        n_layers, d_transcoder, d_model = 2, 3, 4
        def __init__(self):
            # W_dec[L]: [d_t, n_layers - L, d_model]
            self.W_dec = [torch.zeros(3, 2, 4), torch.zeros(3, 1, 4)]
            self.W_dec[0][0, 0, :] = torch.tensor([1., 0., 0., 0.])  # L0 feat0 -> layer0
            self.W_dec[0][0, 1, :] = torch.tensor([0., 1., 0., 0.])  # L0 feat0 -> layer1

    W_U = torch.eye(4)[:, :5] * 0 + torch.randn(4, 5)
    eff = decoder_logit_effects(FakeCLT(), W_U)
    assert eff.shape == (2, 3, 5)  # [n_layers, d_t, vocab]
    # feature (L0, f0): summed decoder = [1,1,0,0] -> @ W_U
    expected = torch.tensor([1., 1., 0., 0.]) @ W_U
    assert torch.allclose(eff[0, 0], expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_decoder_logit_effects_shape_and_values -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clts.gen_feature_dashboards'`

- [ ] **Step 3: Write `decoder_logit_effects` (and module skeleton)**

Create `clts/gen_feature_dashboards.py`:

```python
"""Generate per-feature dashboards for the circuit-tracer viewer (run on PSC).

For every CLT feature we compute, over a bios corpus:
  - top-activating examples bucketed by activation quantile (tokens + per-token
    activations + argmax position),
  - an activation histogram, act_min/act_max, activation_frequency,
  - top/bottom logits via the summed decoder direction through the unembed.

Output: features_dir/<scan_name>/<cantor_pair(layer, feat)>.json conforming to
circuit_tracer.frontend.feature_models.Model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clts.clt import CrossLayerTranscoder  # noqa: E402
from clts.feature_index import cantor_pair  # noqa: E402
from clts.evalCLT import capture_activations  # noqa: E402
from clts.tl_model import build_hooked_transformer  # noqa: E402
from util.condensed_tokenizer import CondensedTokenizer  # noqa: E402
from util.bio_sampler import BioSampler  # noqa: E402


def decoder_logit_effects(clt, W_U: torch.Tensor) -> torch.Tensor:
    """[n_layers, d_transcoder, vocab] logit effect of each feature.

    A feature at layer L writes to layers L..N-1; its net residual direction is
    the sum of its decoder columns across those target layers. Project through
    the unembed W_U ([d_model, vocab]) to get its additive logit effect.
    """
    effects = []
    for L in range(clt.n_layers):
        summed_dec = clt.W_dec[L].sum(dim=1)        # [d_t, d_model]
        effects.append(summed_dec @ W_U)            # [d_t, vocab]
    return torch.stack(effects)                     # [n_layers, d_t, vocab]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_decoder_logit_effects_shape_and_values -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for `build_feature_model` (schema + filename)**

Add to `tests/test_attribution.py`:

```python
def test_build_feature_model_validates_schema():
    from circuit_tracer.frontend.feature_models import Model as FeatureModel
    from clts.gen_feature_dashboards import build_feature_model
    from clts.feature_index import cantor_pair

    # synthetic per-feature inputs
    examples = [
        {"tokens": ["a", "b", "c"], "acts": [0.0, 1.2, 0.3], "argmax": 1},
        {"tokens": ["x", "y"], "acts": [0.9, 0.0], "argmax": 0},
    ]
    m = build_feature_model(
        layer=2, feat_idx=100, examples=examples,
        act_min=0.0, act_max=1.2, histogram=[3.0, 1.0, 2.0],
        quantile_values=[0.0, 0.6, 1.2], activation_frequency=0.05,
        top_logits=["Jan", "Feb"], bottom_logits=["zzz", "qqq"],
    )
    obj = FeatureModel.model_validate(m)             # raises if schema-invalid
    assert obj.index == cantor_pair(2, 100)
    assert obj.examples_quantiles[0].examples[0].tokens == ["a", "b", "c"]
```

- [ ] **Step 6: Run test to verify it fails**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_build_feature_model_validates_schema -v`
Expected: FAIL with `ImportError: cannot import name 'build_feature_model'`

- [ ] **Step 7: Write `build_feature_model`**

Append to `clts/gen_feature_dashboards.py`:

```python
def build_feature_model(layer, feat_idx, examples, act_min, act_max, histogram,
                        quantile_values, activation_frequency,
                        top_logits, bottom_logits, n_quantiles=5):
    """Assemble a feature_models.Model-shaped dict.

    examples: list of {tokens: list[str], acts: list[float], argmax: int},
    pre-sorted descending by max activation.
    """
    idx = cantor_pair(layer, feat_idx)

    def to_example(e):
        return {
            "tokens_acts_list": [float(a) for a in e["acts"]],
            "train_token_ind": int(e["argmax"]),
            "is_repeated_datapoint": False,
            "tokens": list(e["tokens"]),
        }

    # Bucket examples into n_quantiles bands over [act_min, act_max] by their max act.
    span = max(act_max - act_min, 1e-9)
    buckets = [[] for _ in range(n_quantiles)]
    for e in examples:
        m = max(e["acts"]) if e["acts"] else act_min
        b = min(n_quantiles - 1, int((m - act_min) / span * n_quantiles))
        buckets[b].append(to_example(e))

    examples_quantiles = []
    for b in range(n_quantiles - 1, -1, -1):         # high → low
        lo = act_min + span * b / n_quantiles
        hi = act_min + span * (b + 1) / n_quantiles
        examples_quantiles.append({
            "quantile_name": f"{lo:.3f}-{hi:.3f}",
            "examples": buckets[b],
        })

    return {
        "transcoder_id": str(layer),
        "index": idx,
        "examples_quantiles": examples_quantiles,
        "top_logits": list(top_logits),
        "bottom_logits": list(bottom_logits),
        "act_min": float(act_min),
        "act_max": float(act_max),
        "quantile_values": [float(q) for q in quantile_values],
        "histogram": [float(h) for h in histogram],
        "activation_frequency": float(activation_frequency),
    }
```

- [ ] **Step 8: Run test to verify it passes**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_build_feature_model_validates_schema -v`
Expected: PASS

- [ ] **Step 9: Write the failing end-to-end test (tiny corpus → files written + parse)**

Add to `tests/test_attribution.py`:

```python
@pytest.mark.integration
def test_generate_dashboards_small(tmp_path):
    import glob
    from circuit_tracer.frontend.feature_models import Model as FeatureModel
    from clts.gen_feature_dashboards import generate_dashboards

    out = generate_dashboards(
        model_dir=MODEL_DIR, clt_dir=CLT_DIR, data_dir=DATA_DIR,
        scan_name=SCAN_NAME, features_root=str(tmp_path),
        device="cpu", n_per_person=1, context_size=32, n_people=8,
        top_k=5, n_bins=10,
    )
    written = glob.glob(str(tmp_path / SCAN_NAME / "*.json"))
    assert len(written) > 0
    # every written file parses against the schema
    import json
    for p in written[:20]:
        FeatureModel.model_validate(json.loads(open(p).read()))
    assert out["n_features_written"] == len(written)
```

- [ ] **Step 10: Run test to verify it fails**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_generate_dashboards_small -v`
Expected: FAIL with `ImportError: cannot import name 'generate_dashboards'`

- [ ] **Step 11: Write `generate_dashboards` (two-pass accumulation + writer + CLI)**

Append to `clts/gen_feature_dashboards.py`:

```python
def _build_corpus(sampler, tokenizer, n_per_person, context_size, n_people, device):
    """Small, deterministic corpus tensor [rows, context_size] of token ids.

    Reuses the project's bio sampling; takes the first `n_people` people so the
    test corpus is tiny and reproducible. For full PSC runs pass a large
    n_people (or None for all)."""
    import numpy as np
    from util.diverse_subset import DiverseBioSubset
    subset = DiverseBioSubset(sampler, tokenizer, context_size=context_size, seed=0)
    n_examples = (n_people or len(sampler.people)) * n_per_person
    rows = subset.to_hf_dataset(n_examples, verbose=False)["input_ids"]
    return torch.tensor(np.array(rows), dtype=torch.long, device=device)


def generate_dashboards(model_dir, clt_dir, data_dir, scan_name, features_root,
                        device="cpu", n_per_person=2, context_size=64,
                        n_people=None, top_k=20, n_bins=40, batch_rows=8):
    from transformers import AutoTokenizer
    from clts.export_tokenizer import ensure_hf_tokenizer

    out_dir = Path(features_root) / scan_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_hooked_transformer(model_dir, device)
    clt = CrossLayerTranscoder.load_from_dir(clt_dir).to(device)
    cond = CondensedTokenizer.from_remap_path(Path(data_dir) / "old_to_new.json")
    hf_tok = AutoTokenizer.from_pretrained(str(ensure_hf_tokenizer(data_dir)))
    sampler = BioSampler(Path(data_dir) / "people.json", fields=("birthday",), seed=0)
    tokens = _build_corpus(sampler, cond, n_per_person, context_size, n_people, device)

    N, d_t = clt.n_layers, clt.d_transcoder
    W_U = model.W_U.detach().to(device)                       # [d_model, vocab]
    effects = decoder_logit_effects(clt, W_U)                 # [N, d_t, vocab]

    # --- Pass 1: per-feature act_max and firing count ---
    act_max = torch.zeros(N, d_t, device=device)
    fire_count = torch.zeros(N, d_t, device=device)
    total_pos = 0
    for s in range(0, tokens.shape[0], batch_rows):
        batch = tokens[s:s + batch_rows]
        x_list, _ = capture_activations(model, batch)         # list[N] [B*T, D]
        total_pos += x_list[0].shape[0]
        a = clt.encode(x_list)                                # list[N] [B*T, d_t]
        for L in range(N):
            act_max[L] = torch.maximum(act_max[L], a[L].max(dim=0).values)
            fire_count[L] += (a[L] > 0).sum(dim=0)
    activation_frequency = (fire_count / max(total_pos, 1)).cpu()

    # --- Pass 2: histograms + top-k examples per feature ---
    hist = torch.zeros(N, d_t, n_bins, device=device)
    # top-k tracked on CPU as lists of (max_act, example_dict)
    top_examples = [[[] for _ in range(d_t)] for _ in range(N)]
    rows_per = tokens.shape[1]
    for s in range(0, tokens.shape[0], batch_rows):
        batch = tokens[s:s + batch_rows]
        x_list, _ = capture_activations(model, batch)
        a = clt.encode(x_list)                                # list[N] [B*T, d_t]
        B = batch.shape[0]
        for L in range(N):
            aL = a[L].reshape(B, rows_per, d_t)               # [B, T, d_t]
            # histogram (only > 0 contributions)
            for b in range(n_bins):
                lo = act_max[L] * b / n_bins
                hi = act_max[L] * (b + 1) / n_bins
                mask = (aL > lo.view(1, 1, -1)) & (aL <= hi.view(1, 1, -1))
                hist[L, :, b] += mask.sum(dim=(0, 1)).float()
            # per-row max activation per feature, to pick example rows
            row_max, row_argmax = aL.max(dim=1)               # [B, d_t]
            fired_feats = (row_max > 0).any(dim=0).nonzero().flatten().tolist()
            for f in fired_feats:
                for bi in range(B):
                    m = float(row_max[bi, f])
                    if m <= 0:
                        continue
                    lst = top_examples[L][f]
                    if len(lst) < top_k or m > lst[-1][0]:
                        ids = batch[bi].tolist()
                        toks = [cond.decode([t]) for t in ids]
                        ex = {"tokens": toks,
                              "acts": aL[bi, :, f].tolist(),
                              "argmax": int(row_argmax[bi, f])}
                        lst.append((m, ex))
                        lst.sort(key=lambda z: z[0], reverse=True)
                        del lst[top_k:]

    # --- Write one JSON per feature that fired ---
    n_written = 0
    for L in range(N):
        for f in range(d_t):
            if not top_examples[L][f]:
                continue
            amax = float(act_max[L, f])
            eff = effects[L, f]
            order = torch.argsort(eff, descending=True)
            top_logits = [cond.decode([int(i)]) for i in order[:10]]
            bottom_logits = [cond.decode([int(i)]) for i in order[-10:]]
            quantile_values = [amax * b / n_bins for b in range(n_bins + 1)]
            model_dict = build_feature_model(
                layer=L, feat_idx=f,
                examples=[e for _, e in top_examples[L][f]],
                act_min=0.0, act_max=amax,
                histogram=hist[L, f].cpu().tolist(),
                quantile_values=quantile_values,
                activation_frequency=float(activation_frequency[L, f]),
                top_logits=top_logits, bottom_logits=bottom_logits,
            )
            import json
            (out_dir / f"{cantor_pair(L, f)}.json").write_text(json.dumps(model_dict))
            n_written += 1

    return {"n_features_written": n_written, "out_dir": str(out_dir),
            "total_positions": total_pos}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--clt-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scan-name", required=True)
    ap.add_argument("--features-root", required=True,
                    help="STORAGE_ROOT/clt_features (subfolder <scan-name> is created).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-per-person", type=int, default=2)
    ap.add_argument("--context-size", type=int, default=64)
    ap.add_argument("--n-people", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--n-bins", type=int, default=40)
    args = ap.parse_args()
    out = generate_dashboards(
        args.model_dir, args.clt_dir, args.data_dir, args.scan_name,
        args.features_root, device=args.device, n_per_person=args.n_per_person,
        context_size=args.context_size, n_people=args.n_people,
        top_k=args.top_k, n_bins=args.n_bins)
    print(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 12: Run test to verify it passes**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_generate_dashboards_small -v`
Expected: PASS. (If `DiverseBioSubset.to_hf_dataset`/`n_tokens` signatures differ from trainCLT's usage, mirror exactly how `clts/trainCLT.py:200-205` calls them.)

- [ ] **Step 13: Commit**

```bash
git add clts/gen_feature_dashboards.py tests/test_attribution.py
git commit -m "feat(clt): per-feature dashboard generator (schema-locked, cantor-named)"
```

---

## Task 7: `clts/validate_graph.py` — intervention validation

**Files:**
- Create: `clts/validate_graph.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing test (ablating a top feature changes the target logit)**

Add to `tests/test_attribution.py`:

```python
@pytest.mark.integration
def test_intervention_changes_target_logit():
    from clts.validate_graph import ablate_top_feature_effect

    res = ablate_top_feature_effect(
        model_dir=MODEL_DIR, clt_dir=CLT_DIR, data_dir=DATA_DIR,
        scan_name=SCAN_NAME, device="cpu", max_feature_nodes=512,
    )
    # Ablating the single most influential feature should move the target-logit
    # probability (sanity that the attributed feature is causally relevant).
    assert "target_prob_before" in res and "target_prob_after" in res
    assert res["target_prob_before"] != res["target_prob_after"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_intervention_changes_target_logit -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clts.validate_graph'`

- [ ] **Step 3: Write the validator**

Create `clts/validate_graph.py`:

```python
"""Validate an attribution graph by intervention (the paper's standard check).

Build a graph, take the most influential CLT feature, ablate it via
model.feature_intervention, and report the change in the target-logit
probability. A causally-relevant feature should move the target logit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clts.build_attribution_graph import build_graph, default_birthday_prompt  # noqa: E402
from clts.load_replacement_model import load_replacement_model  # noqa: E402
from clts.export_tokenizer import ensure_hf_tokenizer  # noqa: E402


def ablate_top_feature_effect(model_dir, clt_dir, data_dir, scan_name,
                              device="cpu", prompt=None, max_feature_nodes=4096):
    from circuit_tracer.utils.demo_utils import get_top_features

    if prompt is None:
        prompt = default_birthday_prompt(data_dir)

    # Build graph to find the top feature (layer, pos, feat_idx).
    out = build_graph(model_dir, clt_dir, data_dir, scan_name,
                      graph_dir=str(Path(clt_dir).parent / "_validate"),
                      slug="_validate", prompt=prompt, device=device,
                      max_feature_nodes=max_feature_nodes, verbose=False)
    graph = out["graph"]
    feats, _scores = get_top_features(graph, n=1)
    layer, pos, feat_idx = feats[0]

    hf_tok = ensure_hf_tokenizer(data_dir)
    model = load_replacement_model(model_dir, clt_dir, hf_tok, scan_name, device=device)

    target_id = int(graph.logit_tokens()[0].item())
    logits_before, _ = model.get_activations(prompt)
    p_before = torch.softmax(logits_before[0, -1].float(), dim=-1)[target_id].item()

    # Ablate the feature (set activation to 0) at its position.
    logits_after, _ = model.feature_intervention(
        prompt, [(layer, pos, feat_idx, 0.0)], freeze_attention=False)
    p_after = torch.softmax(logits_after[0, -1].float(), dim=-1)[target_id].item()

    return {
        "prompt": prompt,
        "top_feature": [int(layer), int(pos), int(feat_idx)],
        "target_token_id": target_id,
        "target_prob_before": float(p_before),
        "target_prob_after": float(p_after),
        "delta": float(p_after - p_before),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--clt-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scan-name", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    import json
    print(json.dumps(ablate_top_feature_effect(
        args.model_dir, args.clt_dir, args.data_dir, args.scan_name,
        device=args.device), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py::test_intervention_changes_target_logit -v`
Expected: PASS. (If `get_top_features` is unavailable in the installed version, replace with: read `graph.selected_features`, compute `compute_node_influence(graph.adjacency_matrix, graph.logit_probabilities)`, take the argmax feature row and map via `graph.active_features`.)

- [ ] **Step 5: Commit**

```bash
git add clts/validate_graph.py tests/test_attribution.py
git commit -m "feat(clt): intervention-based graph validation"
```

---

## Task 8: Sync script extension

**Files:**
- Modify: `scripts/sync_from_psc.sh`

- [ ] **Step 1: Read the current bundle structure**

Run: `sed -n '1,80p' scripts/sync_from_psc.sh`
Expected: see the `bundle` / `transfer` / `extract` subcommands and the `REMOTE_BASE` / tar layout.

- [ ] **Step 2: Add a `clt_features` line to the bundle**

In the section of `scripts/sync_from_psc.sh` that builds the tar on PSC (the `bundle` path), add `clt_features/` to the set of paths included in the tarball, mirroring how SAE dashboards are added. Concretely, where the script lists tar inputs relative to `REMOTE_BASE`, add:

```bash
    clt_features \
```

so the generated `STORAGE_ROOT/clt_features/<scan_name>/*.json` trees travel in the same gzipped tarball. Match the existing quoting/escaping style exactly (the file uses `set -euo pipefail`).

- [ ] **Step 3: Verify the script still parses**

Run: `bash -n scripts/sync_from_psc.sh && echo "syntax OK"`
Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/sync_from_psc.sh
git commit -m "build(clt): sync clt_features dashboards from PSC"
```

---

## Task 9: End-to-end smoke + docs

**Files:**
- Modify: `clts/build_attribution_graph.py` (module docstring smoke block) — or create `clts/README_attribution.md`

- [ ] **Step 1: Create the smoke-run doc**

Create `clts/README_attribution.md`:

````markdown
# CLT attribution graphs — run guide

## One-time (PSC): generate feature dashboards
```bash
clts/.venv-ct/bin/python clts/gen_feature_dashboards.py \
    --model-dir model/grid-L4-H6 \
    --clt-dir   clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final \
    --data-dir  data/bioS_N-Bd_final_grid \
    --scan-name grid-L4-H6 \
    --features-root "$CLT_STORAGE_ROOT/clt_features" --device cuda
# then: ./scripts/sync_from_psc.sh   (pulls clt_features to the Mac)
```

## Per prompt (Mac): build + explore a graph
```bash
clts/.venv-ct/bin/python clts/build_attribution_graph.py \
    --model-dir model/grid-L4-H6 \
    --clt-dir   clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final \
    --data-dir  data/bioS_N-Bd_final_grid --slug bday-recall

clts/.venv-ct/bin/python clts/serve_ui.py \
    --graph-dir clts/clt_graphs/grid-L4-H6/bday-recall \
    --features-dir <synced>/clt_features
# open http://localhost:8032
```

Acceptance: graph builds; the report's top logit is the birthday-date token;
`<slug>.json` + `graph-metadata.json` written; the page renders and clicking a
feature node shows its dashboard. The per-graph report prints
`error_influence_share` — if it is high, prefer a higher-`ce_recovered` CLT.
````

- [ ] **Step 2: Run the full fast test suite (non-integration)**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py -v -m "not integration"`
Expected: PASS (cantor, decoder_logit_effects, build_feature_model schema).

- [ ] **Step 3: Run the integration suite**

Run: `clts/.venv-ct/bin/python -m pytest tests/test_attribution.py -v -m integration`
Expected: PASS (adapter A/B, build graph, serve, dashboards, intervention).

- [ ] **Step 4: Commit**

```bash
git add clts/README_attribution.md
git commit -m "docs(clt): attribution-graph run guide + acceptance"
```

---

## Self-review (completed against the spec)

- **Spec coverage:** Adapter + invariants (Task 3) ↔ spec §2; graph driver + create_graph_files (Task 4) ↔ §3; dashboards format/cantor/schema/corpus (Task 6) ↔ §4; serve (Task 5) ↔ §5; fidelity Checks A & B (Task 3), error-share report C (Task 4), intervention D (Task 7) ↔ §6; tests (all tasks) ↔ §7; storage/sync/CLI (Tasks 4, 6, 8) ↔ §8; dedicated venv + `convert_llama_weights` gate + trainCLT refactor (Tasks 0, 2) ↔ §Dependencies.
- **Placeholder scan:** every code/test step contains complete code; no TODO/TBD.
- **Type consistency:** `build_tl_config`/`build_hooked_transformer`, `load_replacement_model(...)`, `build_graph(...)`, `generate_dashboards(...)`, `cantor_pair`/`cantor_unpair`, `decoder_logit_effects`, `build_feature_model` names match across tasks and tests. `scan_name` is the single identifier threaded through adapter → graph → dashboard subfolder.
- **Known fallbacks noted inline:** tokenizer-kwarg construction (Task 3 Step 4), `get_top_features` availability (Task 7 Step 4), `DiverseBioSubset` call shape (Task 6 Step 12), CPU `max_feature_nodes` tuning (Task 4 Step 4).
