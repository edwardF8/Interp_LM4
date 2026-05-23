# SAE Local Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the custom SAE at `sae_runs/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final/` browsable in a Neuronpedia-style UI, fully local, via static HTML generated from `sae_dashboard`.

**Architecture:** New helper module `sae_explorer.py` exposes three functions: `build_index_corpus` (assemble + cache the 100k-bio tokenized corpus), `make_dashboard` (wrap `sae_dashboard.SaeVisRunner`, attach `CondensedTokenizer` to `model.tokenizer`, emit per-feature HTML + index page), `steer` (causal probe — boost a feature's decoder direction at a hook and report logit deltas). Notebook `analyzingSAE.ipynb` gains four cells that call them.

**Tech Stack:** Python 3.11, `sae_lens` 6.44.0 (installed), `sae_dashboard` 0.8.0 (to install), `transformer_lens`, `torch` (MPS), `pytest` for the tests we add.

**Spec:** [`docs/superpowers/specs/2026-05-23-sae-local-explorer-design.md`](../specs/2026-05-23-sae-local-explorer-design.md)

---

## File structure

- Create: `sae_explorer.py` — the helper module (one file, ~250 lines). Three public functions; one private helper to attach the tokenizer to the model.
- Create: `tests/__init__.py`, `tests/test_sae_explorer.py` — pytest tests for `build_index_corpus` and `steer`. `make_dashboard` is exercised end-to-end via the notebook (its job is to wrap an external API; unit tests would mostly mock it).
- Modify: `analyzingSAE.ipynb` — append four cells (build corpus, load SAE, generate dashboard, steer example).
- Modify: `condensed_tokenizer.py` — add the minimal tokenizer-method shims `sae_dashboard` needs, discovered in Task 2.
- New (cache only, not committed): `sae_runs/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final/dashboards/` — corpus tensor + HTML output. Add to `.gitignore`.

---

## Task 1: Install sae_dashboard and verify environment

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Install sae_dashboard 0.8.0**

Run (use the project venv directly — the user invokes it via `/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python`):

```bash
"/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/pip" install 'sae-dashboard==0.8.0'
```

Expected: install succeeds. Some dep resolution warnings are fine.

- [ ] **Step 2: Verify import**

Run:

```bash
"/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python" -c "import sae_dashboard; print(sae_dashboard.__version__ if hasattr(sae_dashboard, '__version__') else 'ok')"
```

Expected: prints `0.8.0` or `ok`. No import error.

- [ ] **Step 3: Gitignore the dashboards output dir**

Add to the end of `.gitignore`:

```
# SAE dashboard HTML output + cached corpora
sae_runs/**/dashboards/
```

- [ ] **Step 4: Commit**

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
git add .gitignore
git commit -m "Add sae_dashboard to env; gitignore dashboard output"
```

Note: `pip install` modifies the venv (not the repo), so the install itself is not committed. The lock/requirements file in this project (if any) should be updated separately if the team uses one — check for a `requirements.txt` or `pyproject.toml` before assuming.

---

## Task 2: API discovery — pin down `sae_dashboard`'s actual entry point

This task produces no code commit; it produces a short notes file you'll reference in Task 4. The point is to read the installed package's source/docstrings and write down exactly: (a) the class/function name we'll call, (b) its required arguments, (c) which tokenizer methods it calls.

**Files:**
- Create: `docs/superpowers/specs/sae_dashboard_api_notes.md` (scratch notes; commit it so future-you/the next agent can read it)

- [ ] **Step 1: Locate the package**

Run:

```bash
"/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python" -c "import sae_dashboard, os; print(os.path.dirname(sae_dashboard.__file__))"
```

Expected: prints an absolute path like `.../site-packages/sae_dashboard`. Note it — call it `$SDPATH` below.

- [ ] **Step 2: Identify the main runner class**

```bash
ls "$SDPATH"
grep -rE "class +Sae[A-Za-z]*Runner" "$SDPATH" --include="*.py" -l
```

Expected: one or more files defining a runner class (likely `sae_vis_runner.py` with `SaeVisRunner`, or `feature_data_generator.py`). Open the top hit with `Read` and write to `docs/superpowers/specs/sae_dashboard_api_notes.md`:

- The exact class name (e.g. `SaeVisRunner`).
- Its `__init__` signature.
- The method that runs the index pass (likely `run`).
- The config dataclass it consumes (likely `SaeVisConfig`) and its required fields.

- [ ] **Step 3: Find tokenizer method calls**

```bash
grep -rE "\.tokenizer\.(decode|encode|convert_ids_to_tokens|tokenize|batch_decode|vocab|added_tokens|get_vocab|__call__)" "$SDPATH" --include="*.py"
```

List every tokenizer method `sae_dashboard` calls in the notes file. Mark which `CondensedTokenizer` already has (`decode`, `batch_decode`, `__call__`, `encode`) and which need shims.

- [ ] **Step 4: Find the HTML output entry point**

```bash
grep -rE "def +save_feature_centric_vis|def +.*to_html|html.*save|write.*html" "$SDPATH" --include="*.py"
```

Note: the function or method that writes per-feature HTML, and what arguments it takes (typically a path).

- [ ] **Step 5: Commit the notes**

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
git add docs/superpowers/specs/sae_dashboard_api_notes.md
git commit -m "API discovery notes for sae_dashboard 0.8.0"
```

---

## Task 3: `build_index_corpus` — tokenized bio corpus, with cache

**Files:**
- Create: `sae_explorer.py`
- Create: `tests/__init__.py` (empty file)
- Create: `tests/test_sae_explorer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` as an empty file.

Create `tests/test_sae_explorer.py`:

```python
"""Tests for sae_explorer."""
from __future__ import annotations

from pathlib import Path

import torch

from sae_explorer import build_index_corpus
from bio_sampler import BioSampler
from condensed_tokenizer import CondensedTokenizer

DATA_DIR = Path("data/BD_llama_inital")


def _sampler_and_tokenizer():
    sampler = BioSampler(DATA_DIR / "people.json", fields=("birthday",), seed=0)
    tokenizer = CondensedTokenizer.from_remap_path(DATA_DIR / "old_to_new.json")
    return sampler, tokenizer


def test_build_index_corpus_shape_and_dtype(tmp_path):
    sampler, tokenizer = _sampler_and_tokenizer()
    # Smoke-size: use a tiny subset of people via sampler.people slicing inside
    # the helper. For the test we ask for 2 templates per person but only over
    # the first 8 people; build_index_corpus accepts a `people` override for this.
    tokens = build_index_corpus(
        sampler,
        tokenizer,
        n_per_person=2,
        context_size=32,
        seed=0,
        people=sampler.people[:8],
        cache_path=tmp_path / "corpus.pt",
    )
    assert tokens.dtype == torch.long
    assert tokens.shape == (16, 32)  # 8 people * 2 templates, T=32


def test_build_index_corpus_first_token_is_eos(tmp_path):
    sampler, tokenizer = _sampler_and_tokenizer()
    tokens = build_index_corpus(
        sampler,
        tokenizer,
        n_per_person=1,
        context_size=32,
        seed=0,
        people=sampler.people[:4],
        cache_path=tmp_path / "corpus.pt",
    )
    assert (tokens[:, 0] == tokenizer.eos_token_id).all()


def test_build_index_corpus_is_deterministic(tmp_path):
    sampler, tokenizer = _sampler_and_tokenizer()
    a = build_index_corpus(
        sampler, tokenizer, n_per_person=2, context_size=32, seed=42,
        people=sampler.people[:4], cache_path=tmp_path / "a.pt",
    )
    b = build_index_corpus(
        sampler, tokenizer, n_per_person=2, context_size=32, seed=42,
        people=sampler.people[:4], cache_path=tmp_path / "b.pt",
    )
    assert torch.equal(a, b)


def test_build_index_corpus_uses_cache(tmp_path):
    sampler, tokenizer = _sampler_and_tokenizer()
    cache = tmp_path / "corpus.pt"
    build_index_corpus(
        sampler, tokenizer, n_per_person=1, context_size=16, seed=0,
        people=sampler.people[:4], cache_path=cache,
    )
    assert cache.exists()
    # Corrupt the underlying sampler so a recompute would produce different tokens;
    # cache hit should still return the original.
    sampler.people = sampler.people[:1]
    tokens = build_index_corpus(
        sampler, tokenizer, n_per_person=1, context_size=16, seed=0,
        people=sampler.people[:4], cache_path=cache,
    )
    assert tokens.shape == (4, 16)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
"/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python" -m pytest tests/test_sae_explorer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sae_explorer'`.

- [ ] **Step 3: Implement `build_index_corpus`**

Create `sae_explorer.py`:

```python
"""Local Neuronpedia-style exploration for the custom SAE.

Three public functions:

  build_index_corpus  - assemble + cache the tokenized bio corpus the
                        dashboard indexes against.
  make_dashboard      - wrap sae_dashboard.SaeVisRunner; write per-feature
                        HTML panels + an index page. (Task 4.)
  steer               - causal probe: boost a feature's decoder direction
                        at a hook, report logit deltas. (Task 5.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch


def build_index_corpus(
    sampler,
    tokenizer,
    n_per_person: int = 2,
    context_size: int = 64,
    seed: int = 0,
    people: Sequence[dict] | None = None,
    cache_path: str | Path | None = None,
) -> torch.Tensor:
    """Build the [N, T] long tensor of token ids the dashboard indexes against.

    For each person in `people` (default: sampler.people), draw `n_per_person`
    distinct template indices, render each one, tokenize with the condensed
    tokenizer, prepend the EOS token, and pad/truncate to `context_size`.

    If `cache_path` is given and exists, load + return it without recomputing.
    On a miss, save the result there.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    if people is None:
        people = sampler.people

    rng = torch.Generator().manual_seed(seed)
    n_templates = sampler.n_templates
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    rows: list[list[int]] = []
    for person in people:
        # distinct templates per person (or all of them if n_per_person >= n_templates)
        k = min(n_per_person, n_templates)
        perm = torch.randperm(n_templates, generator=rng).tolist()
        for exposure_idx in perm[:k]:
            text = sampler.render(person, exposure_idx)
            ids = [eos] + tokenizer.encode(text)
            if len(ids) >= context_size:
                ids = ids[:context_size]
            else:
                ids = ids + [pad] * (context_size - len(ids))
            rows.append(ids)

    tokens = torch.tensor(rows, dtype=torch.long)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokens, cache_path)

    return tokens
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
"/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python" -m pytest tests/test_sae_explorer.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
git add sae_explorer.py tests/
git commit -m "Add build_index_corpus + tests for SAE explorer"
```

---

## Task 4: `make_dashboard` — wrap sae_dashboard, write HTML

**Files:**
- Modify: `sae_explorer.py`
- Modify: `condensed_tokenizer.py` (add tokenizer shims from Task 2 notes)

This task does NOT use TDD — the function is a thin wrapper around an external library, and its real acceptance test is "open `index.html` and look at it". Tests would mostly mock `sae_dashboard`, defeating the point. We do a smoke run on a tiny feature subset as the validation step.

- [ ] **Step 1: Add tokenizer shims**

Open the notes you wrote in Task 2 (`docs/superpowers/specs/sae_dashboard_api_notes.md`). For each tokenizer method `sae_dashboard` calls that `CondensedTokenizer` does not implement, add a minimal shim. Common cases:

- `convert_ids_to_tokens(ids) -> list[str]`: implement as `[self.gpt2.convert_ids_to_tokens(self.new_to_old[int(i)]) for i in ids]`.
- `tokenize(text) -> list[str]`: implement as `self.gpt2.tokenize(text)` (returns gpt2-space token strings; sae_dashboard uses these only for display).
- `get_vocab() -> dict[str, int]`: if needed, build `{tok: new_id for tok, gpt2_id in self.gpt2.get_vocab().items() if gpt2_id in self.old_to_new for new_id in [self.old_to_new[gpt2_id]]}`.

Add only the methods actually called. Each shim is 1–3 lines. Put them at the bottom of the `CondensedTokenizer` class in `condensed_tokenizer.py`.

- [ ] **Step 2: Implement `make_dashboard`**

Append to `sae_explorer.py`. The exact `SaeVisRunner` / `SaeVisConfig` field names come from the notes you wrote in Task 2 — replace `_SaeVisRunner` and `_SaeVisConfig` below with the actual imports, and adjust the constructor kwargs to match the notes.

```python
def _attach_tokenizer(model, tokenizer) -> None:
    """sae_dashboard reaches for model.tokenizer; make sure ours is there."""
    if getattr(model, "tokenizer", None) is None:
        model.tokenizer = tokenizer


def make_dashboard(
    model,
    sae,
    tokens: torch.Tensor,
    tokenizer,
    out_dir: str | Path,
    hook_name: str,
    features: Iterable[int] | None = None,
    batch_size: int = 8,
    minibatch_size_tokens: int = 128,
    verbose: bool = True,
) -> Path:
    """Run sae_dashboard against (model, sae, tokens) and write HTML panels.

    Writes one file per feature under {out_dir}/feature_{idx}.html plus an
    {out_dir}/index.html linking to all non-empty features. Returns the index
    path.

    `features=None` runs every feature in the SAE; pass a list/range to render
    a subset (use this for iteration).
    """
    from sae_dashboard.sae_vis_runner import SaeVisRunner  # adjust per Task 2 notes
    from sae_dashboard.sae_vis_data import SaeVisConfig    # adjust per Task 2 notes

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _attach_tokenizer(model, tokenizer)

    if features is None:
        features = list(range(sae.cfg.d_sae))
    else:
        features = list(features)

    cfg = SaeVisConfig(
        hook_point=hook_name,
        features=features,
        minibatch_size_tokens=minibatch_size_tokens,
        verbose=verbose,
    )
    runner = SaeVisRunner(cfg)
    data = runner.run(encoder=sae, model=model, tokens=tokens)

    # Per-feature HTML. The exact save method comes from Task 2 notes.
    for feat in features:
        data.save_feature_centric_vis(
            filename=str(out_dir / f"feature_{feat}.html"),
            feature_idx=feat,
        )

    _write_index_html(out_dir, sae, data, features)
    return out_dir / "index.html"


def _write_index_html(out_dir: Path, sae, data, features: list[int]) -> None:
    """Write a small index.html listing each feature with a preview blurb."""
    rows = []
    for feat in features:
        # `data` exposes per-feature stats via .feature_data_dict (verify in Task 2);
        # rendered as table cells here for a one-page overview. If a feature is
        # dead (max_act == 0), skip its link but still note it on the index.
        fd = data.feature_data_dict.get(feat) if hasattr(data, "feature_data_dict") else None
        max_act = getattr(getattr(fd, "feature_tables_data", None), "max_act", None) if fd else None
        if max_act is None or max_act == 0.0:
            rows.append(f'<tr><td>{feat}</td><td colspan="2">(dead)</td></tr>')
        else:
            rows.append(
                f'<tr><td>{feat}</td><td>{max_act:.3f}</td>'
                f'<td><a href="feature_{feat}.html">open</a></td></tr>'
            )

    html = (
        "<!doctype html><html><head><title>SAE features</title>"
        "<style>body{font-family:system-ui;margin:2rem}"
        "table{border-collapse:collapse}td{padding:.25rem .75rem;border-bottom:1px solid #eee}"
        "</style></head><body>"
        f"<h1>SAE features ({len(features)})</h1>"
        "<table><thead><tr><th>feature</th><th>max activation</th><th></th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )
    (out_dir / "index.html").write_text(html)
```

- [ ] **Step 3: Smoke test on 4 features**

Run from the repo root:

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
"/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python" - <<'PY'
from pathlib import Path
import torch
from transformers import LlamaForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_llama_weights

from bio_sampler import BioSampler
from condensed_tokenizer import CondensedTokenizer
from evalSAE import load_sae
from sae_explorer import build_index_corpus, make_dashboard

MODEL_DIR  = Path("model/BD_llama_6heads_1epoch_4layers")
DATA_DIR   = Path("data/BD_llama_inital")
SAE_PATH   = Path("sae_runs/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final")
OUT_DIR    = SAE_PATH / "dashboards_smoke"
HOOK       = "blocks.1.hook_mlp_out"

device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = CondensedTokenizer.from_remap_path(DATA_DIR / "old_to_new.json")
sampler   = BioSampler(DATA_DIR / "people.json", fields=("birthday",), seed=0)

hf_model = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
hf_cfg = hf_model.config
tl_cfg = HookedTransformerConfig(
    n_layers=hf_cfg.num_hidden_layers, d_model=hf_cfg.hidden_size,
    d_head=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
    n_heads=hf_cfg.num_attention_heads, d_mlp=hf_cfg.intermediate_size,
    d_vocab=hf_cfg.vocab_size, n_ctx=hf_cfg.max_position_embeddings,
    act_fn="silu", normalization_type="RMS", gated_mlp=True,
    positional_embedding_type="rotary",
    rotary_base=int(getattr(hf_cfg, "rope_theta", 10000.0)),
    rotary_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
    final_rms=True, tie_word_embeddings=hf_cfg.tie_word_embeddings,
    initializer_range=hf_cfg.initializer_range,
    n_key_value_heads=hf_cfg.num_key_value_heads, device=device,
)
model = HookedTransformer(tl_cfg)
model.load_state_dict(convert_llama_weights(hf_model, tl_cfg), strict=False)
model.to(device).eval()

sae = load_sae(SAE_PATH, device)
tokens = build_index_corpus(
    sampler, tokenizer, n_per_person=2, context_size=64, seed=0,
    people=sampler.people[:200],  # smoke: 400 bios only
    cache_path=OUT_DIR / "smoke_corpus.pt",
).to(device)
print("tokens", tokens.shape, tokens.dtype)

idx = make_dashboard(model, sae, tokens, tokenizer, OUT_DIR, HOOK, features=range(4))
print("wrote", idx)
PY
```

Expected: prints `tokens torch.Size([400, 64]) torch.int64` and `wrote sae_runs/.../dashboards_smoke/index.html`. Four `feature_*.html` files exist in that directory.

If `sae_dashboard` raises `AttributeError` on a tokenizer method, add that shim to `CondensedTokenizer` and rerun. Repeat until the smoke run finishes.

- [ ] **Step 4: Open the HTML in a browser and eyeball it**

```bash
open "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4/sae_runs/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final/dashboards_smoke/index.html"
```

Expected: a page listing features 0–3 with max activations and links. Click into a feature and verify the bio strings render with tokens visible (your custom tokenizer's `decode` is being used). If you see `<|endoftext|>` or raw token-ids instead of bio text, the tokenizer shim is wrong — go back to Step 1 and check `convert_ids_to_tokens`.

- [ ] **Step 5: Commit**

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
git add sae_explorer.py condensed_tokenizer.py
git commit -m "Add make_dashboard + tokenizer shims for sae_dashboard"
```

---

## Task 5: `steer` — causal feature probe

**Files:**
- Modify: `sae_explorer.py`
- Modify: `tests/test_sae_explorer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sae_explorer.py`:

```python
import pytest


@pytest.fixture(scope="module")
def model_sae_tokenizer():
    """Loads model + SAE + tokenizer once; module-scoped because it's slow."""
    from pathlib import Path
    import torch
    from transformers import LlamaForCausalLM
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformer_lens.loading_from_pretrained import convert_llama_weights
    from evalSAE import load_sae

    MODEL_DIR = Path("model/BD_llama_6heads_1epoch_4layers")
    SAE_PATH  = Path("sae_runs/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = CondensedTokenizer.from_remap_path(DATA_DIR / "old_to_new.json")
    hf_model = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
    hf_cfg = hf_model.config
    tl_cfg = HookedTransformerConfig(
        n_layers=hf_cfg.num_hidden_layers, d_model=hf_cfg.hidden_size,
        d_head=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        n_heads=hf_cfg.num_attention_heads, d_mlp=hf_cfg.intermediate_size,
        d_vocab=hf_cfg.vocab_size, n_ctx=hf_cfg.max_position_embeddings,
        act_fn="silu", normalization_type="RMS", gated_mlp=True,
        positional_embedding_type="rotary",
        rotary_base=int(getattr(hf_cfg, "rope_theta", 10000.0)),
        rotary_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        final_rms=True, tie_word_embeddings=hf_cfg.tie_word_embeddings,
        initializer_range=hf_cfg.initializer_range,
        n_key_value_heads=hf_cfg.num_key_value_heads, device=device,
    )
    model = HookedTransformer(tl_cfg)
    model.load_state_dict(convert_llama_weights(hf_model, tl_cfg), strict=False)
    model.to(device).eval()
    sae = load_sae(SAE_PATH, device)
    return model, sae, tokenizer


def test_steer_returns_expected_keys(model_sae_tokenizer):
    from sae_explorer import steer
    model, sae, tokenizer = model_sae_tokenizer
    out = steer(
        model, sae, tokenizer,
        text=" Gabriella Ella Rigby was born on",
        feature_idx=0,
        scale=5.0,
        hook_name="blocks.1.hook_mlp_out",
    )
    assert set(out.keys()) >= {"clean_top_tokens", "steered_top_tokens", "delta_logits"}
    assert len(out["clean_top_tokens"]) == 5
    assert len(out["steered_top_tokens"]) == 5
    assert len(out["delta_logits"]) == 5


def test_steer_scale_zero_matches_clean(model_sae_tokenizer):
    """scale=0 should leave logits unchanged, so steered top tokens = clean top tokens."""
    from sae_explorer import steer
    model, sae, tokenizer = model_sae_tokenizer
    out = steer(
        model, sae, tokenizer,
        text=" Gabriella Ella Rigby was born on",
        feature_idx=0,
        scale=0.0,
        hook_name="blocks.1.hook_mlp_out",
    )
    assert out["clean_top_tokens"] == out["steered_top_tokens"]
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
"/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python" -m pytest tests/test_sae_explorer.py -k steer -v
```

Expected: FAIL with `ImportError` or `AttributeError` because `steer` doesn't exist.

- [ ] **Step 3: Implement `steer`**

Append to `sae_explorer.py`:

```python
def steer(
    model,
    sae,
    tokenizer,
    text: str,
    feature_idx: int,
    scale: float,
    hook_name: str,
) -> dict:
    """Boost feature `feature_idx`'s decoder direction by `scale` at `hook_name`.

    Encodes `text`, runs a clean forward and a steered forward, and returns
    the top-5 next-token predictions from each plus the top-5 tokens whose
    logit *gained* the most under steering.
    """
    device = next(model.parameters()).device
    ids = [tokenizer.eos_token_id] + tokenizer.encode(text)
    input_tokens = torch.tensor([ids], device=device)

    # sae.W_dec is [d_sae, d_in]; pick row, broadcast across the sequence.
    direction = sae.W_dec[feature_idx].to(device)

    def steering_hook(act, hook):
        return act + scale * direction

    with torch.no_grad():
        clean_logits = model(input_tokens)[0, -1]               # [d_vocab]
        steered_logits = model.run_with_hooks(
            input_tokens, fwd_hooks=[(hook_name, steering_hook)],
        )[0, -1]

    def topk_tokens(logits, k=5):
        vals, idxs = logits.topk(k)
        return [
            {"token_id": int(i), "text": tokenizer.decode([int(i)]), "logit": float(v)}
            for v, i in zip(vals, idxs)
        ]

    delta = steered_logits - clean_logits
    return {
        "clean_top_tokens":   topk_tokens(clean_logits),
        "steered_top_tokens": topk_tokens(steered_logits),
        "delta_logits":       topk_tokens(delta),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
"/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python" -m pytest tests/test_sae_explorer.py -v
```

Expected: all 6 tests pass (4 from Task 3 + 2 from this task).

- [ ] **Step 5: Commit**

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
git add sae_explorer.py tests/test_sae_explorer.py
git commit -m "Add steer() for causal feature probing"
```

---

## Task 6: Wire the four notebook cells

**Files:**
- Modify: `analyzingSAE.ipynb`

The notebook already has cells that load the model, tokenizer, sampler, and a `SAE_PATH` global. We append four new cells after the `### Exploring SAE` markdown heading.

- [ ] **Step 1: Open `analyzingSAE.ipynb` and locate the empty cell after `### Exploring SAE`**

That empty cell (the one currently showing `model` as its only content is two below it). Append the following new cells *after* the existing `model` cell — keep `### Exploring SAE - Specific Example` and its cells *below* the new ones.

- [ ] **Step 2: Add cell — build corpus**

New code cell:

```python
from pathlib import Path
from sae_explorer import build_index_corpus

DASHBOARD_DIR = Path(SAE_PATH) / "dashboards"
tokens = build_index_corpus(
    sampler,
    tokenizer,
    n_per_person=2,
    context_size=64,
    seed=0,
    cache_path=DASHBOARD_DIR / "index_corpus.pt",
)
print("corpus shape:", tokens.shape)  # expect [100000, 64]
```

- [ ] **Step 3: Add cell — load SAE**

New code cell:

```python
from evalSAE import load_sae

sae = load_sae(SAE_PATH, device)
print("d_sae =", sae.cfg.d_sae)
HOOK = "blocks.1.hook_mlp_out"
```

- [ ] **Step 4: Add cell — generate dashboard**

New code cell. Comment explains the runtime expectation:

```python
from sae_explorer import make_dashboard

# Full pass: ~6144 features over 2.5M tokens. Expect 10-20 min on MPS.
# To iterate faster while debugging a single feature, pass features=[123].
index_html = make_dashboard(
    model,
    sae,
    tokens.to(device),
    tokenizer,
    out_dir=DASHBOARD_DIR,
    hook_name=HOOK,
)
print("open in browser:", index_html)
```

- [ ] **Step 5: Add cell — steering example**

New code cell:

```python
from sae_explorer import steer

# Pick a feature from the dashboard's index page and probe it causally.
result = steer(
    model, sae, tokenizer,
    text=" Gabriella Ella Rigby was born on",
    feature_idx=0,         # replace with a feature that looked interesting
    scale=5.0,
    hook_name=HOOK,
)
for k, rows in result.items():
    print(f"\n=== {k} ===")
    for r in rows:
        print(f"  {r['logit']:+.2f}  {r['text']!r}  (id={r['token_id']})")
```

- [ ] **Step 6: Run the notebook from the top through Step 4 once**

In Jupyter (or VS Code's notebook UI), run all cells top-to-bottom up through and including the steering cell. The dashboard cell takes 10-20 minutes; the steering cell takes <5 seconds.

Expected:
- `corpus shape: torch.Size([100000, 64])` (or whatever `len(sampler.people) * n_per_person` evaluates to).
- `d_sae = 6144` (or whatever `mult16 * d_model` evaluates to).
- The dashboard cell prints a path ending in `dashboards/index.html`.
- The steering cell prints three sections, with `scale=5.0` shifting the top tokens vs the clean run.

- [ ] **Step 7: Commit**

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
git add analyzingSAE.ipynb
git commit -m "Add SAE explorer cells to analyzingSAE notebook"
```

---

## Task 7: Verify acceptance criteria from spec

These are the four acceptance criteria from the design doc. Walk through them as a final check, then commit any small fixes you find.

- [ ] **Step 1: AC1 — `sae_explorer.py` exists with three public functions**

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
"/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python" -c "from sae_explorer import build_index_corpus, make_dashboard, steer; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 2: AC2 — Notebook cells produce HTML output**

After running the four new cells from Task 6, verify:

```bash
ls "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4/sae_runs/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final/dashboards/" | head -5
```

Expected: `index.html`, `feature_0.html`, `feature_1.html`, … and `index_corpus.pt` are all present.

- [ ] **Step 3: AC3 — Browser view works**

```bash
open "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4/sae_runs/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final/dashboards/index.html"
```

Visually confirm: (a) a list of features with max-activation column, (b) clicking a non-dead feature opens its panel, (c) the panel shows bio strings with token-level highlighting.

- [ ] **Step 4: AC4 — `steer` finishes in under 5s**

Already verified by the steering cell printing within seconds in Task 6 Step 6. If you want a hard check:

```bash
cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4"
"/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python" -m pytest tests/test_sae_explorer.py -k steer -v --durations=5
```

Expected: both steer tests pass, each <5s after the model-load fixture warms up.

- [ ] **Step 5: If any AC failed, fix and commit**

If you patched anything in this task, commit it:

```bash
git add -A && git commit -m "Address acceptance-criteria fixes for SAE explorer"
```

If everything passed, no commit needed — the feature is done.
