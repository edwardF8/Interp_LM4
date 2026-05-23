# SAE Local Explorer — Design

**Date:** 2026-05-23
**Repo:** `Interp_LM4`

## Goal

Make the custom SAE at `sae_runs/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final/` browsable in a Neuronpedia-style UI, fully local. Same visual UX (per-feature top activating examples with token-level highlighting, activation histograms, logit attribution), no upload required.

## Why not Neuronpedia

Two blockers for the hosted route:

1. **Custom tokenizer.** `CondensedTokenizer` is an arbitrary 1836-id remap of GPT-2 (`old_to_new.json`). Neuronpedia decodes token ids server-side and has no way to register this remap.
2. **Custom model.** The 4-layer Llama at `model/BD_llama_6heads_1epoch_4layers` is a research artifact, not a public checkpoint. The dataset is synthetic bios for an interp project.

Use `sae_dashboard` instead — the same library Neuronpedia uses to generate its panels. It runs against our local model/SAE/tokenizer and writes static HTML.

## Components

### 1. New module: `sae_explorer.py`

Three functions:

#### `build_index_corpus(sampler, tokenizer, n_per_person=2, context_size=64, seed=0) -> LongTensor[N, T]`

Builds the corpus the dashboard indexes against. For each of the 50,000 people in `sampler.people`, draw `n_per_person` distinct templates → tokenize with `CondensedTokenizer` → prepend EOS (matches the leading-EOS convention used in the notebook's existing inspection cell) → pad/truncate to `context_size`.

- Default `n_per_person=2` → 100,000 bios → ~2.5M tokens.
- Returns a `[N, context_size]` `torch.long` tensor on CPU. Caller moves to device.
- Caches the result to `{SAE_PATH}/dashboards/index_corpus.pt` keyed on `(n_per_person, context_size, seed, sampler_path)`; rebuild only if the key changes.

#### `make_dashboard(model, sae, tokens, tokenizer, out_dir, hook_name, features=None, batch_size=8)`

Wraps `sae_dashboard.SaeVisRunner` (current API — confirmed at impl time against the installed version 0.8.0).

Responsibilities:
- Attach `tokenizer` to `model.tokenizer` if not already attached. `sae_dashboard` reaches for `model.tokenizer` for token-level rendering.
- Shim any tokenizer methods `sae_dashboard` calls that `CondensedTokenizer` does not implement (most likely `convert_ids_to_tokens`; possibly `tokenize`, `batch_decode` — we already have `batch_decode`). Shims live on `CondensedTokenizer` itself, kept minimal.
- Build the `SaeVisConfig` (features to render, hook name, batch size).
- Run the index pass over `tokens`.
- Write one HTML panel per feature: `{out_dir}/feature_{idx}.html`.
- Write `{out_dir}/index.html` — a static page that links to each feature panel, with a small inline preview of the feature's L0, activation density, and max activation, so you can scan features without opening each one.

If `features=None`, dashboard every feature in the SAE (`sae.cfg.d_sae`). If a list/range, dashboard only those — useful for iterating.

#### `steer(model, sae, text, feature_idx, scale, hook_name, n_new_tokens=20) -> dict`

Causal probe. Encode `text` with `CondensedTokenizer`, run a forward pass with a hook that adds `scale * sae.W_dec[feature_idx]` to the residual at `hook_name`, return a dict with:
- `clean_top_tokens` — top-5 predicted next tokens from the clean forward pass
- `steered_top_tokens` — top-5 predicted next tokens with the feature boosted
- `delta_logits` — top-5 tokens that gained the most logit under steering

Synchronous, static output. Called inline in the notebook; no widget.

### 2. Notebook cells appended to `analyzingSAE.ipynb`

After the existing setup cells (model, tokenizer, sampler all loaded):

1. **Build corpus** — `tokens = build_index_corpus(sampler, tokenizer, n_per_person=2)`.
2. **Load SAE** — using the existing `load_sae` from `evalSAE.py` (already imported in the project; we re-import here).
3. **Generate dashboard** — `make_dashboard(model, sae, tokens.to(device), tokenizer, out_dir=Path(SAE_PATH) / "dashboards", hook_name="blocks.1.hook_mlp_out")`. Prints the local `index.html` path. User opens in browser, or notebook displays via `IPython.display.IFrame`.
4. **Steering example** — one `steer(...)` call on a feature picked from the dashboard, to show the workflow for causal experiments.

The hook name and the SAE checkpoint path are taken from the notebook's existing globals (`SAE_PATH`). The hook layer used during training (`blocks.1.hook_mlp_out` per `evalSAE.py`) is the default but exposed as a parameter.

### 3. Output layout

```
sae_runs/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final/
└── dashboards/
    ├── index_corpus.pt        # cached tokenized corpus
    ├── index.html             # browse page (one row per feature)
    ├── feature_0.html
    ├── feature_1.html
    └── ...                    # one per feature (d_sae total ≈ 6144)
```

## Dependencies

- Add `sae-dashboard==0.8.0` to the project venv (`pip install sae-dashboard` into `/Users/efmac/Code/Project Code/CRL-Interp/.venv`).
- No new top-level dependencies expected beyond that — `sae_dashboard` brings its own template/HTML machinery.

## Risks and mitigations

- **Tokenizer compatibility.** `sae_dashboard` may call `CondensedTokenizer` methods we don't implement. Mitigation: discover by running, add minimal shims to `CondensedTokenizer`. If the surface turns out to be unreasonably large or assumes HF-tokenizer internals (e.g., a real `vocab` dict, `added_tokens_encoder`), fall back to the custom-helpers route — write our own minimal panel renderer (`top_activations`, `inspect_text`, `logit_lens` as plain functions producing HTML/DataFrames).
- **Compute.** ~2.5M tokens through model+SAE on MPS. Expected runtime 10–20 min. The corpus tensor is cached, so re-runs of `make_dashboard` (e.g., regenerating only a subset of features) don't re-tokenize.
- **Dead features.** If ~60%+ of the 6144 features never fire across the corpus, their panels will be empty. `make_dashboard` should skip features whose max activation is zero across the index pass, and note skipped features on `index.html`.

## Out of scope

- Auto-interpretation (asking an LLM to label features). Easy add later; not part of this spec.
- Search across features (e.g. "show me features that fire on dates"). Easy add later.
- Interactive widgets — sticking with static HTML for v1; `steer` is the only interactive piece, and it's called from notebook cells.

## Acceptance criteria

1. `sae_explorer.py` exists with the three functions above, importable from the notebook.
2. Running the three new notebook cells end-to-end produces an `index.html` plus per-feature HTML files under the SAE checkpoint's `dashboards/` directory.
3. Opening `index.html` in a browser shows a list of features with previews; clicking a feature opens its panel with top activating examples and activation histogram.
4. `steer(...)` returns a dict with the three fields above and runs in under 5 seconds for a short bio.
