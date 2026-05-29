# CLT Attribution Graphs & Circuit Exploration — Design

**Status:** approved, ready for implementation plan
**Date:** 2026-05-29
**Subproject:** 2 of 3 (CLT training → **attribution-graph engine + interactive exploration** → Neuronpedia-grade UI polish)
**Consumes:** [2026-05-27-clt-training-pipeline-design.md](2026-05-27-clt-training-pipeline-design.md) and its implementation under [clts/](../../../clts/)
**Library:** Anthropic-style circuit tracing via `circuit-tracer` (the `decoderesearch/circuit-tracer` fork, v0.4.1), used **unmodified**.

## Goal

Produce **attribution graphs** for the project's trained Cross-Layer Transcoders and explore the resulting circuits in `circuit-tracer`'s bundled viewer, for the `grid-L4-H6` bios model (names → birthdays) first. The pipeline must (a) require **no weight conversion** — trained CLTs load as-is, (b) **match the standard definition** of attribution graphs (Ameisen et al. 2025, ["Circuit Tracing"](https://transformer-circuits.pub/2025/attribution-graphs/methods.html); Lindsey et al. 2025), and (c) be **reliable and reproducible** — results must not diverge from what the canonical algorithm produces on this model.

The central design commitment that delivers (b) and (c): **we do not reimplement the attribution algorithm.** We call `circuit_tracer.attribute()` exactly as the library intends. The only project-specific code is the *adapter* that wires our custom-sized local model + trained CLT into circuit-tracer, plus *drivers* and a *feature-dashboard generator*. "Matching the standard definition" therefore reduces to **proving the wiring is faithful**, which is a verification problem (Section 6), not an algorithm-design problem.

## Non-goals

- New attribution algorithm or new visualization frontend — both come from `circuit-tracer`.
- Improving CLT quality / retraining (subproject #1). Graph faithfulness is *bounded* by the chosen CLT's reconstruction; we report that bound, we do not raise it here.
- Neuronpedia upload, hosted graphs, or the sharded-binary feature format (deferred; see Open Questions).
- Multi-prompt / batched attribution UI, supernode-annotation tooling beyond what the bundled viewer ships.
- Editing `saes/`, `model/`, or `util/`. The **one** existing-file change is a safe refactor of `clts/trainCLT.py` (Section 2).

## Background

### Why no converter is needed (verified against source)

`circuit-tracer`'s `CrossLayerTranscoder.load_clt()` reads precisely the on-disk layout [clts/clt.py](../../../clts/clt.py) already writes:

- Encoder file `W_enc_{i}.safetensors` with keys `W_enc_{i}` `[d_t, D]`, `b_enc_{i}` `[d_t]`, `b_dec_{i}` `[D]`, `threshold_{i}` `[d_t]`; decoder file `W_dec_{i}.safetensors` with key `W_dec_{i}` `[d_t, N-i, D]`. Matches [clts/clt.py:189-198](../../../clts/clt.py#L189-L198).
- `threshold_{i}` is reshaped `[d_t] → [N, 1, d_t]` on load — matching how we save `self.threshold.data[i]`.
- Presence of `threshold_*` ⇒ JumpReLU; its forward is `features * (features > threshold)`, **identical** to our [clts/clt.py:39](../../../clts/clt.py#L39).

So the trained weights load and compute identically. Section 6's Equivalence Check A asserts this numerically rather than trusting it.

### The one real obstacle: a custom-sized local model

`grid-L4-H6` is `n_layers=4, d_model=384, n_heads=6, vocab=1836` ([model/grid-L4-H6/config.json](../../../model/grid-L4-H6/config.json)). No Hugging Face Llama alias has these dims, so circuit-tracer's `ReplacementModel.from_pretrained("<alias>")` path (which derives the config from the alias) cannot be used. The fix reuses our own proven model-build: [clts/trainCLT.py:110-134](../../../clts/trainCLT.py#L110-L134) already constructs a `HookedTransformer` from this exact checkpoint via a hand-built `HookedTransformerConfig` + `convert_llama_weights`. We factor that out and feed the result into circuit-tracer's `_configure_replacement_model` (Section 2).

### The PSC ↔ Mac split (matches the existing workflow)

Training and heavy batch compute run on **PSC** (CUDA); artifacts sync to the **Mac** for analysis via [scripts/sync_from_psc.sh](../../../scripts/sync_from_psc.sh). This subproject keeps that seam:

- **Feature dashboards** (a GPU pass over a large corpus, accumulating per-feature top-activating examples) → generated on **PSC**, like the existing SAE dashboards; synced to the Mac.
- **Attribution graph building + the interactive viewer** (one short prompt, tiny model, local browser) → run on the **Mac**.

## Section 1 — Architecture & components

All new code lives under [clts/](../../../clts/), mirroring the existing module conventions.

| Module | Runs on | Responsibility |
|---|---|---|
| `clts/tl_model.py` | both | Two helpers extracted verbatim from [trainCLT.py:110-134](../../../clts/trainCLT.py#L110-L134): `build_tl_config(hf_config, device) -> HookedTransformerConfig` (the adapter needs the config alone, to set `model_name`/`tokenizer_name` before constructing the replacement subclass) and `build_hooked_transformer(model_dir, tokenizer, device, dtype) -> HookedTransformer` (used by `trainCLT.setup()`, refactored to call it — behavior-preserving). |
| `clts/load_replacement_model.py` | both | The adapter. Returns a configured `TransformerLensReplacementModel` from a local checkpoint + trained CLT dir + HF tokenizer dir. The only subtle module. |
| `clts/build_attribution_graph.py` | Mac | Driver: `attribute(...) → graph.to_pt(...) → create_graph_files(...)`. CLI; defaults to a birthday-recall prompt. |
| `clts/gen_feature_dashboards.py` | PSC | Batch generator: run the CLT over `index_corpus.pt`, write one `feature_models.Model` JSON per feature. |
| `clts/serve_ui.py` | Mac | Thin wrapper over `circuit_tracer.frontend.local_server.serve(data_dir, port, features_dir)`. |
| `clts/circuit_env/` | both | Committed venv-setup files: `requirements.txt` + a short `README.md` describing how to create the venv and the `convert_llama_weights` import check. (The venv itself, e.g. `clts/.venv-ct`, is gitignored.) |

### Environment

The committed `clts/circuit_env/` is used to create a dedicated venv per machine (e.g. `clts/.venv-ct`, gitignored), holding `circuit-tracer` and its pins (`transformers>=4.56,<=4.57.3`, `transformer-lens>=2.16`, `torch`, `safetensors`, `pyyaml`, `pydantic`, `numpy`). It is isolated from the training/SAE env so no `transformers` downgrade can break the existing pipeline. The drivers add the repo root to `sys.path` and import `util/` + `clts/` as source — they are **not** installed, and require none of the training-env packages (`sae_lens`, `wandb`). Install-time gate: assert `from transformer_lens.loading_from_pretrained import convert_llama_weights` succeeds under the pinned TL version before proceeding.

## Section 2 — The adapter (`load_replacement_model.py`)

This is the only place correctness can silently break, so the build order and invariants are specified exactly.

```python
def load_replacement_model(model_dir, clt_dir, hf_tokenizer_dir,
                           scan_name, device="cpu", dtype=torch.float32):
    hf_model  = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_dir)
    cfg       = build_tl_config(hf_model.config, device)        # from tl_model.py
    cfg.model_name     = scan_name                              # MUST be non-None (see invariants)
    cfg.tokenizer_name = str(hf_tokenizer_dir)                  # MUST be set (see invariants)

    model = TransformerLensReplacementModel(cfg, tokenizer=tokenizer)
    model.load_state_dict(convert_llama_weights(hf_model, cfg), strict=False)   # weights BEFORE configure
    model = model.to(device=device, dtype=dtype)

    clt = load_clt(clt_dir, feature_input_hook="hook_resid_mid",
                   feature_output_hook="hook_mlp_out", scan_name=scan_name,
                   device=device, dtype=dtype, lazy_decoder=False, lazy_encoder=False)
    model._configure_replacement_model(clt)                     # wraps MLP/unembed, sets gradient flow
    return model
```

**Invariants (each is a fidelity requirement, not a nicety):**

1. **Weights load before `_configure_replacement_model`.** Configuration renames state-dict keys (`blocks.*.mlp → …mlp.old_mlp`, `unembed → unembed.old_unembed`); `convert_llama_weights` emits the un-renamed keys, so loading after would silently no-op those weights.
2. **`cfg.model_name` is a non-None string.** circuit-tracer evaluates `"gemma-3" in self.cfg.model_name`; `None` raises `TypeError`. Set it to `scan_name`.
3. **`cfg.tokenizer_name = str(hf_tokenizer_dir)`.** `create_graph_files` later calls `AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)`, and the `Graph` inherits `cfg=model.cfg`. Without this the graph-file step fails. The dir is produced by [clts/export_tokenizer.py](../../../clts/export_tokenizer.py) (`ensure_hf_tokenizer`).
4. **`feature_input_hook`/`feature_output_hook` must equal the CLT's training hooks** (`hook_resid_mid` → `hook_mlp_out`, recorded in the CLT's `config.yaml`). Passing anything else attributes through the wrong activations. We read them from the CLT dir's `config.yaml` and pass them explicitly (defaults happen to match).
5. **`dtype=torch.float32` everywhere.** Our model and our eval ([clts/evalCLT.py](../../../clts/evalCLT.py)) are fp32; `load_clt` defaults to **bf16**, which would numerically diverge. Override to fp32. bf16 is out of scope.
6. **No LayerNorm folding.** circuit-tracer freezes LN denominators via `ln*.hook_scale`, which requires unfolded norm. Our config uses `normalization_type="RMS"` (unfolded) and we build with `HookedTransformer(cfg)` (never `from_pretrained`, which might fold). Do not fold.
7. **`scan_name`** is a single stable identifier (e.g. `"grid-L4-H6"`). It flows CLT → `model.scan_name` → `graph.scan_name`, and **must equal the feature-dashboard subfolder name** (Section 4) so the viewer can find dashboards.

## Section 3 — Attribution graph driver (`build_attribution_graph.py`, Mac)

```python
model = load_replacement_model(model_dir, clt_dir, hf_tokenizer_dir, scan_name, device="cpu")
graph = attribute(prompt=prompt, model=model,
                  max_n_logits=10, desired_logit_prob=0.95,
                  batch_size=256, max_feature_nodes=4096, verbose=True)
graph.to_pt(out_pt)
create_graph_files(graph_or_path=graph, slug=slug, output_path=graph_dir,
                   scan_name=scan_name, node_threshold=0.8, edge_threshold=0.98)
```

- **Birthday-recall default.** The driver builds a prompt from a real person in [data/bioS_N-Bd_final_grid/people.json](../../../data/bioS_N-Bd_final_grid/) (e.g. `"<Name> was born on"`), tokenizes with the condensed tokenizer, and lets `attribute()` auto-select salient logits — the predicted date token should top the list. Flags allow an arbitrary `--prompt` and explicit `--target` later.
- **Position-0 caveat (standard behavior, documented for the user):** circuit-tracer prepends a BOS-like token and zeroes position 0; never place essential content at position 0.
- Output: `<graph_dir>/<slug>.json` + `<graph_dir>/graph-metadata.json`, the format the bundled viewer reads.
- `attribute()` is deterministic (gradient-based, no sampling) ⇒ reproducible given pinned versions.

## Section 4 — Feature dashboards (`gen_feature_dashboards.py`, PSC)

Generated **for all features** in one batch pass (decoupled from any specific graph), matching the existing SAE dashboard workflow.

**Output layout** (verified against the frontend's `init-feature-examples.js`):

```
STORAGE_ROOT / clt_features / <scan_name> / <featureIndex>.json
```

- The viewer requests `./features/<scan_name>/<featureIndex>.json` (served from `features_dir`), where **`featureIndex = cantor_pairing(layer, feat_idx)`** with `cantor_pairing(x, y) = (x+y)(x+y+1)//2 + y` — exactly `Node.feature_node`'s encoding and the inverse of the frontend's `cantorUnpair`. A unit test asserts round-trip correctness.
- Each JSON conforms to `circuit_tracer.frontend.feature_models.Model`: `transcoder_id`, `index`, `examples_quantiles` (buckets of `Example{tokens, tokens_acts_list, train_token_ind, is_repeated_datapoint}`), `top_logits`, `bottom_logits`, `act_min`, `act_max`, `quantile_values`, `histogram`, `activation_frequency`. A test validates a sample against this pydantic model so format drift fails fast.

**Computation:**

- **Corpus:** reuse `saes/sae_inference/<model>/index_corpus.pt` (recommended) so CLT and SAE features are described over identical text; fall back to a fresh `DiverseBioSubset` sample sized by `--n-examples`.
- **Activations:** run the base model with `run_with_cache` on `hook_resid_mid` for all layers (reusing [clts/evalCLT.py:capture_activations](../../../clts/evalCLT.py#L10)), encode through the CLT, and maintain a running **top-K examples per feature** bucketed into activation **quantiles**, plus a per-feature **histogram**, **act_min/max**, and **activation_frequency**. Mirrors the buckets in [saes/featureBucketing.ipynb](../../../saes/featureBucketing.ipynb) (K and bucket counts default to its values).
- **Top/bottom logits:** static per feature — project the feature's summed decoder direction through the unembed. `tie_word_embeddings=True` ⇒ `W_U = W_E.T`; the 1836-token vocab makes this trivial. Decode token ids with the HF tokenizer.
- **Device/dtype:** CUDA, fp32, batched over the corpus.

**Sync:** tar the `clt_features/<scan_name>/` tree (many small files — tar+gzip per the rationale in [sync_from_psc.sh](../../../scripts/sync_from_psc.sh)) and extend the sync script to fetch it to the Mac.

## Section 5 — Serving / UI (`serve_ui.py`, Mac)

```python
serve(data_dir=graph_dir, port=8032, features_dir=synced_clt_features_dir)
```

`data_dir` holds the graph JSON(s); `features_dir` is the synced `clt_features/` root (the viewer appends `<scan_name>/<idx>.json`). Open `http://localhost:8032`; clicking a feature node loads its dashboard. The server returns a handle with `.stop()`. No build step — the frontend ships pre-built in `circuit_tracer/frontend/assets/`.

## Section 6 — Fidelity & validation (the "match the standard definition" pillar)

Because we use `attribute()` unmodified, the **node taxonomy and edge definition are already the standard ones**: nodes are `[active CLT features, MLP-reconstruction error nodes, embedding/token nodes, logit nodes]`; edges are direct linear effects computed with attention patterns and LayerNorm denominators **frozen** (handled by circuit-tracer's `_configure_gradient_flow`); influence/pruning use the library's `compute_influence` (B = A + A² + … = (I−A)⁻¹ − I) and `prune_graph`. We adopt all of it verbatim.

What remains is to prove our *inputs* to that algorithm are faithful and to bound interpretation. Four checks, ordered cheapest-first; the first two are blocking gates in CI/smoke, the last two are reported per graph.

- **Equivalence Check A — CLT compute (blocking).** On a batch of real `hook_resid_mid` activations, assert circuit-tracer's loaded CLT produces encode activations and decode reconstruction equal to [clts/clt.py](../../../clts/clt.py)'s within `1e-4` (fp32). Proves zero loading/format/semantics drift.
- **Equivalence Check B — replacement faithfulness (blocking).** Assert the assembled `TransformerLensReplacementModel`'s full-MLP-replacement cross-entropy matches [clts/evalCLT.py:ce_recovered_full](../../../clts/evalCLT.py#L140) for the same CLT within `1e-3`. Proves correct hooks/dtype/wiring — i.e. circuit-tracer is attributing through *the same model we evaluated*.
- **Reporting C — graph quality (per graph).** Log the target-logit probability, the **error-node influence share** (fraction of total influence carried by error nodes), and the feature-node count after pruning. **High error share ⇒ the graph is still standard but under-explained by this CLT;** surface it instead of shipping silently. This makes the CLT-quality bound explicit and is why we recommend attributing with the sweep's best CLT (highest `final_eval/ce_recovered`), not an arbitrary trial.
- **Validation D — intervention (recommended, optional driver `clts/validate_graph.py`).** The paper/tutorial validate graphs experimentally: ablate or scale a top feature via `model.feature_intervention(...)` and confirm the logit change matches the feature's attributed sign/magnitude. This is the gold-standard confirmation that an edge means what the graph claims.

**Reproducibility:** pin `circuit-tracer`, `transformers`, `transformer-lens` versions in the venv and record them (plus `scan_name`, CLT dir, model dir, prompt) in a sidecar next to each graph. Attribution is deterministic, so identical inputs + pinned versions ⇒ identical graphs.

## Section 7 — Testing & verification

**File:** `tests/test_attribution.py`, parallel to [tests/test_sae_explorer.py](../../../tests/test_sae_explorer.py). Tiny synthetic model/CLT where possible; the real `grid-L4-H6` only in the smoke path.

Unit (pytest, fast, CPU):
1. `test_cantor_index_roundtrip` — `cantor_pairing` inverts the frontend's `cantorUnpair` for a grid of `(layer, feat)`; results are in-range.
2. `test_adapter_builds_and_reconstructs` — adapter returns a `TransformerLensReplacementModel`; **Equivalence Check A** holds on random activations.
3. `test_replacement_ce_matches_eval` — **Equivalence Check B** on a small token batch.
4. `test_dashboard_schema` — a generated dashboard JSON validates against `feature_models.Model`, and its filename equals `cantor_pairing(layer, feat)`.
5. `test_graph_nodes_invert` — for a small built graph, every feature node's `feature` int inverts to an in-range `(layer, feat)` and matches an existing dashboard filename (protects the viewer's feature lookup).

Smoke (documented, not CI), Mac:
```bash
clts/.venv-ct/bin/python clts/build_attribution_graph.py \
    --model-dir model/grid-L4-H6 \
    --clt-dir   clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final \
    --data-dir  data/bioS_N-Bd_final_grid
clts/.venv-ct/bin/python clts/serve_ui.py --graph-dir <out> --features-dir <synced>
```
Acceptance: graph builds without error; auto-selected top logit is the birthday-date token; `<slug>.json` + `graph-metadata.json` written; Checks A & B pass and report C prints; page renders at `localhost:8032` with a populated feature panel for ≥1 node.

Format-compat (manual, one-time): confirm a real graph + a real dashboard load in the bundled viewer with no console errors. If it fails, a unit test caught the wrong thing — fix both.

Out of scope for testing: attribution-algorithm correctness (it is circuit-tracer's, covered by their tests), convergence quality, sharded-binary format, Neuronpedia upload.

## Section 8 — Storage, CLI, sync

- **Storage** (reuse `STORAGE_ROOT` from [trainCLT.py:41](../../../clts/trainCLT.py#L41)):
  - Graphs (Mac): `STORAGE_ROOT/clt_graphs/<scan_name>/<slug>/` → `<slug>.json`, `graph-metadata.json`, `<slug>.pt`, `run-meta.json` (versions/inputs).
  - Dashboards (PSC→Mac): `STORAGE_ROOT/clt_features/<scan_name>/<featureIndex>.json`.
- **CLI flags** follow the existing `--model-dir` / `--data-dir` / `--clt-dir` conventions; `--scan-name` defaults to the CLT's `config.yaml` `model_name`; `--device` defaults to `cpu` on Mac, `cuda` on PSC.
- **Sync:** extend [scripts/sync_from_psc.sh](../../../scripts/sync_from_psc.sh) with a `clt_features` bundle target (tar+gzip the `<scan_name>/` tree, transfer, extract under the Mac's `STORAGE_ROOT`).

## Dependencies / environment

- New, **isolated** venv only; nothing added to the training/SAE env. No changes to `saes/`, `model/`, `util/`.
- One behavior-preserving refactor: extract the TL-config/model build from [trainCLT.py:110-134](../../../clts/trainCLT.py#L110-L134) into `clts/tl_model.py`; `trainCLT.setup()` calls it. Covered by re-running the trainCLT smoke test.
- `circuit-tracer` installed from the `decoderesearch/circuit-tracer` fork (`pip install .`).

## Open questions deferred to implementation

1. **Feature-data format.** Per-feature JSON (chosen now: simple, ~24.5k small files, tar-synced like the SAE dashboards) vs. the sharded-binary format (`index.json.gz` + per-layer `.bin`, byte-range served — fewer files, sync-friendly, but requires emitting exact offsets and a `/`-prefixed scan path). Switch to binary later only if file count/sync becomes painful.
2. **Corpus source.** Reuse `index_corpus.pt` (recommended) vs. a fresh, larger bio sample for "massive prompts." Decide based on whether the SAE corpus is large/representative enough for CLT features.
3. **Top-K and quantile-bucket counts** for dashboards — adopt `saes/featureBucketing.ipynb` defaults unless they produce sparse buckets at `d_model=384`.
4. **Which CLT to attribute by default.** The current single trial (`mult16_l02_…`) may not be the sweep winner; default `--clt-dir` should point at the highest `final_eval/ce_recovered` CLT once a sweep has run (Reporting C makes the consequence visible).
5. **`max_feature_nodes` / pruning thresholds** for this small model — `4096` / `0.8` / `0.98` are starting points; tune against graph readability for birthday recall.
