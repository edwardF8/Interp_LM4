# Attribution Explorer Notebook — Design

**Date:** 2026-05-30
**Artifact:** `clts/explore_attribution.ipynb`

## Purpose

An interactive notebook for sampling people from the BioS dataset and tracing
the birthday-recall circuit for each one — wrapping the existing
`build_attribution_graph` / `serve_ui` pipeline so exploration is a matter of
re-running one cell, not re-typing a CLI invocation.

## Context

The building blocks already exist and are reused verbatim (no logic
duplicated):

- `util.bio_sampler.BioSampler` — renders bios with the exact training templates.
- `clts.load_replacement_model.load_replacement_model` — assembles the
  circuit-tracer replacement model (model + CLT).
- `clts.build_attribution_graph.build_graph` — runs circuit-tracer's
  `attribute()` and writes viewer files + fidelity report.
- `clts.serve_ui.start_server` — runs the bundled web viewer in-process
  (returns a live server object on a background thread; no subprocess).
- `clts.storage.storage_root` — resolves the artifact root.

Fixed inputs (only one trained CLT exists):

| Input | Value |
|-------|-------|
| model-dir | `model/grid-L4-H6` |
| clt-dir | `clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final` |
| data-dir | `data/bioS_N-Bd_final_grid` |
| scan-name | `grid-L4-H6` |
| device | `cpu` |

20,577 feature dashboards are already synced under
`clt_storage/clt_features/grid-L4-H6/`, so the web viewer renders populated
feature cards.

## Notebook structure (6 cells, run top-to-bottom)

1. **Setup** — add repo root to `sys.path`, pin the fixed inputs, build the
   `CondensedTokenizer` for the in-vocab name check. The replacement model is
   loaded inside `build_graph()` per call; the model is tiny (4 layers,
   d=384), so this is cheap and avoids duplicating `build_graph`'s body.
2. **Load people** — build `BioSampler`, report count, preview a few people.
3. **Sample a person** *(the re-run cell)* — draw a random person whose
   first+last name tokenizes cleanly in the condensed vocab (reusing the
   in-vocab check logic from `default_birthday_prompt`), display fields + the
   full rendered bio, and construct the prompt `"<Name> was born on the"` (the
   `" the"`-suffixed form that traces date recall, not "predict the word
   'the'"). Optional `seed` for reproducible draws.
4. **Build the graph** — call `build_graph()` for the sampled person (slug
   from the name), write viewer files to
   `clt_storage/clt_graphs/grid-L4-H6/<slug>/`, print the fidelity report.
5. **Inline visualization** — compute per-node influence via
   `compute_node_influence(graph.adjacency_matrix, logit_weights)` where
   `logit_weights[-n_logits:] = graph.logit_probabilities`. Rank the first
   `n_features` nodes, map each back through
   `active_features[selected_features[i]]` → `(layer, pos, feature_idx)` with
   `activation_values[i]`, and draw a horizontal bar chart of the top-k by
   influence. Show the fidelity metrics as a text panel.
6. **Launch web viewer** — call `start_server(graph_dir, features_dir,
   scan_name)` in-process, print `http://localhost:8032`.

## Decisions

- **Single CLT hardwired** (`grid-L4-H6`) — it's the only trained run.
- **Inline viz = top-features bar chart**, not a node-link graph. The full
  node-link rendering is exactly what the web viewer is for; reproducing it
  inline is not worth the complexity.
- **Prompt suffix `" the"`** by default to surface the date-recall circuit
  (per `README_attribution.md` prompt caveat).
- **In-process server**, not a subprocess — `serve()` returns a live server
  object on a background thread.

## Non-goals

- No new attribution/scoring logic — the notebook orchestrates existing code.
- No field selection (birthday only) and no manual-prompt mode in v1.
- No multi-CLT comparison.

## Acceptance

- Notebook runs top-to-bottom in the `.venv-ct` kernel without error.
- Re-running cell 3 → cell 6 produces a graph + inline chart + live viewer for
  a new person.
- Viewer at `http://localhost:8032` renders the graph and populated feature
  cards.
