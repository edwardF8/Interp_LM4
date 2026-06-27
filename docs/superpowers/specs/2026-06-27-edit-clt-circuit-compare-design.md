# Fact-Edit → CLT-Adaptation → Circuit Comparison — Design

**Date:** 2026-06-27
**Status:** approved (design); implementation plan pending
**Repos touched:** `Interp_LM4/` (primary), reaches into `FactEditing/` for the edit step.
**Files touched (new unless noted):**
- `Interp_LM4/clts/edit_clt/` — new experiment package (notebooks + modules + driver).
- `Interp_LM4/clts/trainCLT.py` — *modified*, pure add-on (opt-in flags only).
- `Interp_LM4/scripts/train_clt_psc.sh` — *modified*, pure add-on (env→flag passthrough).

## 1. Goal & framing

Apply one **MEMIT fact edit** to the `grid-L4-H6` model, then study how a cross-layer
transcoder (CLT) must change to keep explaining the model, and what that does to the
**attribution graph** of the edited fact.

The three methods are a **spectrum of how much the CLT is allowed to change to absorb
the edit**:

| Config | CLT | Model | Adaptation | Expected drift |
|---|---|---|---|---|
| **Reference** | apricot (baseline) | original | — | — |
| **Method 3** | apricot (stale) | edited | none | 0 by construction |
| **Method 2** | apricot → fine-tuned | edited | minimal | small, local |
| **Method 1** | trained from scratch | edited | full | large |

Research intent for **Method 2**: find *the smallest change to the baseline dictionary
that restores replacement fidelity on the edited model* — "adopt the edit without
developing new capabilities (reorganizing features)." This is **not** "train to best
`ce_recovered`" (that slides toward Method 1).

**Method 3 is an analysis, not a training run** (the stale baseline CLT evaluated on the
edited model). It lives entirely in Notebook 2.

## 2. Locked decisions

- **Base model:** `grid-L4-H6` (4 layers, 6 heads, d_model 384, intermediate 1024,
  condensed vocab 1836, ctx 512). Loaded fp32, RMSNorm **eps=1e-5** (TL default — *not*
  the checkpoint's 1e-6; do not "fix").
- **Baseline CLT = `apricot-sweep-8`** = `clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final`
  (expansion 16, l0 2.0, lr 1e-4, 50 epochs, n_examples 10000). Used as **both** the
  reference CLT and Method 2's starting checkpoint. Note: apricot has known weak
  mid-layer reconstruction (`nmse_L1≈0.44`, `nmse_L2≈0.55`, large error nodes) — so
  Method 2's parity bar is easy to clear and baseline graphs already carry big error
  nodes. Acceptable; just context for reading results.
- **Fact to edit (default, parameterized):** person 0 (Gabriella Rigby), field `month`,
  Feb→**July**, `edit_template=0`. Month-only edits are the documented-reliable MEMIT case.
- **Method 1:** from scratch on the edited model with apricot's exact config.
- **Method 2:** fine-tune from apricot on the edited model. v1 variant matrix:
  - `v2-basic` — lr 2e-5, no/short warmup, stop on **plateau-or-parity**, epoch cap ≤5,
    `eval-every` 100, drift *measured* (not enforced).
  - `v2-fixed` — lr 2e-5, fixed 2 epochs, no target.
  - `v2-anchor` *(optional, off by default)* — `v2-basic` + `λ·‖W−W_base‖²` proximity penalty.
- **Convergence ("same replacement-model stats"):** primary signal is `ce_recovered`
  (the replacement-model headline metric / sweep objective). Stop at first of:
  *parity* (edited-model `ce_recovered` ≥ baseline `ce_recovered` − tol) or *plateau*
  (`ce_recovered` improves by < `min-delta` for `patience` consecutive evals; falls back
  to eval reconstruction `mse_total` if the 64-row `ce_recovered` is too noisy),
  capped at the epoch budget. Drift metrics are the real test of "no new capabilities".
- **Execution:** "local-prep → HPC-train", and **also test + submit on PSC**. Heavy
  training (Methods 1 & 2 at full config) runs on PSC. The Mac runs the edit + a CPU
  smoke test + job emission. Notebook 1 is environment-aware (PSC vs Mac).

## 3. Artifacts, layout & the manifest

Single source of truth is a **manifest** written by Notebook 1 and read by Notebook 2,
so neither notebook hard-codes paths.

- **Edited model** (`save_pretrained`): `model/grid-L4-H6-edit-p0-month-jul/`
  (config.json + model.safetensors; loads in trainCLT + circuit-tracer; eps handled at
  load time as always). Naming: `grid-L4-H6-edit-<person>-<field>-<value>`.
- **CLTs** (under the resolved `CLT_STORAGE_ROOT`):
  - baseline (apricot): existing `clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final`
  - Method 1: `clt_runs/grid-L4-H6-edit-p0-month-jul/standalone/mult16_l02_lr0.0001_ep50_n10000/final`
  - Method 2: `clt_runs/grid-L4-H6-edit-p0-month-jul/<out-tag>/mult16_l02_lr2e-05_ep<E>_n10000/final`
    for `out-tag ∈ {method2-v2-basic, method2-v2-fixed, method2-v2-anchor}`.
- **Attribution graphs:** `clt_storage/clt_graphs/<scan_name>/<person_slug>/…`, one dir
  per config, all on the **same prompt**. `scan_name ∈ {baseline_orig, m3_stale,
  m1_scratch, m2_finetune}`. Shared-feature configs (baseline/m3/m2) may reuse apricot's
  existing `clt_features/grid-L4-H6/` dashboards; `m1_scratch` has independent features
  (own dashboards optional, deferrable).
- **Manifest:** `clt_storage/edit_experiments/<exp_id>/manifest.json` — fact spec, all
  model/CLT/data paths, apricot **target stats** (`ce_recovered` + per-layer `nmse` on the
  *original* model), per-method configs + variants, the trace prompt, and per-job status.

## 4. `trainCLT.py` — pure add-on (opt-in flags; defaults byte-identical to today)

New CLI flags (all default to off/None; when none are set, behavior is unchanged):

| Flag | Purpose | Insertion point (verified) |
|---|---|---|
| `--resume-from <final_dir>` | `CrossLayerTranscoder.load_from_dir` instead of fresh init (Method 2); assert expansion/d_model/n_layers match | CLT build, `train_one_run` L208-210 |
| `--out-tag <str>` | replace `sweep_folder` ("standalone") in the run dir so M2 variants don't collide | run-dir build, L182-184 |
| `--target-ce-recovered <float>` | parity early-stop: after a periodic eval, `break` when reached | after periodic eval, L259-274 |
| `--plateau-patience <int>` / `--plateau-min-delta <float>` | plateau early-stop on monitored metric (`ce_recovered`, or `mse_total` if noisy) | same |
| `--eval-every <int>` | make `EVAL_EVERY` (hard-coded 600) configurable so short fine-tunes hit the knee | L224 |
| `--anchor-lambda <float>` (default 0) | add `λ·‖W−W_base‖²` (snapshot of resumed weights) to the loss | snapshot post-load L210; penalty in step L243-246 |

The post-loop final-eval + `save_to_dir` (L278-308) already runs after any early `break`,
so early-stopped runs still save + log normally. Built **TDD**; unit tests assert
default-path invariance and each flag's effect.

*Alternative considered & rejected:* a separate `retrainCLT.py` wrapper — duplicates the
training loop. Opt-in flags keep one tested code path.

## 5. Notebook 1 — `clts/edit_clt/01_edit_and_train.ipynb`

Runs locally (training env) or on PSC (env-aware). Cells:
1. Setup — both repos on `sys.path`, storage roots, device, env detection (PSC vs Mac).
2. Load experiment config (`edit_clt_config.py`); echo it.
3. **Baseline target stats** — load apricot + the *original* model → `ce_recovered_full`
   + `compute_layer_metrics` (per-layer nmse). Write to manifest (Method 2's target).
4. **MEMIT edit** (CPU) via `run_single_edit(person=0, fields=('month',),
   new_values={'month':'July'}, edit_template=0, …)`; print efficacy / generalization /
   specificity to confirm it took.
5. **Save edited model dir** (`save_pretrained`) + reload-verify; quick greedy generation
   sanity ("…July…") via the birthday probe.
6. **Local CPU smoke test** — run the Method 1 and Method 2 code paths at a tiny budget
   (n≈200, 1 epoch, `eval-every` small) to validate the new flags end-to-end.
7. **Emit / submit HPC jobs** — render `submit_edit_clt.sh` invocations (M1 + each M2
   variant) with `CONDA_ENV=lm4-ct`; on PSC offer to `sbatch` directly (incl. `--test`
   first) and print `squeue` watch; on Mac print exact `git push` + `sbatch` commands
   (+ optional rsync of the edited model).
8. **Write manifest** (status `submitted` / `smoke-only`) + "next steps" markdown.

## 6. Notebook 2 — `clts/edit_clt/02_circuit_compare.ipynb`

Runs in the circuit-tracer env (`clts/.venv-ct` locally, `lm4-ct` on PSC). Lifts
setup/build/influence/serve cells from `clts/explore_attribution.ipynb`. Cells:
1. Setup + load manifest → the 4 configs.
2. **Build/cache attribution graphs** for all 4 configs on the same prompt
   (`build_attribution_graph.build_graph`). Trace both each config's top logit **and** a
   fixed target set `{" February", " July"}` so the same target is comparable across
   configs. Per-config report: `replacement_score`, `completeness_score`,
   `error_influence_share`, top logit, `target_logit_prob`, n feature nodes.
3. **Replacement-stats comparison table** across configs (+ training `ce_recovered` /
   per-layer `nmse` from the manifest). This is the "investigate the baseline CLT graph" view.
4. **Feature diff ("what goes away / appears"):**
   - M2 & M3 (share apricot's indexing): per-index decoder cosine drift; active-feature
     Jaccard on the prompt; appeared/disappeared active-feature sets; influence-change
     scatter (`compute_node_influence`).
   - M1 (independent init, *but* same `CLT_SEED=0` as apricot — partial alignment worth
     measuring): feature-match by decoder cosine (greedy/Hungarian) + graph-level
     active-feature comparison; do **not** compare blindly by index.
5. **Method-3 focus:** `error_influence_share` baseline_orig vs m3_stale (same CLT, orig
   vs edited model) — does the stale CLT route the edited fact through error nodes / lose
   the date-recall path?
6. **Feature selector** — filter graph features by criteria (layer, influence ≥ τ,
   fires-on-month-token, in-baseline-not-edited, …) and pull dashboards (`writefeatures`
   `feature_rank`/`incoming_feature_edges` + `feature_index.cantor_pair`).
7. **Drift-spectrum summary** — drift (m3≈0 < m2 < m1) vs `replacement_score`, tying back
   to "adopt the edit, no new capabilities."
8. Optional: launch `serve_ui` per config on separate ports; save a markdown + figures
   comparison report under `clt_storage/edit_experiments/<exp_id>/`.

## 7. Supporting modules (`clts/edit_clt/`)

- `edit_clt_config.py` — experiment dataclass(es): fact spec, model/CLT/data paths,
  method-variant matrix, scan names, trace prompt, storage. Single source of truth.
- `prepare_edited_model.py` — `make_edited_model(cfg) -> Path` (edit + `save_pretrained`
  + verify). CLI + importable.
- `drift.py` — `decoder_cosine_drift`, `active_feature_overlap`, `match_features`
  (pure, unit-tested).
- `submit_edit_clt.sh` — experiment driver (modeled on `submit_clt_sweeps_grid.sh`):
  sets env (incl. `CONDA_ENV=lm4-ct`, `CLT_STORAGE_ROOT`) + `sbatch`es M1 and each M2
  variant via the extended `scripts/train_clt_psc.sh`. Supports **`--test`** =
  `N_EXAMPLES=1000 EPOCHS=3 --time=00:30:00` exercising M1 + M2 paths on the real GPU.
- `README.md` — the runbook.

## 8. PSC path

`scripts/train_clt_psc.sh` is the canonical single-CLT SBATCH job (env-overridable;
fail-fast preflight; sets `CLT_STORAGE_ROOT=/jet/home/friedmae/data_storage/LM4_Results`).
Submit idiom: set env vars, then `sbatch --export=ALL --job-name=… --time=… scripts/train_clt_psc.sh`.

**Gotchas surfaced by checking the real scripts:**
1. **`lm4` conda env was deleted — training stack now lives in `lm4-ct`.** The sweep
   driver overrides `CONDA_ENV=lm4-ct`; `train_clt_psc.sh` still *defaults* to `lm4`
   (stale → preflight fails). **All edit-clt jobs pass `CONDA_ENV=lm4-ct`.** (`lm4-ct`
   also runs circuit-tracer, so Notebook 2 shares it on PSC.) The canonical
   `docs/environments-and-hpc.md` still lists `lm4` as live — stale.
2. **Extend `scripts/train_clt_psc.sh`** (pure add-on, env-gated, mirroring its
   `ROBUSTNESS_MANIFEST→ROBUST_ARGS` pattern): `RESUME_FROM`, `TARGET_CE_RECOVERED`,
   `PLATEAU_PATIENCE`, `PLATEAU_MIN_DELTA`, `OUT_TAG`, `ANCHOR_LAMBDA`, `EVAL_EVERY`
   → append the matching `trainCLT.py` flags + add preflight existence checks
   (esp. `RESUME_FROM/config.yaml` for Method 2).

**Verify on PSC (cannot be checked from the Mac — become job preflight + runbook steps):**
- apricot present at `$CLT_STORAGE_ROOT/clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final`
  (for `--resume-from`); if missing, rsync ~130 MB up.
- edited model + new code present on PSC: run the edit on PSC (deterministic, CPU-minutes,
  no transfer) **or** rsync the 30 MB up; `git push` locally → `git pull` on PSC.

**Runbook:** push code → on PSC `git pull` + verify apricot/data → run edit (or rsync) →
`submit_edit_clt.sh --test` → on success submit full M1 + M2 variants → sync `clt_runs`
back to Mac → open Notebook 2.

## 9. Conventions honored

TDD; **pure add-on** (no behavior change to existing tools when new flags unset); fp32 +
RMSNorm eps=1e-5; condensed vocab (`KeyError` on OOV — keep edits in-vocab); run from repo
root; bios byte-identical to training corpus. Notebook 2 / circuit-tracer strictly in the
ct env (`.venv-ct` local, `lm4-ct` on PSC); everything else in the training env.

## 10. Testing

- `trainCLT.py` add-on: unit tests for default-path invariance + each flag (`--resume-from`
  loads weights; `--out-tag` redirects dir; target/plateau trigger an early `break`;
  `--eval-every` honored; `--anchor-lambda` adds the penalty term).
- `drift.py`: unit tests on synthetic CLT weight pairs.
- `prepare_edited_model.py`: integration test (skipif no model) — edit, save, reload,
  assert the edited fact present + an unrelated fact unchanged.
- End-to-end: the Notebook 1 CPU smoke test (tiny budget) + the PSC `--test` job.

## 11. Out of scope (v1)

- Drift as an active stop-guard (v1 measures/reports only).
- Multi-fact / multi-person sweeps (single edit; person/field parameterized).
- New feature dashboards for `m1_scratch` (optional; defer unless needed).
- Editing day/year (month-only is the reliable case; the config allows it but it's untested here).
