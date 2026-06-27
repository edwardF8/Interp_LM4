# Edit-CLT → Circuit-Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply one MEMIT fact edit to `grid-L4-H6`, retrain a CLT three ways (from scratch / fine-tune-from-checkpoint / stale), and compare the resulting attribution graphs.

**Architecture:** A small **pure add-on** to `clts/trainCLT.py` (opt-in flags for checkpoint-resume + early-stop + out-tag + anchor) and `scripts/train_clt_psc.sh` (env→flag passthrough), plus a new self-contained experiment package `clts/edit_clt/` (config + manifest, edited-model builder, drift metrics, graph-comparison helpers, two notebooks, a PSC submit driver). Notebook logic lives in tested modules; notebooks are thin orchestration.

**Tech Stack:** Python 3.11/3.13, PyTorch, TransformerLens, Anthropic circuit-tracer (0.4.1, isolated venv), MEMIT (vendored), wandb, pytest, SLURM (PSC Bridges-2).

## Global Constraints

- **Run all Python from the Interp_LM4 repo root** (`/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4`). Absolute imports (`from clts… import …`, `from util… import …`).
- **Two environments, never mixed:** training/edit code (trainCLT, prepare_edited_model, drift, config) runs in the **training env** (local `../CRL-Interp/.venv`; PSC conda **`lm4-ct`**). Anything importing `circuit_tracer` (compare.py, Notebook 2, build_graph) runs in the **circuit-tracer env** (`clts/.venv-ct` local; PSC **`lm4-ct`**). Invoke ct tools as `clts/.venv-ct/bin/python …`.
- **On PSC always pass `CONDA_ENV=lm4-ct`** — the `lm4` env was deleted; `train_clt_psc.sh` still defaults to the dead `lm4`.
- **fp32 everywhere; RMSNorm eps=1e-5** — model builders deliberately keep TransformerLens's default eps=1e-5 (not the checkpoint's 1e-6). Never set eps.
- **Condensed vocab** — `CondensedTokenizer.encode` raises `KeyError` on OOV; keep edits/prompts in-vocab (month names, day 1–28, year 1700–1899, real people).
- **Pure add-on rule** — when none of the new `trainCLT.py` flags are set, behavior must be byte-identical to today. Tests assert this.
- **Baseline CLT (apricot-sweep-8):** `clt_storage/clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final` (expansion 16, l0 2.0, lr 1e-4, ep 50, n 10000). Reference + Method 2 start.
- **Constants (from `trainCLT.py`):** `CLT_SEED=0`, `BATCH_SIZE=4096`. Trial dir name = `f"mult{expansion}_l0{l0:g}_lr{lr:g}_ep{epochs}_n{n_examples}"`.
- **Tests:** `pytest` from repo root. Mark real-weight/circuit-tracer tests `@pytest.mark.integration` and `skipif` the artifact dirs are absent (repo convention). Fast suite: `pytest -m "not integration"`.
- **Commits:** conventional-commit messages; one per completed step group. Do not push unless asked; branch before committing if on the default branch.

## File Structure

| File | Responsibility |
|---|---|
| `clts/trainCLT.py` *(modify)* | add `_run_dir`, `_should_stop`, `_anchor_penalty` pure helpers; opt-in CLI flags; wire resume/out-tag/eval-every/early-stop/anchor into `train_one_run`; make `parse_args(argv=None)` |
| `scripts/train_clt_psc.sh` *(modify)* | env→flag passthrough (`RESUME_FROM`, `TARGET_CE_RECOVERED`, `PLATEAU_PATIENCE`, `PLATEAU_MIN_DELTA`, `OUT_TAG`, `ANCHOR_LAMBDA`, `EVAL_EVERY`) + preflight for `RESUME_FROM` |
| `clts/edit_clt/__init__.py` | empty package marker |
| `clts/edit_clt/edit_clt_config.py` | `FactSpec`, `MethodConfig`, `EditCLTConfig`, `default_config`, `expected_clt_dir`, manifest read/write |
| `clts/edit_clt/prepare_edited_model.py` | `make_edited_model` (MEMIT edit → `save_pretrained` → verify) |
| `clts/edit_clt/drift.py` | `decoder_cosine_drift`, `active_feature_sets`, `active_feature_overlap`, `match_features` |
| `clts/edit_clt/compare.py` | `GraphConfig`, `configs_from_manifest`, `build_or_load_graph`, `comparison_table`, `feature_diff_table` |
| `clts/edit_clt/submit_edit_clt.sh` | PSC driver: `sbatch` M1 + M2 variants via `train_clt_psc.sh`; `--test` mode |
| `clts/edit_clt/01_edit_and_train.ipynb` | Notebook 1 (edit + target stats + smoke + emit/submit jobs + manifest) |
| `clts/edit_clt/02_circuit_compare.ipynb` | Notebook 2 (build 4 graphs + compare + feature diff) |
| `clts/edit_clt/README.md` | runbook |
| `docs/environments-and-hpc.md` *(modify)* | fix stale `lm4`→`lm4-ct` note |
| `tests/test_trainclt_addon.py` | unit tests for the 3 pure helpers + flag defaults |
| `tests/test_edit_clt_config.py` | config defaults, `expected_clt_dir`, manifest round-trip |
| `tests/test_drift.py` | drift metrics on synthetic CLTs |
| `tests/test_compare.py` | `comparison_table` / `feature_diff_table` on synthetic reports |
| `tests/test_prepare_edited_model.py` | integration: edit→save→reload→fact present |

---

### Task 1: `trainCLT.py` pure add-on (resume / early-stop / out-tag / anchor)

**Files:**
- Modify: `clts/trainCLT.py`
- Test: `tests/test_trainclt_addon.py`

**Interfaces:**
- Consumes: `CrossLayerTranscoder.load_from_dir` (clts/clt.py), `ce_recovered_full` (clts/evalCLT.py), module globals `STORAGE_ROOT`, `ARGS`, `device`.
- Produces:
  - `_run_dir(storage_root: Path, model_name: str, name: str, out_tag: str | None = None, sweep_id: str | None = None) -> Path`
  - `_should_stop(history: list[float], target: float | None = None, patience: int | None = None, min_delta: float | None = None) -> tuple[bool, str | None]`
  - `_anchor_penalty(clt, base_params: dict[str, torch.Tensor], lam: float) -> torch.Tensor`
  - `parse_args(argv: list[str] | None = None)` and new CLI flags: `--resume-from` (Path), `--out-tag` (str), `--target-ce-recovered` (float), `--plateau-patience` (int), `--plateau-min-delta` (float), `--eval-every` (int), `--anchor-lambda` (float, default 0.0).

- [ ] **Step 1: Write failing tests for the three pure helpers**

Create `tests/test_trainclt_addon.py`:

```python
from pathlib import Path
import torch
from clts import trainCLT
from clts.clt import CrossLayerTranscoder


def test_run_dir_out_tag_overrides_sweep_folder():
    root = Path("/tmp/store")
    # default -> standalone
    assert trainCLT._run_dir(root, "m", "trial") == root / "clt_runs/m/standalone/trial"
    # sweep id -> sweep-<id>
    assert trainCLT._run_dir(root, "m", "trial", sweep_id="abc") == root / "clt_runs/m/sweep-abc/trial"
    # out_tag wins over both
    assert trainCLT._run_dir(root, "m", "trial", out_tag="method2-v2-basic", sweep_id="abc") \
        == root / "clt_runs/m/method2-v2-basic/trial"


def test_should_stop_parity():
    # reaching target stops with reason 'parity'
    assert trainCLT._should_stop([0.1, 0.5, 0.81], target=0.8) == (True, "parity")
    assert trainCLT._should_stop([0.1, 0.5, 0.79], target=0.8) == (False, None)


def test_should_stop_plateau():
    # 3 consecutive gains all < min_delta -> plateau
    hist = [0.50, 0.70, 0.705, 0.708, 0.710]
    assert trainCLT._should_stop(hist, patience=3, min_delta=0.01) == (True, "plateau")
    # a big recent gain prevents plateau
    hist2 = [0.50, 0.70, 0.705, 0.90]
    assert trainCLT._should_stop(hist2, patience=3, min_delta=0.01) == (False, None)
    # not enough history yet
    assert trainCLT._should_stop([0.7, 0.705], patience=3, min_delta=0.01) == (False, None)


def test_should_stop_disabled_returns_false():
    assert trainCLT._should_stop([0.1, 0.2, 0.3]) == (False, None)


def test_anchor_penalty_zero_when_lambda_zero():
    clt = CrossLayerTranscoder(n_layers=2, d_model=4, expansion=2)
    base = {n: p.detach().clone() for n, p in clt.named_parameters()}
    assert float(trainCLT._anchor_penalty(clt, base, 0.0)) == 0.0


def test_anchor_penalty_positive_after_perturbation():
    clt = CrossLayerTranscoder(n_layers=2, d_model=4, expansion=2)
    base = {n: p.detach().clone() for n, p in clt.named_parameters()}
    with torch.no_grad():
        clt.W_enc[0] += 1.0
    pen = trainCLT._anchor_penalty(clt, base, 0.5)
    assert float(pen) > 0.0


def test_new_flags_default_off():
    args = trainCLT.parse_args(["--model-dir", "x", "--data-dir", "y"])
    assert args.resume_from is None
    assert args.out_tag is None
    assert args.target_ce_recovered is None
    assert args.plateau_patience is None
    assert args.plateau_min_delta is None
    assert args.eval_every is None
    assert args.anchor_lambda == 0.0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4" && pytest tests/test_trainclt_addon.py -v`
Expected: FAIL (`AttributeError: module 'clts.trainCLT' has no attribute '_run_dir'`, and `parse_args` rejects the unknown args / takes no argv).

- [ ] **Step 3: Add the three pure helpers**

In `clts/trainCLT.py`, after `trial_name` (ends line ~153), add:

```python
def _run_dir(storage_root, model_name, name, out_tag=None, sweep_id=None):
    """Resolve the run directory. out_tag (when set) replaces the
    standalone/sweep folder so Method-2 variants sharing a trial name don't
    collide. Mirrors the legacy layout otherwise."""
    folder = out_tag or (f"sweep-{sweep_id}" if sweep_id else "standalone")
    return storage_root / "clt_runs" / model_name / folder / name


def _should_stop(history, target=None, patience=None, min_delta=None):
    """Early-stop decision for resume/fine-tune runs. `history` is the monitored
    metric (ce_recovered) in eval order, higher = better.
    - parity: latest >= target
    - plateau: the last `patience` gains are each < min_delta
    Returns (stop, reason)."""
    if not history:
        return (False, None)
    if target is not None and history[-1] >= target:
        return (True, "parity")
    if patience is not None and min_delta is not None and len(history) > patience:
        recent = history[-(patience + 1):]
        gains = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        if all(g < min_delta for g in gains):
            return (True, "plateau")
    return (False, None)


def _anchor_penalty(clt, base_params, lam):
    """L2 proximity penalty pulling CLT params toward `base_params` (a snapshot
    of the resumed checkpoint). Returns a 0-d tensor; exactly 0 when lam == 0."""
    dev = next(clt.parameters()).device
    if not lam:
        return torch.zeros((), device=dev)
    total = torch.zeros((), device=dev)
    for name, p in clt.named_parameters():
        total = total + (p - base_params[name]).pow(2).sum()
    return lam * total
```

- [ ] **Step 4: Add the CLI flags and make `parse_args` accept argv**

In `clts/trainCLT.py`, change the signature `def parse_args() -> argparse.Namespace:` to:

```python
def parse_args(argv=None) -> argparse.Namespace:
```

and change its final line `return p.parse_args()` to `return p.parse_args(argv)`.

Immediately before that `return`, add:

```python
    # --- edit-CLT add-on (opt-in; defaults preserve legacy behavior) ----------
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Load CLT weights from this final/ dir instead of fresh "
                        "init (Method 2 fine-tune).")
    p.add_argument("--out-tag", type=str, default=None,
                   help="Replace the standalone/sweep folder in the run path "
                        "(disambiguates fine-tune variants).")
    p.add_argument("--target-ce-recovered", type=float, default=None,
                   help="Early-stop once eval ce_recovered >= this value.")
    p.add_argument("--plateau-patience", type=int, default=None,
                   help="Early-stop after this many consecutive sub-threshold "
                        "eval gains (needs --plateau-min-delta).")
    p.add_argument("--plateau-min-delta", type=float, default=None,
                   help="Min ce_recovered gain per eval to count as progress.")
    p.add_argument("--eval-every", type=int, default=None,
                   help="Override EVAL_EVERY (default 600) for fine eval cadence.")
    p.add_argument("--anchor-lambda", type=float, default=0.0,
                   help="L2 proximity penalty toward the resumed checkpoint.")
```

- [ ] **Step 5: Run the helper + flag tests, verify they pass**

Run: `pytest tests/test_trainclt_addon.py -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Wire the helpers into `train_one_run` (no new test; covered by smoke + invariance)**

In `clts/trainCLT.py`, `train_one_run`:

(a) Replace the run-dir block (currently L182-184):

```python
    sweep_folder = f"sweep-{sweep_id}" if sweep_id else "standalone"
    run_dir = STORAGE_ROOT / "clt_runs" / ARGS.model_name / sweep_folder / name
    final_dir = run_dir / "final"
```

with:

```python
    run_dir = _run_dir(STORAGE_ROOT, ARGS.model_name, name,
                       out_tag=getattr(ARGS, "out_tag", None), sweep_id=sweep_id)
    final_dir = run_dir / "final"
```

(b) Replace the CLT-build block (currently L207-211):

```python
    # Build CLT.
    clt = CrossLayerTranscoder(
        n_layers=model.cfg.n_layers, d_model=model.cfg.d_model, expansion=expansion,
    ).to(device)
    opt = torch.optim.Adam(clt.parameters(), lr=lr, betas=(0.9, 0.999))
```

with:

```python
    # Build CLT — fresh, or resumed from a checkpoint (Method 2 fine-tune).
    resume_from = getattr(ARGS, "resume_from", None)
    if resume_from:
        clt = CrossLayerTranscoder.load_from_dir(resume_from).to(device)
        assert clt.n_layers == model.cfg.n_layers and clt.d_model == model.cfg.d_model \
            and clt.d_transcoder == expansion * model.cfg.d_model, \
            f"resumed CLT {clt.n_layers}x{clt.d_model}x{clt.d_transcoder} != " \
            f"model {model.cfg.n_layers}x{model.cfg.d_model} exp{expansion}"
        print(f"[resume]  loaded CLT from {resume_from}")
    else:
        clt = CrossLayerTranscoder(
            n_layers=model.cfg.n_layers, d_model=model.cfg.d_model, expansion=expansion,
        ).to(device)
    anchor_lambda = float(getattr(ARGS, "anchor_lambda", 0.0) or 0.0)
    base_params = ({n: p.detach().clone() for n, p in clt.named_parameters()}
                  if anchor_lambda else None)
    opt = torch.optim.Adam(clt.parameters(), lr=lr, betas=(0.9, 0.999))
    ce_history = []
```

(c) Replace `EVAL_EVERY = 600` (L224) with:

```python
    EVAL_EVERY = getattr(ARGS, "eval_every", None) or 600
```

(d) Add the anchor term to the backward. Replace (L243-246):

```python
        losses = clt.compute_loss(x_list, y_list, l0_coefficient=lam)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        opt.step()
```

with:

```python
        losses = clt.compute_loss(x_list, y_list, l0_coefficient=lam)
        loss = losses["total"] + _anchor_penalty(clt, base_params, anchor_lambda) \
            if base_params is not None else losses["total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
```

(e) Add early-stop after the periodic eval logs (immediately after the `wandb.log({f"clt_eval/...})` call that ends at L274, still inside `if step > 0 and step % EVAL_EVERY == 0:`):

```python
            ce_history.append(ce["ce_recovered"])
            stop, reason = _should_stop(
                ce_history,
                target=getattr(ARGS, "target_ce_recovered", None),
                patience=getattr(ARGS, "plateau_patience", None),
                min_delta=getattr(ARGS, "plateau_min_delta", None),
            )
            if stop:
                print(f"[early-stop] {reason} at step {step}, "
                      f"ce_recovered={ce['ce_recovered']:.4f}")
                wandb.log({"early_stop/reason": reason,
                           "early_stop/step": step}, step=step)
                break
```

(The post-loop final eval + `save_to_dir` already runs after a `break`, so early-stopped runs still save normally.)

- [ ] **Step 7: Verify default-path invariance + full suite still green**

Run: `pytest tests/test_trainclt_addon.py -v && pytest -m "not integration" -q`
Expected: PASS. (No behavior change when new flags unset: `getattr(..., None)` everywhere, `base_params` None → original backward path.)

- [ ] **Step 8: Commit**

```bash
git add clts/trainCLT.py tests/test_trainclt_addon.py
git commit -m "feat(clt): opt-in resume/early-stop/out-tag/anchor flags for trainCLT"
```

---

### Task 2: Extend `scripts/train_clt_psc.sh` for the new flags

**Files:**
- Modify: `scripts/train_clt_psc.sh`
- Test: `tests/test_train_clt_psc_sh.py`

**Interfaces:**
- Produces: env vars `RESUME_FROM`, `TARGET_CE_RECOVERED`, `PLATEAU_PATIENCE`, `PLATEAU_MIN_DELTA`, `OUT_TAG`, `ANCHOR_LAMBDA`, `EVAL_EVERY` → appended `trainCLT.py` flags in the single-run command.

- [ ] **Step 1: Write a failing grep-based test**

Create `tests/test_train_clt_psc_sh.py`:

```python
import subprocess
from pathlib import Path

SH = Path(__file__).resolve().parent.parent / "scripts" / "train_clt_psc.sh"


def test_script_has_valid_bash_syntax():
    # bash -n parses without executing.
    subprocess.run(["bash", "-n", str(SH)], check=True)


def test_script_threads_addon_env_vars():
    text = SH.read_text()
    for token in ("RESUME_FROM", "TARGET_CE_RECOVERED", "PLATEAU_PATIENCE",
                  "PLATEAU_MIN_DELTA", "OUT_TAG", "ANCHOR_LAMBDA", "EVAL_EVERY",
                  "--resume-from", "--out-tag", "--target-ce-recovered",
                  "--plateau-patience", "--plateau-min-delta", "--eval-every",
                  "--anchor-lambda"):
        assert token in text, f"missing {token}"
```

- [ ] **Step 2: Run test, verify the env-var test fails**

Run: `pytest tests/test_train_clt_psc_sh.py -v`
Expected: `test_script_threads_addon_env_vars` FAILS (tokens absent); syntax test passes.

- [ ] **Step 3: Add the env→flag block**

In `scripts/train_clt_psc.sh`, in SECTION A (after the `LR=` line, ~L65), add:

```bash
# --- edit-CLT add-on (opt-in; unset -> legacy behavior) ---------------------
RESUME_FROM="${RESUME_FROM:-}"            # final/ dir of a CLT to fine-tune from
OUT_TAG="${OUT_TAG:-}"                     # run-path folder (variant disambiguator)
TARGET_CE_RECOVERED="${TARGET_CE_RECOVERED:-}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-}"
PLATEAU_MIN_DELTA="${PLATEAU_MIN_DELTA:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
ANCHOR_LAMBDA="${ANCHOR_LAMBDA:-}"
ADDON_ARGS=()
[ -n "$RESUME_FROM" ]         && ADDON_ARGS+=(--resume-from "$RESUME_FROM")
[ -n "$OUT_TAG" ]             && ADDON_ARGS+=(--out-tag "$OUT_TAG")
[ -n "$TARGET_CE_RECOVERED" ] && ADDON_ARGS+=(--target-ce-recovered "$TARGET_CE_RECOVERED")
[ -n "$PLATEAU_PATIENCE" ]    && ADDON_ARGS+=(--plateau-patience "$PLATEAU_PATIENCE")
[ -n "$PLATEAU_MIN_DELTA" ]   && ADDON_ARGS+=(--plateau-min-delta "$PLATEAU_MIN_DELTA")
[ -n "$EVAL_EVERY" ]          && ADDON_ARGS+=(--eval-every "$EVAL_EVERY")
[ -n "$ANCHOR_LAMBDA" ]       && ADDON_ARGS+=(--anchor-lambda "$ANCHOR_LAMBDA")
```

In the preflight loop (the `for p in "$MODEL_DIR/config.json" …` block, ~L140), add a resume check right after it:

```bash
if [ -n "$RESUME_FROM" ] && [ ! -e "$RESUME_FROM/config.yaml" ]; then
    echo "  MISSING (resume): $RESUME_FROM/config.yaml" >&2; fail=1
fi
```

In the single-run command builder (the `cmd=( … )` array, ~L188-199), add the addon args before the closing `)` — change the `--n-examples "$N_EXAMPLES")` line to:

```bash
    --n-examples "$N_EXAMPLES"
    ${ADDON_ARGS[@]+"${ADDON_ARGS[@]}"})
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_train_clt_psc_sh.py -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_clt_psc.sh tests/test_train_clt_psc_sh.py
git commit -m "feat(psc): thread edit-CLT add-on flags through train_clt_psc.sh"
```

---

### Task 3: `edit_clt_config.py` — config, path resolution, manifest

**Files:**
- Create: `clts/edit_clt/__init__.py` (empty), `clts/edit_clt/edit_clt_config.py`
- Test: `tests/test_edit_clt_config.py`

**Interfaces:**
- Produces:
  - `FactSpec(person=0, fields=("month",), new_values={"month":"July"}, edit_template=0, model_name="grid-L4-H6")` with `.slug() -> str` (e.g. `"p0-month-jul"`).
  - `MethodConfig(key, out_tag, expansion=16, l0=2.0, lr=1e-4, epochs=50, n_examples=10000, resume_from=None, target_ce_recovered=None, plateau_patience=None, plateau_min_delta=None, eval_every=None, anchor_lambda=0.0)`.
  - `expected_clt_dir(storage_root, model_name, out_tag, expansion, l0, lr, epochs, n_examples) -> Path`.
  - `EditCLTConfig` with `.edited_model_name() -> str`, `.edited_model_dir(repo_root) -> Path`.
  - `default_config(repo_root, storage_root, base_clt_dir, data_dir) -> EditCLTConfig`.
  - `write_manifest(path, manifest: dict) -> None`, `read_manifest(path) -> dict`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_edit_clt_config.py`:

```python
import json
from pathlib import Path
from clts.edit_clt import edit_clt_config as cfg


def test_factspec_slug():
    fs = cfg.FactSpec()
    assert fs.slug() == "p0-month-jul"
    fs2 = cfg.FactSpec(person=5, fields=("month",), new_values={"month": "March"})
    assert fs2.slug() == "p5-month-mar"


def test_expected_clt_dir_matches_apricot_layout():
    root = Path("/store")
    # baseline apricot config -> the exact on-disk dir name
    d = cfg.expected_clt_dir(root, "grid-L4-H6", "standalone",
                             expansion=16, l0=2.0, lr=1e-4, epochs=50, n_examples=10000)
    assert d == root / "clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final"


def test_expected_clt_dir_method2_variant():
    root = Path("/store")
    d = cfg.expected_clt_dir(root, "grid-L4-H6-edit-p0-month-jul", "method2-v2-basic",
                             expansion=16, l0=2.0, lr=2e-5, epochs=5, n_examples=10000)
    assert d.as_posix().endswith(
        "clt_runs/grid-L4-H6-edit-p0-month-jul/method2-v2-basic/mult16_l02_lr2e-05_ep5_n10000/final")


def test_edited_model_name_and_dir():
    c = cfg.default_config(Path("/repo"), Path("/store"),
                           base_clt_dir="/store/clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final",
                           data_dir="/repo/data/bioS_N-Bd_final_grid")
    assert c.edited_model_name() == "grid-L4-H6-edit-p0-month-jul"
    assert c.edited_model_dir(Path("/repo")) == Path("/repo/model/grid-L4-H6-edit-p0-month-jul")


def test_default_config_has_method1_and_two_method2_variants():
    c = cfg.default_config(Path("/repo"), Path("/store"),
                           base_clt_dir="/b/final", data_dir="/d")
    keys = {m.key for m in c.methods}
    assert "m1_scratch" in keys
    assert "m2-v2-basic" in keys
    assert "m2-v2-fixed" in keys
    m1 = next(m for m in c.methods if m.key == "m1_scratch")
    assert (m1.expansion, m1.l0, m1.lr, m1.epochs, m1.n_examples) == (16, 2.0, 1e-4, 50, 10000)
    assert m1.resume_from is None
    m2 = next(m for m in c.methods if m.key == "m2-v2-basic")
    assert m2.resume_from == "/b/final" and m2.lr == 2e-5 and m2.out_tag == "method2-v2-basic"


def test_manifest_roundtrip(tmp_path):
    man = {"exp_id": "x", "fact": {"person": 0}, "target_stats": {"ce_recovered": 0.5}}
    p = tmp_path / "manifest.json"
    cfg.write_manifest(p, man)
    assert cfg.read_manifest(p) == man
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/test_edit_clt_config.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the module**

Create `clts/edit_clt/__init__.py` (empty). Create `clts/edit_clt/edit_clt_config.py`:

```python
"""Single source of truth for the edit-CLT experiment: the fact to edit, the
method/variant matrix, path resolution, and manifest I/O. Pure + dependency-light
so it imports in both the training and circuit-tracer envs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

_MONTH_ABBR = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr",
    "may": "may", "june": "jun", "july": "jul", "august": "aug",
    "september": "sep", "october": "oct", "november": "nov", "december": "dec",
}


@dataclass
class FactSpec:
    person: int = 0
    fields: tuple = ("month",)
    new_values: dict = field(default_factory=lambda: {"month": "July"})
    edit_template: int = 0
    model_name: str = "grid-L4-H6"

    def slug(self) -> str:
        parts = []
        for f in self.fields:
            v = str(self.new_values[f]).lower()
            v = _MONTH_ABBR.get(v, v)
            parts.append(f"{f}-{v}")
        return f"p{self.person}-" + "-".join(parts)


@dataclass
class MethodConfig:
    key: str
    out_tag: str
    expansion: int = 16
    l0: float = 2.0
    lr: float = 1e-4
    epochs: int = 50
    n_examples: int = 10000
    resume_from: str | None = None
    target_ce_recovered: float | None = None
    plateau_patience: int | None = None
    plateau_min_delta: float | None = None
    eval_every: int | None = None
    anchor_lambda: float = 0.0


def _trial_name(expansion, l0, lr, epochs, n_examples) -> str:
    # MUST mirror clts.trainCLT.trial_name.
    return f"mult{expansion}_l0{l0:g}_lr{lr:g}_ep{epochs}_n{n_examples}"


def expected_clt_dir(storage_root, model_name, out_tag,
                     expansion, l0, lr, epochs, n_examples) -> Path:
    name = _trial_name(expansion, l0, lr, epochs, n_examples)
    return Path(storage_root) / "clt_runs" / model_name / out_tag / name / "final"


@dataclass
class EditCLTConfig:
    fact: FactSpec
    base_clt_dir: str
    data_dir: str
    storage_root: str
    methods: list
    trace_prompt_template: str = "{first} {last} was born on the"
    enc_hook_template: str = "blocks.{layer}.hook_resid_mid"
    dec_hook_template: str = "blocks.{layer}.hook_mlp_out"

    def edited_model_name(self) -> str:
        return f"{self.fact.model_name}-edit-{self.fact.slug()}"

    def edited_model_dir(self, repo_root) -> Path:
        return Path(repo_root) / "model" / self.edited_model_name()


def default_config(repo_root, storage_root, base_clt_dir, data_dir) -> EditCLTConfig:
    fact = FactSpec()
    methods = [
        MethodConfig(key="m1_scratch", out_tag="standalone",
                     expansion=16, l0=2.0, lr=1e-4, epochs=50, n_examples=10000),
        MethodConfig(key="m2-v2-basic", out_tag="method2-v2-basic",
                     expansion=16, l0=2.0, lr=2e-5, epochs=5, n_examples=10000,
                     resume_from=str(base_clt_dir),
                     target_ce_recovered=None,   # filled from target_stats at submit time
                     plateau_patience=3, plateau_min_delta=0.01, eval_every=100),
        MethodConfig(key="m2-v2-fixed", out_tag="method2-v2-fixed",
                     expansion=16, l0=2.0, lr=2e-5, epochs=2, n_examples=10000,
                     resume_from=str(base_clt_dir), eval_every=100),
    ]
    return EditCLTConfig(fact=fact, base_clt_dir=str(base_clt_dir),
                         data_dir=str(data_dir), storage_root=str(storage_root),
                         methods=methods)


def write_manifest(path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def read_manifest(path) -> dict:
    return json.loads(Path(path).read_text())
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_edit_clt_config.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add clts/edit_clt/__init__.py clts/edit_clt/edit_clt_config.py tests/test_edit_clt_config.py
git commit -m "feat(edit-clt): experiment config + path resolution + manifest I/O"
```

---

### Task 4: `prepare_edited_model.py` — MEMIT edit → save → verify

**Files:**
- Create: `clts/edit_clt/prepare_edited_model.py`
- Test: `tests/test_prepare_edited_model.py`

**Interfaces:**
- Consumes: `FactSpec` (Task 3); `run_single_edit` from `FactEditing/single-edit/run_single_edit.py` (loaded via importlib because the dir name has a hyphen).
- Produces: `make_edited_model(fact: FactSpec, edited_model_dir: Path, *, factediting_root: Path, device="cpu", controls=250, save_metrics_to=None) -> dict` returning `{"edited_model_dir": str, "edit_metrics": dict, "verified": bool}`.

- [ ] **Step 1: Write the integration test**

Create `tests/test_prepare_edited_model.py`:

```python
import shutil
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent
FACTEDIT = REPO.parent / "FactEditing"
MODEL = REPO / "model" / "grid-L4-H6"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not (MODEL / "config.json").exists() or not FACTEDIT.exists(),
                    reason="needs grid-L4-H6 weights + FactEditing repo")
def test_make_edited_model_saves_and_verifies(tmp_path):
    from clts.edit_clt import edit_clt_config as cfg
    from clts.edit_clt.prepare_edited_model import make_edited_model

    out = tmp_path / "grid-L4-H6-edit-p0-month-jul"
    res = make_edited_model(
        cfg.FactSpec(), out, factediting_root=FACTEDIT,
        device="cpu", controls=2,   # tiny for speed
    )
    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists()
    assert res["verified"] is True
    shutil.rmtree(out, ignore_errors=True)
```

- [ ] **Step 2: Run, verify it fails**

Run: `pytest tests/test_prepare_edited_model.py -v`
Expected: FAIL (`ModuleNotFoundError: clts.edit_clt.prepare_edited_model`) or skip if artifacts absent. If skipped locally, proceed (the smoke in Notebook 1 covers it).

- [ ] **Step 3: Implement**

Create `clts/edit_clt/prepare_edited_model.py`:

```python
"""Run one MEMIT edit and persist the edited model as a first-class HF model dir
that trainCLT + circuit-tracer can load. Runs in the training env (imports the
FactEditing stack, not circuit_tracer)."""
from __future__ import annotations

import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _pushd(path):
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def _load_run_single_edit(factediting_root: Path):
    path = Path(factediting_root) / "single-edit" / "run_single_edit.py"
    spec = importlib.util.spec_from_file_location("rse_edit_clt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_edited_model(fact, edited_model_dir, *, factediting_root,
                      device="cpu", controls=250, save_metrics_to=None) -> dict:
    # Resolve to absolute BEFORE chdir: factedit's vendored MEMIT resolves
    # globals.yml relative to CWD, so the edit must run from the FactEditing root.
    edited_model_dir = Path(edited_model_dir).resolve()
    factediting_root = Path(factediting_root).resolve()

    with _pushd(factediting_root):
        rse = _load_run_single_edit(factediting_root)
        result, edited, _orig = rse.run_single_edit(
            person=fact.person, fields=tuple(fact.fields),
            new_values=dict(fact.new_values), edit_template=fact.edit_template,
            controls=controls, device=device, model_name=fact.model_name, save=False,
        )
        edited_model_dir.mkdir(parents=True, exist_ok=True)
        edited.save_pretrained(str(edited_model_dir))
        verified = _verify_edit(edited_model_dir, fact, device)   # CWD = FactEditing root

    metrics = result.to_df().to_dict(orient="records")
    if save_metrics_to:
        Path(save_metrics_to).write_text(json.dumps(metrics, indent=2))
    return {"edited_model_dir": str(edited_model_dir),
            "edit_metrics": metrics, "verified": bool(verified)}


def _verify_edit(model_dir, fact, device) -> bool:
    """Reload the saved model from disk (exactly as trainCLT/circuit-tracer will)
    and check the edited field scores positive on the edit template. Caller must
    have CWD = FactEditing root (factedit import resolves globals.yml from CWD)."""
    import sys
    import torch
    sys.path.insert(0, os.getcwd())
    import factedit as fe  # noqa: E402
    from transformers import LlamaForCausalLM

    _model_unused, toks = fe.load_lm4(fact.model_name, device)   # tokenizer/probe only
    model = LlamaForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32).to(device).eval()
    people = fe.load_people()
    exp = fe.expected_person(people[fact.person], dict(fact.new_values))
    score = fe.score_full(model, toks, exp, fe.TEMPLATES[fact.edit_template], device)
    return bool(score.get("FP", 0) or score.get("LP", 0) or score.get("MP", 0))
```

- [ ] **Step 4: Run the integration test (where artifacts exist)**

Run: `pytest tests/test_prepare_edited_model.py -v -m integration`
Expected: PASS (or SKIP where weights/FactEditing absent — then validate via Notebook 1 smoke on a machine with artifacts).

- [ ] **Step 5: Commit**

```bash
git add clts/edit_clt/prepare_edited_model.py tests/test_prepare_edited_model.py
git commit -m "feat(edit-clt): MEMIT edit -> saved HF model dir with verification"
```

---

### Task 5: `drift.py` — feature-drift metrics

**Files:**
- Create: `clts/edit_clt/drift.py`
- Test: `tests/test_drift.py`

**Interfaces:**
- Consumes: `CrossLayerTranscoder` (clts/clt.py); a circuit-tracer `graph` object (only its `.active_features`, `.selected_features`).
- Produces:
  - `decoder_cosine_drift(clt_a, clt_b) -> dict` with `cosine_L{L}` (torch.Tensor [d_t]), `mean_cosine` (float), `frac_moved` (float, fraction with cosine < 0.9).
  - `active_feature_sets(graph) -> set[tuple[int,int]]` of `(layer, feat_idx)`.
  - `active_feature_overlap(graph_a, graph_b) -> dict` with `jaccard`, `n_a`, `n_b`, `appeared`, `disappeared` (counts) + `appeared_set`, `disappeared_set`.
  - `match_features(clt_a, clt_b, layer) -> dict` with `match_idx` (torch.LongTensor [d_t]) and `match_cosine` (torch.Tensor [d_t]) — greedy argmax decoder-cosine matching for from-scratch CLTs.

- [ ] **Step 1: Write failing tests**

Create `tests/test_drift.py`:

```python
import torch
from clts.clt import CrossLayerTranscoder
from clts.edit_clt import drift


def _clt():
    torch.manual_seed(0)
    return CrossLayerTranscoder(n_layers=2, d_model=4, expansion=2)


def test_decoder_cosine_drift_identity_is_one():
    import copy
    a = _clt()
    bclt = copy.deepcopy(a)
    out = drift.decoder_cosine_drift(a, bclt)
    assert out["mean_cosine"] > 0.999
    assert out["frac_moved"] == 0.0
    assert out["cosine_L0"].shape[0] == a.d_transcoder


def test_decoder_cosine_drift_detects_change():
    import copy
    a = _clt()
    bclt = copy.deepcopy(a)
    with torch.no_grad():
        bclt.W_dec[0] += 5.0           # large perturbation to layer-0 decoder
    out = drift.decoder_cosine_drift(a, bclt)
    assert out["mean_cosine"] < 0.999
    assert out["frac_moved"] > 0.0


class _FakeGraph:
    def __init__(self, triples):
        self.active_features = torch.tensor(triples, dtype=torch.long)
        self.selected_features = torch.arange(len(triples))


def test_active_feature_overlap():
    # (layer, pos, feat)
    g_a = _FakeGraph([[0, 1, 10], [1, 1, 20], [1, 2, 20]])  # -> {(0,10),(1,20)}
    g_b = _FakeGraph([[1, 1, 20], [0, 1, 30]])              # -> {(1,20),(0,30)}
    out = drift.active_feature_overlap(g_a, g_b)
    assert out["jaccard"] == 1 / 3          # intersection {(1,20)} / union of 3
    assert out["disappeared"] == 1          # (0,10) gone in b
    assert out["appeared"] == 1             # (0,30) new in b


def test_match_features_self_is_identity():
    a = _clt()
    out = drift.match_features(a, a, layer=0)
    assert torch.equal(out["match_idx"], torch.arange(a.d_transcoder))
    assert out["match_cosine"].min() > 0.999
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/test_drift.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

Create `clts/edit_clt/drift.py`:

```python
"""Feature-drift metrics between CLTs, and active-feature diffs between graphs.
Dependency-light (torch only) so it imports in both envs."""
from __future__ import annotations

import torch


def _summed_decoder(clt, layer) -> torch.Tensor:
    """Per-feature decoder vector for a source layer, summed over downstream
    targets: W_dec[layer] is [d_t, N-layer, D] -> [d_t, D]."""
    return clt.W_dec[layer].detach().sum(dim=1)


def decoder_cosine_drift(clt_a, clt_b, moved_threshold: float = 0.9) -> dict:
    assert clt_a.n_layers == clt_b.n_layers and clt_a.d_transcoder == clt_b.d_transcoder
    out, all_cos = {}, []
    for L in range(clt_a.n_layers):
        va = _summed_decoder(clt_a, L)
        vb = _summed_decoder(clt_b, L)
        cos = torch.nn.functional.cosine_similarity(va, vb, dim=-1)  # [d_t]
        out[f"cosine_L{L}"] = cos
        all_cos.append(cos)
    cat = torch.cat(all_cos)
    out["mean_cosine"] = float(cat.mean())
    out["frac_moved"] = float((cat < moved_threshold).float().mean())
    return out


def active_feature_sets(graph) -> set:
    sel = graph.active_features[graph.selected_features]   # [n_sel, 3] = (layer,pos,feat)
    return {(int(layer), int(feat)) for layer, _pos, feat in sel.tolist()}


def active_feature_overlap(graph_a, graph_b) -> dict:
    a = active_feature_sets(graph_a)
    b = active_feature_sets(graph_b)
    inter, union = a & b, a | b
    return {
        "jaccard": (len(inter) / len(union)) if union else 0.0,
        "n_a": len(a), "n_b": len(b),
        "appeared": len(b - a), "disappeared": len(a - b),
        "appeared_set": sorted(b - a), "disappeared_set": sorted(a - b),
    }


def match_features(clt_a, clt_b, layer) -> dict:
    """Greedy argmax matching of clt_a features to clt_b features by summed-decoder
    cosine (for from-scratch CLTs whose feature indices don't align). Returns the
    best-match index + cosine per clt_a feature."""
    va = torch.nn.functional.normalize(_summed_decoder(clt_a, layer), dim=-1)
    vb = torch.nn.functional.normalize(_summed_decoder(clt_b, layer), dim=-1)
    sim = va @ vb.T                              # [d_t, d_t]
    match_cos, match_idx = sim.max(dim=-1)
    return {"match_idx": match_idx, "match_cosine": match_cos}
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_drift.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add clts/edit_clt/drift.py tests/test_drift.py
git commit -m "feat(edit-clt): CLT feature-drift + active-feature diff metrics"
```

---

### Task 6: `compare.py` — graph configs + comparison tables

**Files:**
- Create: `clts/edit_clt/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `read_manifest` (Task 3); `drift.active_feature_overlap` (Task 5); `clts.build_attribution_graph.build_graph` (circuit-tracer env only — imported lazily inside `build_or_load_graph`).
- Produces:
  - `GraphConfig(key, model_dir, clt_dir, scan_name)`.
  - `configs_from_manifest(manifest: dict) -> list[GraphConfig]` (yields `baseline_orig`, `m3_stale`, plus one per trained method present).
  - `build_or_load_graph(gc, data_dir, graph_root, slug, prompt, target=None, device="cpu") -> dict` (wraps `build_graph`, graph_dir = `graph_root/<gc.scan_name>/<slug>`).
  - `comparison_table(reports: dict[str, dict]) -> pandas.DataFrame`.
  - `feature_diff_table(graphs: dict[str, object], baseline_key="baseline_orig") -> pandas.DataFrame`.

- [ ] **Step 1: Write failing tests (pure parts only)**

Create `tests/test_compare.py`:

```python
import torch
from clts.edit_clt import compare


def test_configs_from_manifest_builds_four_when_all_present():
    man = {
        "edited_model_dir": "/m/edit",
        "orig_model_dir": "/m/orig",
        "base_clt_dir": "/c/base/final",
        "methods": {
            "m1_scratch": {"expected_clt_dir": "/c/m1/final", "status": "done"},
            "m2-v2-basic": {"expected_clt_dir": "/c/m2/final", "status": "done"},
        },
    }
    gcs = {g.key: g for g in compare.configs_from_manifest(man)}
    assert gcs["baseline_orig"].model_dir == "/m/orig"
    assert gcs["baseline_orig"].clt_dir == "/c/base/final"
    assert gcs["m3_stale"].model_dir == "/m/edit"
    assert gcs["m3_stale"].clt_dir == "/c/base/final"
    assert gcs["m1_scratch"].clt_dir == "/c/m1/final"
    assert gcs["m2-v2-basic"].model_dir == "/m/edit"


def test_comparison_table_columns_and_rows():
    reports = {
        "baseline_orig": {"replacement_score": 0.62, "completeness_score": 0.7,
                          "error_influence_share": 0.38, "top_logit_token": " February",
                          "target_logit_prob": 0.8, "n_feature_nodes_after_pruning": 120},
        "m3_stale": {"replacement_score": 0.4, "completeness_score": 0.5,
                     "error_influence_share": 0.6, "top_logit_token": " July",
                     "target_logit_prob": 0.55, "n_feature_nodes_after_pruning": 90},
    }
    df = compare.comparison_table(reports)
    assert list(df.index) == ["baseline_orig", "m3_stale"]
    assert "replacement_score" in df.columns
    assert "error_influence_share" in df.columns
    assert df.loc["m3_stale", "error_influence_share"] == 0.6


class _FakeGraph:
    def __init__(self, triples):
        self.active_features = torch.tensor(triples, dtype=torch.long)
        self.selected_features = torch.arange(len(triples))


def test_feature_diff_table_against_baseline():
    graphs = {
        "baseline_orig": _FakeGraph([[0, 1, 10], [1, 1, 20]]),
        "m3_stale": _FakeGraph([[1, 1, 20], [0, 1, 30]]),
    }
    df = compare.feature_diff_table(graphs, baseline_key="baseline_orig")
    assert df.loc["m3_stale", "appeared"] == 1
    assert df.loc["m3_stale", "disappeared"] == 1
    # baseline row vs itself is trivial
    assert df.loc["baseline_orig", "jaccard"] == 1.0
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/test_compare.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

Create `clts/edit_clt/compare.py`:

```python
"""Graph-comparison helpers for Notebook 2. The build wrapper imports
circuit-tracer lazily so the pure table/diff helpers import in any env."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from clts.edit_clt import drift

_REPORT_COLS = ["replacement_score", "completeness_score", "error_influence_share",
                "target_logit_prob", "top_logit_token", "n_feature_nodes_after_pruning"]


@dataclass
class GraphConfig:
    key: str
    model_dir: str
    clt_dir: str
    scan_name: str


def configs_from_manifest(manifest: dict) -> list:
    edited = manifest["edited_model_dir"]
    orig = manifest["orig_model_dir"]
    base = manifest["base_clt_dir"]
    gcs = [
        GraphConfig("baseline_orig", orig, base, "baseline_orig"),
        GraphConfig("m3_stale", edited, base, "m3_stale"),
    ]
    for key, m in manifest.get("methods", {}).items():
        clt = m.get("expected_clt_dir")
        if clt:
            gcs.append(GraphConfig(key, edited, clt, key))
    return gcs


def build_or_load_graph(gc, data_dir, graph_root, slug, prompt, target=None,
                        device="cpu") -> dict:
    from clts.build_attribution_graph import build_graph
    graph_dir = Path(graph_root) / gc.scan_name / slug
    return build_graph(
        model_dir=gc.model_dir, clt_dir=gc.clt_dir, data_dir=data_dir,
        scan_name=gc.scan_name, graph_dir=str(graph_dir), slug=slug,
        prompt=prompt, target=target, device=device,
    )


def comparison_table(reports: dict) -> pd.DataFrame:
    rows = {k: {c: r.get(c) for c in _REPORT_COLS} for k, r in reports.items()}
    return pd.DataFrame.from_dict(rows, orient="index")[_REPORT_COLS]


def feature_diff_table(graphs: dict, baseline_key="baseline_orig") -> pd.DataFrame:
    base = graphs[baseline_key]
    rows = {}
    for key, g in graphs.items():
        o = drift.active_feature_overlap(base, g)
        rows[key] = {"jaccard": o["jaccard"], "n_features": o["n_b"],
                     "appeared": o["appeared"], "disappeared": o["disappeared"]}
    return pd.DataFrame.from_dict(rows, orient="index")
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_compare.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add clts/edit_clt/compare.py tests/test_compare.py
git commit -m "feat(edit-clt): graph configs + comparison/feature-diff tables"
```

---

### Task 7: `submit_edit_clt.sh` — PSC driver

**Files:**
- Create: `clts/edit_clt/submit_edit_clt.sh`

**Interfaces:**
- Consumes: `scripts/train_clt_psc.sh` (Task 2), env vars it understands.
- Produces: `sbatch` jobs for M1 + each M2 variant; `--test` mode for a cheap GPU validation.

- [ ] **Step 1: Implement the driver**

Create `clts/edit_clt/submit_edit_clt.sh`:

```bash
#!/bin/bash
# Submit the edit-CLT training jobs (Method 1 + Method 2 variants) on PSC by
# setting env vars and sbatch-ing the shared scripts/train_clt_psc.sh.
#   bash clts/edit_clt/submit_edit_clt.sh            # full runs
#   bash clts/edit_clt/submit_edit_clt.sh --test     # cheap GPU smoke (N=1000, ep=3)
# Run from the Interp_LM4 repo root on a Bridges-2 login node.
set -euo pipefail

TEST=0; [ "${1:-}" = "--test" ] && TEST=1
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# ---- experiment paths (edit to match your manifest) ------------------------
REMOTE_BASE="/jet/home/friedmae/data_storage/LM4_Results"
EDITED_MODEL_NAME="${EDITED_MODEL_NAME:-grid-L4-H6-edit-p0-month-jul}"
EDITED_MODEL_DIR="${EDITED_MODEL_DIR:-$REMOTE_BASE/runResults/edited/$EDITED_MODEL_NAME}"
DATA_DIR="${DATA_DIR:-$REMOTE_BASE/Data/bioS_N-Bd_final_grid}"
BASE_CLT_DIR="${BASE_CLT_DIR:-$REMOTE_BASE/clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final}"
TARGET_CE="${TARGET_CE_RECOVERED:-}"   # from manifest target_stats; optional

export CLT_STORAGE_ROOT="$REMOTE_BASE"
COMMON=(--export=ALL --account=cis240072p --partition=GPU-shared --gres=gpu:1)

# Cheap-test overrides.
if [ "$TEST" = 1 ]; then N=1000; EP1=3; EP2=1; WALL="00:30:00"; TAG="-test";
else N=10000; EP1=50; EP2=5; WALL="04:00:00"; TAG=""; fi

submit() {  # name extra_env...
    local name="$1"; shift
    echo "submitting $name ..."
    env MODEL_NAME="$EDITED_MODEL_NAME" MODEL_DIR="$EDITED_MODEL_DIR" \
        DATA_DIR="$DATA_DIR" CONDA_ENV=lm4-ct SWEEP=0 \
        EXPANSION=16 L0=2 N_EXAMPLES="$N" CONTEXT_SIZE=512 "$@" \
        sbatch "${COMMON[@]}" --job-name="$name" --time="$WALL" \
               scripts/train_clt_psc.sh
}

# Method 1: from scratch, apricot config.
submit "clt-edit-m1${TAG}" LR=1e-4 EPOCHS="$EP1"

# Method 2 v2-basic: fine-tune, plateau-or-parity.
submit "clt-edit-m2basic${TAG}" LR=2e-5 EPOCHS="$EP2" \
    RESUME_FROM="$BASE_CLT_DIR" OUT_TAG="method2-v2-basic${TAG}" \
    PLATEAU_PATIENCE=3 PLATEAU_MIN_DELTA=0.01 EVAL_EVERY=100 \
    ${TARGET_CE:+TARGET_CE_RECOVERED="$TARGET_CE"}

# Method 2 v2-fixed: fine-tune, fixed short budget.
submit "clt-edit-m2fixed${TAG}" LR=2e-5 EPOCHS=2 \
    RESUME_FROM="$BASE_CLT_DIR" OUT_TAG="method2-v2-fixed${TAG}" EVAL_EVERY=100

echo "submitted. watch:  squeue -u \$USER"
```

- [ ] **Step 2: Syntax-check + make executable**

Run: `bash -n clts/edit_clt/submit_edit_clt.sh && chmod +x clts/edit_clt/submit_edit_clt.sh && echo OK`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add clts/edit_clt/submit_edit_clt.sh
git commit -m "feat(edit-clt): PSC submit driver for M1 + M2 variants (+--test)"
```

---

### Task 8: Notebook 1 — `01_edit_and_train.ipynb`

**Files:**
- Create: `clts/edit_clt/01_edit_and_train.ipynb`

**Interfaces:** consumes Tasks 3 (config/manifest), 4 (make_edited_model), 1+2+7 (training/submit). Runs in the **training env**.

- [ ] **Step 1: Create the notebook with these cells (markdown `md:` / code `py:`)**

```
md: # 01 — Fact edit + CLT (re)training (Methods 1 & 2)
    Runs the MEMIT edit, saves the edited model, computes the apricot target
    stats, smoke-tests the training add-on locally, and emits/submits PSC jobs.

py: # --- setup ---
    import os, sys, socket, json
    from pathlib import Path
    REPO = Path.cwd()
    while REPO.name != "Interp_LM4" and REPO != REPO.parent: REPO = REPO.parent
    sys.path.insert(0, str(REPO))
    FACTEDIT = REPO.parent / "FactEditing"
    from clts.storage import storage_root
    from clts.edit_clt import edit_clt_config as cfg
    from clts.edit_clt.prepare_edited_model import make_edited_model
    ON_PSC = Path("/jet/home/friedmae").exists()
    STORE = storage_root()
    print("repo:", REPO, "| on_psc:", ON_PSC, "| storage:", STORE)

py: # --- experiment config ---
    BASE_CLT = STORE / "clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final"
    DATA_DIR = REPO / "data/bioS_N-Bd_final_grid"
    C = cfg.default_config(REPO, STORE, base_clt_dir=BASE_CLT, data_dir=DATA_DIR)
    EXP_ID = f"{C.edited_model_name()}"
    print("edited model:", C.edited_model_name())
    for m in C.methods: print(" ", m.key, "->", m.out_tag, "lr", m.lr, "ep", m.epochs)

py: # --- baseline target stats: apricot ce_recovered/nmse on the ORIGINAL model ---
    import torch
    from clts.tl_model import build_hooked_transformer
    from clts.clt import CrossLayerTranscoder
    from clts.evalCLT import capture_activations, compute_layer_metrics, ce_recovered_full
    from clts.export_tokenizer import ensure_hf_tokenizer
    from util.condensed_tokenizer import CondensedTokenizer
    from util.bio_sampler import BioSampler
    from util.diverse_subset import DiverseBioSubset
    orig_model = build_hooked_transformer(REPO / "model/grid-L4-H6", "cpu", torch.float32)
    base_clt = CrossLayerTranscoder.load_from_dir(BASE_CLT).to("cpu")
    tok = CondensedTokenizer.from_remap_path(DATA_DIR / "old_to_new.json")
    sub = DiverseBioSubset(BioSampler(DATA_DIR / "people.json", fields=("birthday",), seed=1),
                           tok, context_size=512, seed=1)
    ev = torch.tensor(sub.to_hf_dataset(64, verbose=False)["input_ids"])
    x, y = capture_activations(orig_model, ev)
    target_stats = {**compute_layer_metrics(base_clt, x, y),
                    **ce_recovered_full(orig_model, base_clt, ev)}
    print("apricot target ce_recovered:", round(target_stats["ce_recovered"], 4))

py: # --- run the MEMIT edit + save the edited model dir ---
    edited_dir = C.edited_model_dir(REPO)
    info = make_edited_model(C.fact, edited_dir, factediting_root=FACTEDIT,
                             device="cpu", controls=25)   # raise controls for the real run
    print("verified:", info["verified"], "| saved:", info["edited_model_dir"])
    import pandas as pd; pd.DataFrame(info["edit_metrics"])

py: # --- LOCAL CPU smoke test of the training add-on (tiny budget) ---
    # Validates Method 1 (scratch) + Method 2 (resume) code paths end-to-end.
    import subprocess
    def run(tag, *extra):
        cmd = [sys.executable, "clts/trainCLT.py",
               "--model-dir", str(edited_dir), "--data-dir", str(DATA_DIR),
               "--model-name", C.edited_model_name(),
               "--expansion","16","--l0","2","--n-examples","200","--epochs","1",
               "--eval-every","50", *extra]
        print(">>", tag)
        subprocess.run(cmd, cwd=REPO, check=True,
                       env={**os.environ, "CLT_STORAGE_ROOT": str(STORE), "WANDB_MODE": "disabled"})
    run("m1-smoke", "--lr","1e-4","--out-tag","smoke-m1")
    run("m2-smoke", "--lr","2e-5","--out-tag","smoke-m2",
        "--resume-from", str(BASE_CLT), "--plateau-patience","2","--plateau-min-delta","0.01")

py: # --- emit / submit PSC jobs ---
    target_ce = round(target_stats["ce_recovered"], 4)
    if ON_PSC:
        env = {**os.environ, "EDITED_MODEL_NAME": C.edited_model_name(),
               "EDITED_MODEL_DIR": str(edited_dir), "DATA_DIR": str(DATA_DIR),
               "BASE_CLT_DIR": str(BASE_CLT), "TARGET_CE_RECOVERED": str(target_ce)}
        subprocess.run(["bash","clts/edit_clt/submit_edit_clt.sh","--test"], cwd=REPO, env=env, check=True)
        print("submitted --test; after it succeeds, rerun without --test for full runs")
    else:
        print("Run on PSC:\n  git push  # then on PSC: git pull")
        print(f"  EDITED_MODEL_NAME={C.edited_model_name()} \\\n  "
              f"EDITED_MODEL_DIR=<psc edited dir> DATA_DIR=<psc data> \\\n  "
              f"BASE_CLT_DIR=<psc apricot final> TARGET_CE_RECOVERED={target_ce} \\\n  "
              f"bash clts/edit_clt/submit_edit_clt.sh --test")

py: # --- write the manifest (consumed by Notebook 2) ---
    methods = {m.key: {"out_tag": m.out_tag, "config": vars(m),
                       "expected_clt_dir": str(cfg.expected_clt_dir(
                           STORE, C.edited_model_name(), m.out_tag,
                           m.expansion, m.l0, m.lr, m.epochs, m.n_examples)),
                       "status": "submitted" if ON_PSC else "pending"}
               for m in C.methods}
    first = BioSampler(DATA_DIR / "people.json", fields=("birthday",), seed=0).people[C.fact.person]
    trace_prompt = C.trace_prompt_template.format(first=first["first_name"], last=first["last_name"])
    manifest = {"exp_id": EXP_ID, "fact": vars(C.fact),
                "orig_model_dir": str(REPO / "model/grid-L4-H6"),
                "edited_model_dir": str(edited_dir), "base_clt_dir": str(BASE_CLT),
                "data_dir": str(DATA_DIR), "target_stats": target_stats,
                "trace_prompt": trace_prompt, "methods": methods}
    man_path = STORE / "edit_experiments" / EXP_ID / "manifest.json"
    cfg.write_manifest(man_path, manifest)
    print("manifest:", man_path)

md: ## Next steps
    1. `git push` → on PSC `git pull`; verify apricot + data present.
    2. Re-run the edit on PSC (or rsync the edited model up).
    3. `bash clts/edit_clt/submit_edit_clt.sh --test` → on success, without `--test`.
    4. Sync `clt_runs` back to the Mac, then open `02_circuit_compare.ipynb`.
```

- [ ] **Step 2: Validate the notebook (jupyter/nbconvert is NOT installed locally)**

Confirm the notebook is well-formed JSON and every code cell compiles:

Run: `cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4" && "/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python" - <<'PY'
import json, ast
nb = json.load(open("clts/edit_clt/01_edit_and_train.ipynb"))
n = 0
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"])); n += 1
print("all", n, "code cells compile")
PY`
Expected: prints "all N code cells compile" with no SyntaxError.

Full top-to-bottom execution is a user runbook step (needs jupyter + the PSC-trained CLTs); the cell logic is already covered by the unit/integration tests on `edit_clt_config`, `prepare_edited_model`, and `trainCLT`'s add-on.

- [ ] **Step 3: Commit**

```bash
git add clts/edit_clt/01_edit_and_train.ipynb
git commit -m "feat(edit-clt): Notebook 1 — edit + train (M1/M2) + manifest"
```

---

### Task 9: Notebook 2 — `02_circuit_compare.ipynb`

**Files:**
- Create: `clts/edit_clt/02_circuit_compare.ipynb`

**Interfaces:** consumes Task 3 (manifest), Task 6 (compare), Task 5 (drift). Runs in the **circuit-tracer env** (`clts/.venv-ct/bin/python -m jupyter …` locally; `lm4-ct` on PSC).

- [ ] **Step 1: Create the notebook with these cells**

```
md: # 02 — Circuit comparison across the 4 configs
    baseline_orig (apricot+orig) · m3_stale (apricot+edited) ·
    m1_scratch (scratch+edited) · m2-* (fine-tuned+edited).
    RUN IN THE CIRCUIT-TRACER ENV (clts/.venv-ct).

py: # --- setup + load manifest ---
    import sys, json
    from pathlib import Path
    REPO = Path.cwd()
    while REPO.name != "Interp_LM4" and REPO != REPO.parent: REPO = REPO.parent
    sys.path.insert(0, str(REPO))
    from clts.storage import storage_root
    from clts.edit_clt import edit_clt_config as cfg, compare, drift
    STORE = storage_root()
    EXP_ID = "grid-L4-H6-edit-p0-month-jul"
    M = cfg.read_manifest(STORE / "edit_experiments" / EXP_ID / "manifest.json")
    PROMPT = M["trace_prompt"]; DATA_DIR = M["data_dir"]
    GRAPH_ROOT = STORE / "clt_graphs"
    print("prompt:", PROMPT)
    print("apricot target ce_recovered:", round(M["target_stats"]["ce_recovered"], 4))

py: # --- build (or reuse) the 4 graphs on the same prompt ---
    gcs = compare.configs_from_manifest(M)
    results = {}
    for gc in gcs:
        if not Path(gc.clt_dir).exists():
            print("SKIP (missing CLT):", gc.key, gc.clt_dir); continue
        print("building:", gc.key)
        results[gc.key] = compare.build_or_load_graph(
            gc, DATA_DIR, GRAPH_ROOT, slug=EXP_ID, prompt=PROMPT, device="cpu")
    reports = {k: r["report"] for k, r in results.items()}
    graphs = {k: r["graph"] for k, r in results.items()}

py: # --- replacement-model stats across configs (the headline comparison) ---
    compare.comparison_table(reports)

py: # --- also trace the SAME target tokens (old vs new month) across configs ---
    for tgt in (" February", " July"):
        print("target:", tgt)
        for gc in gcs:
            if gc.key not in graphs: continue
            r = compare.build_or_load_graph(gc, DATA_DIR, GRAPH_ROOT,
                    slug=f"{EXP_ID}{tgt.strip()}", prompt=PROMPT, target=tgt, device="cpu")
            print(f"  {gc.key:14s} repl={r['report']['replacement_score']:.3f} "
                  f"err={r['report']['error_influence_share']:.3f}")

py: # --- feature diff vs baseline: what appears / disappears ---
    compare.feature_diff_table(graphs, baseline_key="baseline_orig")

py: # --- weight-space drift (M2/M3 share apricot indexing; M1 needs matching) ---
    from clts.clt import CrossLayerTranscoder
    base_clt = CrossLayerTranscoder.load_from_dir(M["base_clt_dir"])
    for key, m in M["methods"].items():
        d = Path(m["expected_clt_dir"])
        if not d.exists(): continue
        other = CrossLayerTranscoder.load_from_dir(d)
        if key.startswith("m2"):
            dr = drift.decoder_cosine_drift(base_clt, other)
            print(f"{key}: mean_cos={dr['mean_cosine']:.4f} frac_moved={dr['frac_moved']:.3f}")
        else:  # m1 from scratch -> per-layer best-match cosine
            cos0 = drift.match_features(base_clt, other, layer=0)["match_cosine"].mean()
            print(f"{key}: L0 mean best-match cos to apricot = {float(cos0):.3f}")

py: # --- Method-3 focus: did unexplained influence rise vs baseline? ---
    if "baseline_orig" in reports and "m3_stale" in reports:
        b, s = reports["baseline_orig"], reports["m3_stale"]
        print(f"error_influence_share: baseline={b['error_influence_share']:.3f} "
              f"-> m3_stale={s['error_influence_share']:.3f} "
              f"(Δ={s['error_influence_share']-b['error_influence_share']:+.3f})")
        ov = drift.active_feature_overlap(graphs["baseline_orig"], graphs["m3_stale"])
        print("features lost when stale CLT meets the edit:", ov["disappeared_set"])

py: # --- save a comparison report (table + diffs) ---
    out = STORE / "edit_experiments" / EXP_ID / "comparison.json"
    out.write_text(json.dumps({"reports": reports,
        "feature_diff": compare.feature_diff_table(graphs).to_dict(orient="index")},
        indent=2, default=str))
    print("wrote", out)

md: ## Optional: interactive viewer
    `clts/.venv-ct/bin/python clts/serve_ui.py --graph-dir <clt_graphs/<key>/<EXP_ID>> --scan-name <key> --port 8050`
```

- [ ] **Step 2: Validate the notebook (jupyter not installed; full run needs PSC CLTs)**

Confirm well-formed JSON + every code cell compiles:

Run: `cd "/Users/efmac/Code/Project Code/CRL-Interp/Interp_LM4" && "/Users/efmac/Code/Project Code/CRL-Interp/.venv/bin/python" - <<'PY'
import json, ast
nb = json.load(open("clts/edit_clt/02_circuit_compare.ipynb"))
n = 0
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"])); n += 1
print("all", n, "code cells compile")
PY`
Expected: prints "all N code cells compile" with no SyntaxError.

Full execution runs in the circuit-tracer env (`clts/.venv-ct`) after the M1/M2 CLTs are trained on PSC and synced back; only `baseline_orig` + `m3_stale` (apricot) are buildable locally. This is a user runbook step.

- [ ] **Step 3: Commit**

```bash
git add clts/edit_clt/02_circuit_compare.ipynb
git commit -m "feat(edit-clt): Notebook 2 — 4-way attribution-graph comparison"
```

---

### Task 10: README + fix stale HPC doc

**Files:**
- Create: `clts/edit_clt/README.md`
- Modify: `docs/environments-and-hpc.md`

- [ ] **Step 1: Write the runbook**

Create `clts/edit_clt/README.md`:

```markdown
# edit_clt — fact-edit → CLT-adaptation → circuit comparison

Spec: `docs/superpowers/specs/2026-06-27-edit-clt-circuit-compare-design.md`.

**Methods (adaptation spectrum):** M1 = CLT from scratch on the edited model;
M2 = fine-tune apricot from checkpoint (gentle, plateau-or-parity stop); M3 =
stale apricot on the edited model (analysis only, Notebook 2).

## Runbook
1. **Notebook 1** (training env) — run the edit, save the edited model, compute
   apricot target stats, CPU smoke-test, write the manifest.
2. **Push code:** `git push`; on PSC `git pull`. Verify on PSC:
   `ls $CLT_STORAGE_ROOT/clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final`
   and the data dir exist.
3. **Edited model on PSC:** rerun Notebook 1's edit cells on PSC, or rsync the
   30 MB `model/grid-L4-H6-edit-p0-month-jul/` up. Point `EDITED_MODEL_DIR` at it.
4. **Test job:** `bash clts/edit_clt/submit_edit_clt.sh --test` (uses `CONDA_ENV=lm4-ct`).
5. **Full runs:** on success, `bash clts/edit_clt/submit_edit_clt.sh`.
6. **Sync back** `clt_runs/grid-L4-H6-edit-p0-month-jul/` to the Mac.
7. **Notebook 2** (circuit-tracer env, `clts/.venv-ct`) — build + compare graphs.

## Gotchas
- PSC: always `CONDA_ENV=lm4-ct` (`lm4` was deleted).
- Notebook 2 + anything importing `circuit_tracer` runs only in `clts/.venv-ct`.
- fp32 / RMSNorm eps=1e-5 / condensed vocab — don't "fix" these.
```

- [ ] **Step 2: Fix the stale env note**

In `docs/environments-and-hpc.md`, in the "B. Training env — conda `lm4`" heading area, add a one-line note. Change the heading line `### B. Training env — conda \`lm4\` (Python 3.11)` to:

```markdown
### B. Training env — conda `lm4-ct` (Python 3.11)  *(was `lm4`; `lm4` was deleted — `lm4-ct` now carries the training stack too)*
```

- [ ] **Step 3: Verify markdown renders + commit**

Run: `ls clts/edit_clt/README.md && grep -n "lm4-ct" docs/environments-and-hpc.md | head`
Expected: file exists; grep shows the new note.

```bash
git add clts/edit_clt/README.md docs/environments-and-hpc.md
git commit -m "docs(edit-clt): runbook + fix stale lm4->lm4-ct training-env note"
```

---

## Self-Review

**1. Spec coverage:**
- Adaptation-spectrum framing → Notebook 2 cells (drift + comparison). ✓
- apricot baseline as reference + M2 start → `default_config`, manifest, submit driver. ✓
- M1 from scratch, apricot config → Task 7 `submit "clt-edit-m1"`, `default_config` m1_scratch. ✓
- M2 fine-tune + variants + convergence (plateau/parity) → Task 1 `_should_stop`, Task 7 variants. ✓
- M3 analysis (stale CLT, error rises) → Notebook 2 Method-3 cell. ✓
- Edited model saved as model dir → Task 4. ✓
- Manifest as glue → Task 3 + written in NB1 + read in NB2. ✓
- trainCLT pure add-on + verified insertion points → Task 1 (matches L182-184/L207-211/L224/L243-246/L259-274). ✓
- PSC: extend train_clt_psc.sh, lm4-ct, --test, apricot/code on PSC → Tasks 2, 7, 10. ✓
- Drift measured not enforced → drift.py used only for reporting; no stop-guard. ✓
- Conventions (fp32/eps, vocab, repo-root, venvs) → Global Constraints + per-task. ✓
- Testing strategy → unit tests Tasks 1,3,5,6; integration Task 4; smoke NB1/NB2. ✓
- Out-of-scope (anchor on by default, multi-fact) → anchor defaults off; single fact. ✓

**2. Placeholder scan:** No "TBD/TODO/handle edge cases/similar to Task N". Every code step has full code. `<...>` only appears in human-facing runbook print strings (intended). ✓

**3. Type consistency:** `expected_clt_dir` signature identical in Task 3 def, test, NB1, manifest. `_trial_name`/`trial_name` format string identical (`mult{e}_l0{l0:g}_lr{lr:g}_ep{ep}_n{n}`). `make_edited_model(fact, edited_model_dir, *, factediting_root, …)` identical in def + test + NB1. `GraphConfig(key, model_dir, clt_dir, scan_name)` + `configs_from_manifest` keys (`baseline_orig`/`m3_stale`/method keys) consistent across compare.py, tests, NB2. `decoder_cosine_drift`/`active_feature_overlap`/`match_features` signatures consistent drift.py ↔ tests ↔ NB2. Manifest keys (`orig_model_dir`, `edited_model_dir`, `base_clt_dir`, `methods[*].expected_clt_dir`, `trace_prompt`, `target_stats`) written in NB1 and read in `configs_from_manifest` + NB2. ✓

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-27-edit-clt-circuit-compare.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
