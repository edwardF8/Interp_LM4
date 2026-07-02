# CLT Self-Data Fine-Tune Control — Design

**Date:** 2026-07-01
**Files touched (all new — purely additive, no shared file modified):**
- `finetuning/submit_finetune_clt.sh` (new — PSC submit driver)
- `finetuning/README.md` (new — usage, variants, caveats; also carries the doc
  content that would otherwise have gone to `CLAUDE.md` / `docs/workflows.md`,
  which were deliberately left untouched)
- `tests/test_finetune_submit_sh.py` (new — bash -n + tokens + `--dry-run` behavior)

## Problem

The edit-CLT experiment's **Method 2** fine-tuned the apricot CLT
(`clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final`) on the
*edited* model (`FactEditingLM4/edit_clt/submit_edit_clt.sh`, variants v2-basic /
v2-fixed). Interpreting the drift of those fine-tunes is confounded: features move
both because the model changed *and* because continued training moves a converged
CLT anyway (fresh warmup ramps, optimizer noise, finite-sample gradient drift).

We need the null: the **same CLT, same recipe, same data, on the *original*
model**. Whatever drifts here is the continued-training baseline; Method-2 drift
is read net of it (analysis in `FactEditingLM4/edit_clt/drift.py`).

## Design

One new bash driver, `finetuning/submit_finetune_clt.sh`, modeled on
`submit_edit_clt.sh`: it composes env vars and sbatch-es the existing
`scripts/train_clt_psc.sh`, which already threads the fine-tune add-on flags into
`clts/trainCLT.py` (`--resume-from`, `--out-tag`, `--plateau-patience`,
`--plateau-min-delta`, `--eval-every`, `--anchor-lambda`). **No training code is
touched** — the entire change is a parameterization.

Key decisions:

- **Two variants matching Method-2 one-for-one.** basic (default) = LR 2e-5,
  epochs 5, plateau patience 3 / min-delta 0.01 / eval every 100 (procedure-matched
  to v2-basic); `--fixed` = LR 2e-5, epochs 2, no plateau (step-matched to
  v2-fixed). Common: `SWEEP=0 EXPANSION=16 L0=2 N_EXAMPLES=10000 CONTEXT_SIZE=512`,
  wall 06:00:00. The point is recipe *identity*, warmup wobble included — trainCLT
  re-runs its L0 warmup (`total_steps//10`) and LR warmup (`total_steps//50`) on
  resume, and Method-2 had the same transient, so we keep it.
- **Original model + data, apricot as RESUME_FROM.** `MODEL_NAME=grid-L4-H6`,
  `MODEL_DIR=$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid/grid-L4-H6/final`,
  `DATA_DIR=$REMOTE_BASE/Data/bioS_N-Bd_final_grid`,
  `BASE_CLT_DIR=<apricot final/>`. `DiverseBioSubset(seed=CLT_SEED=0)` on the same
  data dir reproduces the original training distribution.
- **Distinct output namespace.** `OUT_TAG=apricot-finetune-{basic,fixed}` — via
  trainCLT's `_run_dir` out_tag override the result lands at
  `clt_runs/grid-L4-H6/<out_tag>/mult16_l02_lr2e-05_ep<E>_n<N>/final`, never
  overwriting apricot (`standalone/`) or colliding across variants.
- **`CONDA_ENV=lm4-ct` always.** The rebuilt PSC `lm4` env's cu130 torch reports
  `cuda=False`; `lm4-ct` is the known-good env (memory: `psc-lm4-conda-env-deleted`).
- **wandb identifiability without code changes.** `WANDB_NAME=grid-L4-H6/<out_tag>`
  + `WANDB_RUN_GROUP=clt-finetune-control` in the submitted env. Works because
  trainCLT.py passes no name and `group=None` for non-robust standalone runs, so
  the env vars apply — the same mechanism `train_clt_psc.sh`'s SLURM-array mode
  uses for `WANDB_RUN_GROUP`. Project stays `interpLM4`; the run still optimizes /
  reports `final_eval/ce_recovered`.
- **Dashboards on by default** (`WRITE_DASHBOARDS=1`,
  `DASH_SCAN=grid-L4-H6-<out_tag>`, `DASH_N_PEOPLE=1000`, cuda), using the
  existing opt-in post-training block in `train_clt_psc.sh`; `WRITE_DASHBOARDS=0`
  opts out.
- **`ANCHOR_LAMBDA` pass-through only when set** (`${ANCHOR_LAMBDA:+…}`): default
  unset = off, matching Method-2; setting e.g. `1e-4` adds an L2 pull toward the
  apricot weights for a lower-drift control variant.
- **`--test`** (composable with `--fixed`): `N_EXAMPLES=1000 EPOCHS=1
  DASH_N_PEOPLE=100`, 1h wall, and `-test` suffixed onto OUT_TAG / DASH_SCAN /
  job name so a smoke run can never clobber the real artifact.
- **`--dry-run`** (composable): prints the fully-resolved `env … sbatch …` line
  (`printf '%q '`) plus the resolved output path, executes nothing — testable on a
  Mac with no sbatch. This is what makes the driver unit-testable at all.

Expected behavior note: the basic full run is 6250 steps (5 × 10000×512 / 4096)
with evals every 100; apricot starts converged on its own data, so the plateau
stop should fire near step ~400. Early stop is the expected outcome, not a bug.

## Alternatives considered

- **New flags / code in `trainCLT.py` or a new trainer** — rejected; everything
  needed already exists behind env vars, and the repo convention is add-on
  drivers over shared-file edits.
- **Reusing OUT_TAG `method2-v2-*` names or `standalone/`** — rejected; the
  control must be unambiguously distinguishable from the Method-2 artifacts and
  must never risk overwriting apricot.
- **Anchored-by-default (`ANCHOR_LAMBDA` set)** — rejected; Method-2 ran
  unanchored, and the control must match the recipe. Anchoring stays an opt-in
  extra arm.
- **Editing `CLAUDE.md` / `docs/workflows.md`** — originally planned, dropped by
  amendment (purely-additive requirement); the content lives in
  `finetuning/README.md` instead.

## Verification

- TDD: `tests/test_finetune_submit_sh.py` written first (9 failures on the
  missing script), then the driver; all pass. Tests cover bash syntax, required
  tokens (apricot path, `RESUME_FROM`, `CONDA_ENV=lm4-ct`, both OUT_TAGs, LR,
  plateau vars, `WANDB_NAME`, dashboards vars), and `--dry-run` behavior for
  default / `--fixed` / `--test` / `--test --fixed` / `ANCHOR_LAMBDA=0.5` /
  `WRITE_DASHBOARDS=0` / env overrides / unknown-flag failure. Dry-run needs no
  sbatch, so these stay in the fast (non-`integration`) suite.
- `bash -n finetuning/submit_finetune_clt.sh`; fast suite: the 10 new tests all
  pass and are the entire delta vs. pre-change (this machine has pre-existing,
  unrelated failures: missing local `sae_runs` artifacts + a vendored
  `reference/` collection error — identical with and without this change).
- No training executed locally; the four `--dry-run` command lines eyeballed
  against `submit_edit_clt.sh`'s Method-2 lines.
