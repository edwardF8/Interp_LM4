# finetuning/ — self-data fine-tune control for the edit-CLT experiment

**What this is.** A PSC submit driver that continue-trains ("fine-tunes") the existing
**apricot** CLT — `clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final` —
on its **own original** model (`grid-L4-H6`) + **own original** data (`bioS_N-Bd_final_grid`),
saving the result as a new labeled CLT. This is the **continued-training drift control** for
the edit-CLT experiment: the Method-2 fine-tunes (`FactEditingLM4/edit_clt/submit_edit_clt.sh`)
ran this same recipe against the *edited* model, so any drift they show mixes "the edit moved
the features" with "continued training moves features anyway." This control isolates the
second term — same CLT, same recipe, same data, *nothing* changed about the model.

No shared file is touched: the driver only composes env vars and sbatch-es the existing
`scripts/train_clt_psc.sh`, which threads them into `clts/trainCLT.py`'s opt-in add-on flags
(`--resume-from`, `--out-tag`, `--plateau-patience`, `--plateau-min-delta`, `--eval-every`,
`--anchor-lambda`).

**Data identity.** trainCLT.py builds its corpus with `DiverseBioSubset(seed=CLT_SEED=0)` on
the same data dir, so this run reproduces the original apricot training distribution — the
fine-tune sees the same text the base CLT saw.

## The two variants

| variant | flag | LR | epochs | early stop | OUT_TAG | job name |
|---|---|---|---|---|---|---|
| **basic** (default) | — | 2e-5 | 5 | plateau: patience 3, min-delta 0.01, eval every 100 steps | `apricot-finetune-basic` | `clt-apricot-ft-basic` |
| **fixed** | `--fixed` | 2e-5 | 2 | none (eval every 100 steps) | `apricot-finetune-fixed` | `clt-apricot-ft-fixed` |

basic is **procedure-matched** to Method-2 v2-basic; fixed is **step-matched** to Method-2
v2-fixed. Common to both: `SWEEP=0 EXPANSION=16 L0=2 N_EXAMPLES=10000 CONTEXT_SIZE=512`,
wall `06:00:00`, and `CONDA_ENV=lm4-ct` **always** (the rebuilt `lm4` env's cu130 torch
reports `cuda=False` — never use it).

## How to run

On a Bridges-2 login node (the script cd's to the repo root itself, run it from anywhere):

```bash
bash finetuning/submit_finetune_clt.sh              # basic (full run + dashboards)
bash finetuning/submit_finetune_clt.sh --fixed      # fixed variant
bash finetuning/submit_finetune_clt.sh --test       # GPU smoke: N=1000, ep=1, 1h wall,
                                                    #   "-test" suffixed onto OUT_TAG /
                                                    #   DASH_SCAN / job name (never
                                                    #   overwrites the real artifact)
bash finetuning/submit_finetune_clt.sh --dry-run    # print the fully-resolved
                                                    #   `env … sbatch …` line; run nothing
                                                    #   (works on a Mac, no sbatch needed)
ANCHOR_LAMBDA=1e-4 bash finetuning/submit_finetune_clt.sh   # opt-in L2 pull toward the
                                                    #   apricot weights for even less drift;
                                                    #   default UNSET = off, matching Method-2
WRITE_DASHBOARDS=0 bash finetuning/submit_finetune_clt.sh   # train only, skip dashboards
```

Flags compose: `--dry-run --test --fixed` prints the fixed-variant smoke command.

Env-overridable knobs (defaults in the script): `REMOTE_BASE`, `MODEL_NAME`, `MODEL_DIR`,
`DATA_DIR`, `BASE_CLT_DIR` (what gets resumed), `WRITE_DASHBOARDS`, `FEATURES_ROOT`,
`DASH_SCAN`, `DASH_N_PEOPLE`, `DASH_DEVICE`, `ANCHOR_LAMBDA`.

## Where outputs land

- **CLT:** `$REMOTE_BASE/clt_runs/grid-L4-H6/<out_tag>/mult16_l02_lr2e-05_ep<E>_n<N>/final`
  (trial dir from trainCLT.py's `trial_name()` = `mult{expansion}_l0{l0:g}_lr{lr:g}_ep{epochs}_n{n}`;
  lr 2e-5 renders as `lr2e-05`). E.g. basic full run:
  `clt_runs/grid-L4-H6/apricot-finetune-basic/mult16_l02_lr2e-05_ep5_n10000/final`.
- **Feature dashboards** (on by default, same job, training env):
  `$REMOTE_BASE/clt_features/grid-L4-H6-<out_tag>/` with `DASH_N_PEOPLE=1000` (test: 100).
- **wandb:** project `interpLM4`, group **`clt-finetune-control`**, run name
  `grid-L4-H6/<out_tag>`. This works because trainCLT.py passes no name and `group=None`
  for non-robust standalone runs, so the `WANDB_NAME` / `WANDB_RUN_GROUP` env vars take
  effect — the same mechanism `train_clt_psc.sh`'s SLURM-array mode uses.

## Caveats

- **Warmup ramps re-run.** trainCLT.py always applies its L0-coefficient warmup (first
  `total_steps // 10` steps) and LR warmup (`total_steps // 50`), even on resume — so the
  fine-tune starts with a transient **densify-then-resparsify wobble**. Kept deliberately:
  the Method-2 edit fine-tunes had the same wobble, and the control must match the recipe
  exactly, wobble included.
- **Plateau stop will likely trigger early — that is expected.** The basic full run is 6250
  steps (5 epochs × 10000×512 tokens / batch 4096) with an eval every 100; apricot starts
  *converged* on this data, so `ce_recovered` should plateau almost immediately (stop around
  step ~400, the earliest the patience-3 window allows). An early stop is not a failure —
  a converged start is the point of the control.
- **Drift analysis** tooling (weight/feature drift vs. the apricot baseline) lives in
  `FactEditingLM4/edit_clt/drift.py`, not in this repo.

## Reusing this to fine-tune any CLT from a checkpoint

This driver is the precedent for CLT fine-tuning in general: point `BASE_CLT_DIR` at another
CLT's `final/` dir (plus `MODEL_DIR`/`MODEL_NAME`/`DATA_DIR` overrides as needed), or crib its
env→sbatch pattern. The underlying machinery is `clts/trainCLT.py`'s `--resume-from`/`--out-tag`
et al., threaded by `scripts/train_clt_psc.sh` (env vars `RESUME_FROM`, `OUT_TAG`,
`PLATEAU_PATIENCE`, `PLATEAU_MIN_DELTA`, `EVAL_EVERY`, `ANCHOR_LAMBDA`, `TARGET_CE_RECOVERED`).

## Tests

`tests/test_finetune_submit_sh.py` — `bash -n`, token assertions, and behavioral `--dry-run`
runs (dry-run executes nothing and needs no sbatch, so these run in the fast suite anywhere):

```bash
python -m pytest tests/test_finetune_submit_sh.py -q
```
