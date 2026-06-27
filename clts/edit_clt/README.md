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
