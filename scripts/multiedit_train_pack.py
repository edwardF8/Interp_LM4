"""Concurrent CLT-training packer for the multi-edit Tier-2 sweep (GPU-hour efficient).

grid-L4-H6 is tiny, so one GPU is badly underutilized by a single CLT train. This runs
this shard's slice of a jobs file (`jobs[shard::num_shards]`) with up to `--concurrency`
concurrent `clts/trainCLT.py` subprocesses on the one booked GPU, so GPU-hours ≈
sum(train time) / concurrency instead of a full GPU per train. Run from the Interp_LM4
repo root in the training env (lm4-ct).

  python scripts/multiedit_train_pack.py --jobs train_jobs.jsonl --shard 0 --num-shards 6 --concurrency 4

Each line of the jobs file is one env-dict from FactEditingLM4's multi-edit driver
(MODEL_NAME/MODEL_DIR/OUT_TAG/EXPANSION/L0/LR/EPOCHS/N_EXAMPLES + edit-CLT add-ons incl.
EVAL_PERSON + TARGET_CE_RECOVERED). Mirrors scripts/train_clt_psc.sh's env->flag mapping.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ENC = "blocks.{layer}.hook_resid_mid"
_DEC = "blocks.{layer}.hook_mlp_out"
_ADDON = [("RESUME_FROM", "--resume-from"), ("OUT_TAG", "--out-tag"),
          ("TARGET_CE_RECOVERED", "--target-ce-recovered"),
          ("PLATEAU_PATIENCE", "--plateau-patience"),
          ("PLATEAU_MIN_DELTA", "--plateau-min-delta"),
          ("EVAL_EVERY", "--eval-every"), ("ANCHOR_LAMBDA", "--anchor-lambda"),
          ("EVAL_PERSON", "--eval-person")]


def build_cmd(py, job, data_dir):
    cmd = [py, "-u", "clts/trainCLT.py",
           "--model-dir", job["MODEL_DIR"], "--data-dir", data_dir,
           "--model-name", job["MODEL_NAME"],
           "--enc-hook-template", _ENC, "--dec-hook-template", _DEC,
           "--expansion", job["EXPANSION"], "--l0", job["L0"], "--lr", job["LR"],
           "--epochs", job["EPOCHS"], "--context-size", job.get("CONTEXT_SIZE", "512"),
           "--n-examples", job["N_EXAMPLES"]]
    for key, flag in _ADDON:
        if job.get(key):
            cmd += [flag, job[key]]
    return cmd


def main():
    ap = argparse.ArgumentParser(description="concurrent CLT-training packer")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--storage-root",
                    default=os.environ.get("CLT_STORAGE_ROOT",
                                           "/jet/home/friedmae/data_storage/LM4_Results"))
    ap.add_argument("--data-dir",
                    default=os.environ.get("DATA_DIR",
                                           "/jet/home/friedmae/data_storage/LM4_Results/Data/bioS_N-Bd_final_grid"))
    a = ap.parse_args()

    jobs = [json.loads(ln) for ln in open(a.jobs) if ln.strip()]
    mine = jobs[a.shard::a.num_shards]
    Path("logs").mkdir(exist_ok=True)
    print(f"[pack] shard {a.shard}/{a.num_shards}: {len(mine)} jobs, "
          f"concurrency={a.concurrency}, storage={a.storage_root}", flush=True)

    running, results, i = [], [], 0
    while i < len(mine) or running:
        while i < len(mine) and len(running) < a.concurrency:
            job = mine[i]; i += 1
            env = dict(os.environ)
            env["CLT_STORAGE_ROOT"] = a.storage_root
            env["WANDB_RUN_GROUP"] = "multiedit"
            env["WANDB_NAME"] = f"{job['MODEL_NAME']}/{job['OUT_TAG']}"
            logf = open(f"logs/pack-{job['MODEL_NAME']}-{job['OUT_TAG']}.log", "w")
            p = subprocess.Popen(build_cmd(a.py, job, a.data_dir),
                                 stdout=logf, stderr=subprocess.STDOUT, env=env)
            running.append((p, job, logf))
            print(f"[pack] launch {job['MODEL_NAME']}/{job['OUT_TAG']} ({len(running)} running)", flush=True)
        time.sleep(10)
        still = []
        for p, job, logf in running:
            if p.poll() is None:
                still.append((p, job, logf))
            else:
                logf.close()
                results.append((job, p.returncode))
                print(f"[pack] done  {job['MODEL_NAME']}/{job['OUT_TAG']} rc={p.returncode} "
                      f"({len(results)}/{len(mine)})", flush=True)
        running = still

    nfail = sum(1 for _, rc in results if rc != 0)
    print(f"[pack] shard {a.shard} COMPLETE: {len(results)} jobs, {nfail} failed", flush=True)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
