"""Tests for finetuning/submit_finetune_clt.sh — the self-data fine-tune
control driver (continue-train the apricot CLT on its OWN original model+data).

Mirrors tests/test_train_clt_psc_sh.py (bash -n + token assertions) and adds
behavioral --dry-run runs. --dry-run executes nothing, so these are safe on any
machine (no sbatch required) and stay out of the `integration` marker.
"""
import os
import subprocess
from pathlib import Path

SH = Path(__file__).resolve().parent.parent / "finetuning" / "submit_finetune_clt.sh"

APRICOT = "clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final"

# Every env var the driver reads: stripped from the subprocess env so ambient
# shell state can't skew a behavioral run; tests re-add what they exercise.
DRIVER_ENV_VARS = (
    "REMOTE_BASE", "MODEL_NAME", "MODEL_DIR", "DATA_DIR", "BASE_CLT_DIR",
    "WRITE_DASHBOARDS", "FEATURES_ROOT", "DASH_SCAN", "DASH_N_PEOPLE",
    "DASH_DEVICE", "ANCHOR_LAMBDA",
)


def run_driver(*flags, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in DRIVER_ENV_VARS}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(["bash", str(SH), *flags],
                          capture_output=True, text=True, env=env)


def dry_run(*flags, env_extra=None):
    proc = run_driver("--dry-run", *flags, env_extra=env_extra)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    return proc.stdout


# ---- static checks ----------------------------------------------------------

def test_script_has_valid_bash_syntax():
    # bash -n parses without executing.
    subprocess.run(["bash", "-n", str(SH)], check=True)


def test_script_contains_required_tokens():
    text = SH.read_text()
    for token in (
        APRICOT,                        # BASE_CLT_DIR default = the apricot CLT
        "RESUME_FROM",                  # fine-tune, not from-scratch
        "CONDA_ENV=lm4-ct",             # never the broken cu130 `lm4` env
        "apricot-finetune-basic",
        "apricot-finetune-fixed",
        "LR=2e-5",
        "PLATEAU_PATIENCE=3",
        "PLATEAU_MIN_DELTA=0.01",
        "EVAL_EVERY=100",
        "WANDB_NAME",
        "WANDB_RUN_GROUP=clt-finetune-control",
        "WRITE_DASHBOARDS",
        "DASH_SCAN",
        "ANCHOR_LAMBDA",
        "--dry-run",
        "--test",
        "--fixed",
    ):
        assert token in text, f"missing {token}"


# ---- behavioral: --dry-run prints the sbatch command, executes nothing ------

def test_dry_run_default_is_basic_variant():
    out = dry_run()
    for token in (
        "sbatch",
        "scripts/train_clt_psc.sh",
        "MODEL_NAME=grid-L4-H6",
        f"RESUME_FROM=/jet/home/friedmae/data_storage/LM4_Results/{APRICOT}",
        "OUT_TAG=apricot-finetune-basic",
        "CONDA_ENV=lm4-ct",
        "SWEEP=0", "EXPANSION=16", "L0=2", "CONTEXT_SIZE=512",
        "LR=2e-5", "EPOCHS=5", "N_EXAMPLES=10000",
        "PLATEAU_PATIENCE=3", "PLATEAU_MIN_DELTA=0.01", "EVAL_EVERY=100",
        "WANDB_NAME=grid-L4-H6/apricot-finetune-basic",
        "WANDB_RUN_GROUP=clt-finetune-control",
        "WRITE_DASHBOARDS=1",
        "DASH_SCAN=grid-L4-H6-apricot-finetune-basic",
        "DASH_N_PEOPLE=1000", "DASH_DEVICE=cuda",
        "--job-name=clt-apricot-ft-basic",
        "--time=06:00:00",
        "--account=cis240072p", "--partition=GPU-shared",
        # resolved artifact path for eyeballing (lr 2e-5 renders as lr2e-05)
        "clt_runs/grid-L4-H6/apricot-finetune-basic/mult16_l02_lr2e-05_ep5_n10000/final",
    ):
        assert token in out, f"missing {token}\n--- stdout ---\n{out}"
    assert "ANCHOR_LAMBDA" not in out          # off unless exported by the caller
    assert "-test" not in out                  # no test suffix by default
    assert "Submitted batch job" not in out    # nothing actually ran


def test_dry_run_fixed_variant_has_no_plateau():
    out = dry_run("--fixed")
    for token in (
        "OUT_TAG=apricot-finetune-fixed",
        "EPOCHS=2",
        "EVAL_EVERY=100",
        "WANDB_NAME=grid-L4-H6/apricot-finetune-fixed",
        "--job-name=clt-apricot-ft-fixed",
        "clt_runs/grid-L4-H6/apricot-finetune-fixed/mult16_l02_lr2e-05_ep2_n10000/final",
    ):
        assert token in out, f"missing {token}\n--- stdout ---\n{out}"
    assert "PLATEAU_PATIENCE" not in out
    assert "PLATEAU_MIN_DELTA" not in out


def test_dry_run_test_mode_suffixes_and_cheap_knobs():
    out = dry_run("--test")
    for token in (
        "N_EXAMPLES=1000", "EPOCHS=1", "DASH_N_PEOPLE=100", "--time=01:00:00",
        "OUT_TAG=apricot-finetune-basic-test",
        "DASH_SCAN=grid-L4-H6-apricot-finetune-basic-test",
        "--job-name=clt-apricot-ft-basic-test",
    ):
        assert token in out, f"missing {token}\n--- stdout ---\n{out}"
    # discriminating negatives (the cheap values are substrings of the full ones)
    assert "N_EXAMPLES=10000" not in out
    assert "DASH_N_PEOPLE=1000" not in out
    assert "--time=06:00:00" not in out


def test_dry_run_test_fixed_compose():
    out = dry_run("--test", "--fixed")
    for token in (
        "OUT_TAG=apricot-finetune-fixed-test",
        "EPOCHS=1", "N_EXAMPLES=1000",
        "--job-name=clt-apricot-ft-fixed-test",
        "clt_runs/grid-L4-H6/apricot-finetune-fixed-test/mult16_l02_lr2e-05_ep1_n1000/final",
    ):
        assert token in out, f"missing {token}\n--- stdout ---\n{out}"
    assert "PLATEAU_PATIENCE" not in out


def test_dry_run_anchor_lambda_passthrough_only_when_set():
    out = dry_run(env_extra={"ANCHOR_LAMBDA": "0.5"})
    assert "ANCHOR_LAMBDA=0.5" in out


def test_dry_run_write_dashboards_opt_out():
    out = dry_run(env_extra={"WRITE_DASHBOARDS": "0"})
    assert "WRITE_DASHBOARDS=0" in out
    assert "WRITE_DASHBOARDS=1" not in out


def test_dry_run_env_overrides_flow_through():
    out = dry_run(env_extra={"BASE_CLT_DIR": "/tmp/other-clt/final",
                             "MODEL_NAME": "grid-L8-H6"})
    assert "RESUME_FROM=/tmp/other-clt/final" in out
    assert "MODEL_NAME=grid-L8-H6" in out


def test_unknown_flag_fails():
    proc = run_driver("--bogus")
    assert proc.returncode != 0
