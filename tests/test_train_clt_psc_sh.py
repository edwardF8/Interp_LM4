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
                  "EVAL_PERSON",
                  "--resume-from", "--out-tag", "--target-ce-recovered",
                  "--plateau-patience", "--plateau-min-delta", "--eval-every",
                  "--anchor-lambda", "--eval-person"):
        assert token in text, f"missing {token}"
