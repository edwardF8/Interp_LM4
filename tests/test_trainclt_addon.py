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
    assert args.eval_person is None


def test_eval_person_flag_parses():
    args = trainCLT.parse_args(
        ["--model-dir", "x", "--data-dir", "y", "--eval-person", "42"]
    )
    assert args.eval_person == 42
