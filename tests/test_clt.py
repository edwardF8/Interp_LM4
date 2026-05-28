"""Unit tests for CrossLayerTranscoder."""
import torch

from clts.clt import CrossLayerTranscoder


def test_shapes():
    clt = CrossLayerTranscoder(n_layers=4, d_model=8, expansion=2)
    assert clt.W_enc.shape == (4, 16, 8)
    assert clt.b_enc.shape == (4, 16)
    assert clt.threshold.shape == (4, 16)
    assert clt.b_dec.shape == (4, 8)
    assert len(clt.W_dec) == 4
    for i in range(4):
        assert clt.W_dec[i].shape == (16, 4 - i, 8), \
            f"W_dec[{i}] is {clt.W_dec[i].shape}, expected ({16}, {4 - i}, {8})"


def test_forward_pass_dimensions():
    torch.manual_seed(0)
    clt = CrossLayerTranscoder(n_layers=4, d_model=8, expansion=2)
    x_list = [torch.randn(2, 8) for _ in range(4)]
    y_hat_list = clt(x_list)
    assert len(y_hat_list) == 4
    for L in range(4):
        assert y_hat_list[L].shape == (2, 8)
        assert torch.isfinite(y_hat_list[L]).all()


def test_cross_layer_writes():
    """Feature at source 0 must write only to target k when its decoder
    column for k is the only nonzero one. Catches decoder routing off-by-one.
    """
    torch.manual_seed(1)
    clt = CrossLayerTranscoder(n_layers=4, d_model=8, expansion=2)

    # Zero everything, then set up a single feature path: source L=0, target k=2.
    with torch.no_grad():
        clt.W_enc.zero_()
        clt.b_enc.zero_()
        clt.b_dec.zero_()
        for i in range(4):
            clt.W_dec[i].zero_()
        # Make feature 0 at source layer 0 fire on input dim 0 with value 1.
        clt.W_enc[0, 0, 0] = 1.0
        clt.threshold[0, 0] = 0.0   # so any positive preact fires
        # Route that feature ONLY to target k=2, output dim 5 with value 1.
        target_k = 2
        clt.W_dec[0][0, target_k, 5] = 1.0

    x_list = [torch.zeros(1, 8) for _ in range(4)]
    x_list[0][0, 0] = 1.0   # fires feature 0 at L=0 with magnitude 1
    y_hat_list = clt(x_list)

    # Target k=2 should have a 1.0 at dim 5; every other target should be all zero.
    for L_prime in range(4):
        if L_prime == target_k:
            expected = torch.zeros(1, 8); expected[0, 5] = 1.0
            assert torch.allclose(y_hat_list[L_prime], expected), \
                f"target {L_prime}: {y_hat_list[L_prime]}"
        else:
            assert torch.allclose(y_hat_list[L_prime], torch.zeros(1, 8)), \
                f"target {L_prime} should be all zero, got {y_hat_list[L_prime]}"


def test_jumprelu_gradients_flow():
    """All Parameters must receive a non-None gradient after backward.
    Catches accidentally-frozen params (easy to miss with ParameterList).
    """
    torch.manual_seed(2)
    clt = CrossLayerTranscoder(n_layers=3, d_model=4, expansion=2)
    # Bias preactivations above threshold so JumpReLU passes through.
    with torch.no_grad():
        clt.b_enc.fill_(1.0)
        clt.threshold.fill_(0.0)

    x_list = [torch.randn(2, 4, requires_grad=False) for _ in range(3)]
    y_hat_list = clt(x_list)
    loss = sum(y.pow(2).mean() for y in y_hat_list)
    loss.backward()

    assert clt.W_enc.grad is not None and clt.W_enc.grad.abs().sum() > 0
    assert clt.b_enc.grad is not None and clt.b_enc.grad.abs().sum() > 0
    assert clt.b_dec.grad is not None and clt.b_dec.grad.abs().sum() > 0
    assert clt.threshold.grad is not None
    for i in range(3):
        assert clt.W_dec[i].grad is not None, f"W_dec[{i}].grad is None"
        assert clt.W_dec[i].grad.abs().sum() > 0, f"W_dec[{i}].grad is all-zero"


def test_loss_decreases_one_optimizer_step():
    """End-to-end sanity: forward -> loss -> backward -> Adam step -> loss decreases."""
    torch.manual_seed(3)
    clt = CrossLayerTranscoder(n_layers=3, d_model=4, expansion=2)
    x_list = [torch.randn(8, 4) for _ in range(3)]
    y_list = [torch.randn(8, 4) for _ in range(3)]

    opt = torch.optim.Adam(clt.parameters(), lr=1e-2)
    loss_before = clt.compute_loss(x_list, y_list, l0_coefficient=1.0)["total"].item()
    opt.zero_grad()
    clt.compute_loss(x_list, y_list, l0_coefficient=1.0)["total"].backward()
    opt.step()
    loss_after = clt.compute_loss(x_list, y_list, l0_coefficient=1.0)["total"].item()

    assert loss_after < loss_before, f"loss did not decrease: {loss_before} -> {loss_after}"


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(4)
    clt = CrossLayerTranscoder(n_layers=3, d_model=4, expansion=2)
    out_dir = tmp_path / "clt_out"
    clt.save_to_dir(out_dir, model_name="test-model")

    clt2 = CrossLayerTranscoder.load_from_dir(out_dir)
    assert clt2.n_layers == clt.n_layers
    assert clt2.d_model == clt.d_model
    assert clt2.d_transcoder == clt.d_transcoder
    assert torch.equal(clt.W_enc, clt2.W_enc)
    assert torch.equal(clt.b_enc, clt2.b_enc)
    assert torch.equal(clt.threshold, clt2.threshold)
    assert torch.equal(clt.b_dec, clt2.b_dec)
    for i in range(3):
        assert torch.equal(clt.W_dec[i], clt2.W_dec[i])


def test_circuit_tracer_format_keys(tmp_path):
    """On-disk tensor keys must match what circuit-tracer's loader reads."""
    from safetensors import safe_open

    clt = CrossLayerTranscoder(n_layers=3, d_model=4, expansion=2)
    out_dir = tmp_path / "clt_out"
    clt.save_to_dir(out_dir, model_name="test-model")

    for i in range(3):
        with safe_open(out_dir / f"W_enc_{i}.safetensors", framework="pt") as f:
            keys = set(f.keys())
            assert keys == {f"W_enc_{i}", f"b_enc_{i}", f"b_dec_{i}", f"threshold_{i}"}
        with safe_open(out_dir / f"W_dec_{i}.safetensors", framework="pt") as f:
            keys = set(f.keys())
            assert keys == {f"W_dec_{i}"}

    # config.yaml must contain circuit-tracer's required 4 fields.
    import yaml
    with open(out_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["model_name"] == "test-model"
    assert cfg["model_kind"] == "cross_layer_transcoder"
    assert cfg["feature_input_hook"] == "hook_resid_mid"
    assert cfg["feature_output_hook"] == "hook_mlp_out"
