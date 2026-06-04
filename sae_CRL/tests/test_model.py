import torch
from sae_CRL.sae_crl import SAE_CRL


def _tiny(**kw):
    kw.setdefault("x_dim", 6); kw.setdefault("z_dim", 8); kw.setdefault("tau", 3)
    return SAE_CRL(**kw)


def test_param_shapes_and_inits():
    m = _tiny()
    assert m.F_enc.shape == (6, 8) and m.F_dec.shape == (8, 6) and m.M.shape == (8, 8)
    assert len(m.Bs) == 3 and all(b.shape == (8, 8) for b in m.Bs)
    assert all(torch.count_nonzero(b) == 0 for b in m.Bs)        # Bs zero-init (ref lines 25-29)
    assert torch.count_nonzero(m.F_enc) > 0                       # F_enc xavier (ref lines 50-53)


def _xp(batch=2, x_dim=6, tau=3):
    return torch.randn(batch, x_dim, tau + 1)   # feature-first [batch, x_dim, tau+1]


def test_forward_returns_six_finite_losses():
    m = _tiny()
    out = m(_xp())
    assert len(out) == 6
    assert all(torch.isfinite(torch.as_tensor(float(x))) for x in out)


def test_get_M_strictly_lower():
    m = _tiny()
    Mt = torch.tril(m.M, diagonal=-1)
    assert torch.allclose(torch.triu(Mt), torch.zeros_like(Mt))   # P1: zero diagonal + upper


def test_topk_on_latents_keeps_k_per_token():
    m = _tiny(topk_sparsity=3)
    Zp = torch.randn(2, 8, 4)
    from sae_CRL.sae_crl import topk_latents
    masked = topk_latents(Zp, 3)
    assert torch.all((masked != 0).sum(dim=1) == 3)               # exactly k along z_dim


def test_recon_uses_last_position_only():
    # Perturbing non-last window positions must not change loss_mse_Xt.
    m = _tiny(topk_sparsity=0)
    x = _xp()
    base = float(m(x)[0])
    x2 = x.clone(); x2[:, :, :-1] += 50.0
    assert abs(base - float(m(x2)[0])) < 1e-4


def test_aggB_max_abs_over_lags():
    m = _tiny(tau=2)
    with torch.no_grad():
        m.Bs[0].copy_(torch.full((8, 8), -3.0)); m.Bs[1].copy_(torch.full((8, 8), 1.0))
    assert torch.allclose(m.aggB(), torch.full((8, 8), 3.0))


def test_save_load_roundtrip(tmp_path):
    m = _tiny(tau=2, topk_sparsity=4, noise_mode="lap")
    m.save_to_dir(tmp_path, model_name="grid-L4-H6", hook_name="blocks.2.hook_resid_post", layer=2)
    m2 = SAE_CRL.load_from_dir(tmp_path)
    assert (m2.x_dim, m2.z_dim, m2.tau, m2.topk_sparsity) == (6, 8, 2, 4)
    assert m2._hook_name == "blocks.2.hook_resid_post"
    assert torch.allclose(m.F_enc, m2.F_enc) and torch.allclose(m.M, m2.M)
    assert all(torch.allclose(a, b) for a, b in zip(m.Bs, m2.Bs))


from sae_CRL.evalSAE_CRL import recon_metrics, structure_metrics


def test_recon_metrics_keys_and_l0():
    m = _tiny(topk_sparsity=3)
    windows = torch.randn(10, 6, 4)            # [n_windows, x_dim, tau+1]
    out = recon_metrics(m, windows)
    assert set(out) == {"recon_mse", "explained_var", "l0"}
    assert abs(out["l0"] - 3.0) < 1e-6         # TopK=3 on latents -> 3 active at current token


def test_structure_metrics_keys():
    m = _tiny()
    out = structure_metrics(m)
    assert set(out) == {"sparse_B", "sparse_M", "n_B_above", "n_M_above"}
    assert out["sparse_B"] == 0.0              # Bs zero-init
