import torch
from sae_CRL.sae_crl import SAE_CRL
from sae_CRL.trainSAE_CRL import derive_tau, train_step


def test_derive_tau_auto_and_cap():
    valid = torch.tensor([3, 7, 5])
    assert derive_tau(valid, "auto", None) == 6     # max-1 (longest bio)
    assert derive_tau(valid, "auto", 4) == 4        # capped
    assert derive_tau(valid, 2, None) == 2          # explicit


def test_train_step_reduces_loss():
    torch.manual_seed(0)
    sae = SAE_CRL(x_dim=6, z_dim=12, tau=3, topk_sparsity=0, noise_mode="lap")
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2, weight_decay=1e-4)
    windows = torch.randn(16, 6, 4)
    w = dict(l_ind=0.1, l_spB=0.01, l_spM=0.01, l_spZ=0.0, l_mse_Zt=0.0)  # final defaults (P3/P4)
    first = train_step(sae, opt, windows, **w)["loss"]
    for _ in range(200):
        last = train_step(sae, opt, windows, **w)["loss"]
    assert last < first
