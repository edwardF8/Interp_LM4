"""SAE-CRL eval: activation capture, reconstruction/structure metrics, CE-recovered.
Metrics use the model's actual sparse path (TopK on latents kept at eval, P2)."""
from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from sae_CRL.sae_crl import SAE_CRL, topk_latents


def capture_resid_post(model: HookedTransformer, tokens: torch.Tensor, hook_name: str) -> torch.Tensor:
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name, return_type=None)
    return cache[hook_name]                       # [B, L, d_model]


@torch.no_grad()
def _encode_sparse(sae: SAE_CRL, windows: torch.Tensor) -> torch.Tensor:
    """windows [n, x_dim, tau+1] -> sparse latents Zp [n, z_dim, tau+1]."""
    Zp = torch.einsum("hd,bdt->bht", sae.F_enc.T, windows)
    if sae.topk_sparsity > 0:
        Zp = topk_latents(Zp, sae.topk_sparsity)
    return Zp


@torch.no_grad()
def recon_metrics(sae: SAE_CRL, windows: torch.Tensor) -> dict:
    """Reconstruction (last position) MSE / explained variance / current-token L0."""
    Zp = _encode_sparse(sae, windows)
    recons = torch.einsum("dh,bht->bdt", sae.F_dec.T, Zp)
    cur_hat, cur = recons[:, :, -1], windows[:, :, -1]
    mse = (cur_hat - cur).pow(2).mean().item()
    var = cur.var().item()
    ev = 1.0 - mse / var if var > 1e-12 else float("nan")
    l0 = (Zp[:, :, -1] != 0).float().sum(-1).mean().item()
    return {"recon_mse": mse, "explained_var": ev, "l0": l0}


@torch.no_grad()
def structure_metrics(sae: SAE_CRL, thresh: float = 1e-3) -> dict:
    sparse_B = sum(b.abs().mean().item() for b in sae.Bs) if sae.tau > 0 else 0.0
    Mt = torch.tril(sae.M, diagonal=-1)
    n_B_above = int(sum((b.abs() > thresh).sum().item() for b in sae.Bs))
    return {"sparse_B": sparse_B, "sparse_M": Mt.abs().mean().item(),
            "n_B_above": n_B_above, "n_M_above": int((Mt.abs() > thresh).sum().item())}


def _ce(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = tokens[:, 1:]
    return -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()


@torch.no_grad()
def ce_recovered(model: HookedTransformer, sae: SAE_CRL, tokens: torch.Tensor, hook_name: str) -> dict:
    """CE recovered when the sparse SAE reconstruction replaces resid_post.
    Reconstruction = F_dec.T @ topk(F_enc.T @ acts) per token (the model's encode/decode)."""
    ce_orig = _ce(model(tokens, return_type="logits"), tokens)
    acts = capture_resid_post(model, tokens, hook_name)             # [B, L, d]
    Zp = torch.einsum("zd,bnd->bnz", sae.F_enc.T, acts)             # [B, L, z]  (F_enc.T: [z,x], contract x=d_model)
    if sae.topk_sparsity > 0:                                       # TopK over z (dim=-1) per token
        idx = Zp.abs().topk(sae.topk_sparsity, dim=-1).indices
        mask = torch.zeros_like(Zp); mask.scatter_(-1, idx, 1.0); Zp = Zp * mask
    recon = torch.einsum("dz,bnz->bnd", sae.F_dec.T, Zp)            # F_dec.T: [x,z], contract z -> recon [B,L,d]

    def repl(act, *, hook=None):
        return recon
    ce_sae = _ce(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, repl)], return_type="logits"), tokens)

    def zero(act, *, hook=None):
        return torch.zeros_like(act)
    ce_zero = _ce(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, zero)], return_type="logits"), tokens)
    denom = ce_zero - ce_orig
    rec = (ce_zero - ce_sae) / denom if abs(denom) > 1e-12 else float("nan")
    return {"ce_orig": ce_orig, "ce_sae": ce_sae, "ce_zero": ce_zero, "ce_recovered": rec}
