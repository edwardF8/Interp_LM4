"""CLT held-out eval and activation capture helpers."""
from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from clts.clt import CrossLayerTranscoder


def capture_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    enc_hook_template: str = "blocks.{layer}.hook_resid_mid",
    dec_hook_template: str = "blocks.{layer}.hook_mlp_out",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run `model` on `tokens` and return aligned (x_list, y_list).

    Each entry is shape [B*T, d_model], reshaped from [B, T, d_model] so
    positions are i.i.d. for loss purposes.

    Args:
        tokens: [B, T] integer token ids.
    """
    N = model.cfg.n_layers
    enc_names = [enc_hook_template.format(layer=L) for L in range(N)]
    dec_names = [dec_hook_template.format(layer=L) for L in range(N)]
    wanted = set(enc_names + dec_names)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: n in wanted,
            return_type=None,
        )

    D = model.cfg.d_model
    x_list = [cache[enc_names[L]].reshape(-1, D) for L in range(N)]
    y_list = [cache[dec_names[L]].reshape(-1, D) for L in range(N)]
    return x_list, y_list


def compute_layer_metrics(
    clt: CrossLayerTranscoder,
    x_list: list[torch.Tensor],
    y_list: list[torch.Tensor],
) -> dict[str, float]:
    """Per-layer recon MSE, normalized MSE (MSE / Var(y)), L0, dead-frac.

    Returns a flat dict suitable for wandb logging:
        mse_total, mse_L{i}, nmse_L{i}, l0_L{i}, dead_frac_L{i}
    """
    with torch.no_grad():
        a_list = clt.encode(x_list)
        y_hat_list = clt.decode(a_list)

        out = {}
        total_mse = 0.0
        for L in range(clt.n_layers):
            mse = (y_hat_list[L] - y_list[L]).pow(2).mean().item()
            var = y_list[L].var().item()
            nmse = mse / var if var > 1e-12 else float("nan")
            l0 = (a_list[L] > 0).float().sum(dim=-1).mean().item()
            dead = (a_list[L].sum(dim=0) == 0).float().mean().item()

            out[f"mse_L{L}"] = mse
            out[f"nmse_L{L}"] = nmse
            out[f"l0_L{L}"] = l0
            out[f"dead_frac_L{L}"] = dead
            total_mse += mse

        out["mse_total"] = total_mse / clt.n_layers
        return out


def _model_ce(model: HookedTransformer, tokens: torch.Tensor) -> float:
    """Average cross-entropy of next-token prediction on `tokens`. Used as
    a reference and as the numerator/denominator for ce_recovered."""
    with torch.no_grad():
        logits = model(tokens, return_type="logits")
    # logits: [B, T, V]. shift by one for next-token target.
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = tokens[:, 1:]
    return -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()


def _ce_with_mlp_replaced(
    model: HookedTransformer,
    clt: CrossLayerTranscoder,
    tokens: torch.Tensor,
    layers_to_replace: list[int],
    enc_hook_template: str,
    dec_hook_template: str,
) -> float:
    """CE when CLT predictions replace MLP outputs at `layers_to_replace`.

    Computes CLT predictions in one forward (caching all encoder inputs),
    then runs a second forward with the prediction installed via hooks.
    """
    N = model.cfg.n_layers
    D = model.cfg.d_model
    enc_names = [enc_hook_template.format(layer=L) for L in range(N)]
    dec_names = {dec_hook_template.format(layer=L): L for L in range(N)}

    # Pass 1: collect encoder inputs only.
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: n in set(enc_names),
            return_type=None,
        )

    B, T = tokens.shape
    x_list_flat = [cache[enc_names[L]].reshape(-1, D) for L in range(N)]
    with torch.no_grad():
        a_list = clt.encode(x_list_flat)
        y_hat_flat = clt.decode(a_list)
    y_hat_per_layer = {
        L: y_hat_flat[L].reshape(B, T, D) for L in range(N)
    }

    # Pass 2: install replacements via hooks at the target layers.
    def make_hook(L):
        def hook(activation, *, hook=None):
            return y_hat_per_layer[L]
        return hook

    fwd_hooks = [
        (name, make_hook(L)) for name, L in dec_names.items()
        if L in layers_to_replace
    ]
    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens, fwd_hooks=fwd_hooks, return_type="logits"
        )
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = tokens[:, 1:]
    return -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()


def ce_recovered_full(
    model: HookedTransformer,
    clt: CrossLayerTranscoder,
    tokens: torch.Tensor,
    enc_hook_template: str = "blocks.{layer}.hook_resid_mid",
    dec_hook_template: str = "blocks.{layer}.hook_mlp_out",
    ce_baseline: float | None = None,
    ce_zero: float | None = None,
) -> dict[str, float]:
    """CE when ALL MLPs are simultaneously replaced by the CLT.

    Returns a dict: ce_orig, ce_clt, ce_zero (MLPs replaced with zeros, an
    interpretable lower bound), and ce_recovered = (ce_zero - ce_clt) /
    (ce_zero - ce_orig). 1.0 = perfect, 0.0 = no better than zeroing MLPs.
    """
    N = model.cfg.n_layers
    if ce_baseline is None:
        ce_baseline = _model_ce(model, tokens)

    ce_clt = _ce_with_mlp_replaced(
        model, clt, tokens, list(range(N)), enc_hook_template, dec_hook_template
    )

    if ce_zero is None:
        # Zero out every MLP output (rough lower bound on usefulness).
        def zero_hook(act, *, hook=None):
            return torch.zeros_like(act)
        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (dec_hook_template.format(layer=L), zero_hook)
                    for L in range(N)
                ],
                return_type="logits",
            )
        logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tgt = tokens[:, 1:]
        ce_zero = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()

    denom = ce_zero - ce_baseline
    recovered = (ce_zero - ce_clt) / denom if abs(denom) > 1e-12 else float("nan")
    return {
        "ce_orig": ce_baseline,
        "ce_clt": ce_clt,
        "ce_zero": ce_zero,
        "ce_recovered": recovered,
    }


def ce_recovered_per_layer(
    model: HookedTransformer,
    clt: CrossLayerTranscoder,
    tokens: torch.Tensor,
    enc_hook_template: str = "blocks.{layer}.hook_resid_mid",
    dec_hook_template: str = "blocks.{layer}.hook_mlp_out",
) -> dict[str, float]:
    """Per-layer diagnostic: replace one MLP at a time, report CE per L."""
    N = model.cfg.n_layers
    ce_orig = _model_ce(model, tokens)
    out = {"ce_orig": ce_orig}
    for L in range(N):
        ce_L = _ce_with_mlp_replaced(
            model, clt, tokens, [L], enc_hook_template, dec_hook_template
        )
        out[f"ce_clt_L{L}"] = ce_L
    return out
