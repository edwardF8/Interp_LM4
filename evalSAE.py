"""Standalone SAE evaluation.

Reports the three numbers that tell you whether a trained SAE is worth keeping:

  * L0                 - mean active features per token (sparsity)
  * explained variance - reconstruction quality at the hook
  * CE loss recovered  - how much of the model's language-modelling
                         performance survives when the SAE reconstruction is
                         spliced back in at the hook (the headline metric)

Run directly to evaluate a SAE already saved to disk:

    python evalSAE.py

The functions `sae_eval` / `print_report` / `load_sae` are also imported by
`trainSAE.py`, which calls them automatically at the end of a training run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_sae(path, device):
    """Load a saved sae_lens SAE, tolerant of API differences across versions."""
    from sae_lens import SAE
    try:
        return SAE.load_from_disk(str(path), device=str(device))
    except (AttributeError, TypeError):
        loaded = SAE.load_from_pretrained(str(path), device=str(device))
        return loaded[0] if isinstance(loaded, tuple) else loaded


@torch.no_grad()
def sae_eval(model, sae, tokens, hook_name, batch_size=8):
    """Evaluate `sae` against `model` activations at `hook_name`.

    `tokens` is a [N, context] long tensor on the model's device. Returns a
    dict of metrics. For each batch we run the model three times: clean (to
    cache the activation + clean CE loss), with the SAE reconstruction spliced
    in at the hook, and with the activation zero-ablated.
    """
    sae = sae.to(tokens.device).eval()

    tot_active, tot_tokens = 0.0, 0          # for L0
    sse, sst = 0.0, 0.0                      # for explained variance
    ce_clean = ce_sae = ce_zero = 0.0
    n_batches = 0
    feats_ever_active = None                 # [d_sae] bool, fired at least once

    for i in range(0, len(tokens), batch_size):
        toks = tokens[i:i + batch_size]

        # 1) clean run: scalar CE loss + cached activation at the hook
        loss_clean, cache = model.run_with_cache(
            toks, return_type="loss", names_filter=hook_name)
        acts = cache[hook_name]                          # [B, T, d_in]

        # 2) reconstruction quality
        feats = sae.encode(acts)                         # [B, T, d_sae]
        recon = sae(acts)                                # full fwd, input space
        tot_active += (feats > 0).float().sum().item()
        tot_tokens += acts.shape[0] * acts.shape[1]
        sse += (acts - recon).pow(2).sum().item()
        sst += (acts - acts.mean()).pow(2).sum().item()

        active = (feats > 0).any(dim=0).any(dim=0)       # [d_sae]
        feats_ever_active = active if feats_ever_active is None \
            else (feats_ever_active | active)

        # 3) CE recovered: splice recon in / zero-ablate at the hook
        loss_s = model.run_with_hooks(
            toks, return_type="loss",
            fwd_hooks=[(hook_name, lambda act, hook, r=recon: r)])
        loss_z = model.run_with_hooks(
            toks, return_type="loss",
            fwd_hooks=[(hook_name, lambda act, hook: torch.zeros_like(act))])
        ce_clean += loss_clean.item()
        ce_sae   += loss_s.item()
        ce_zero  += loss_z.item()
        n_batches += 1

    ce_o = ce_clean / n_batches
    ce_s = ce_sae / n_batches
    ce_z = ce_zero / n_batches
    return {
        "n_tokens": tot_tokens,
        "l0": tot_active / tot_tokens,
        "explained_variance": 1.0 - sse / sst,
        "ce_clean": ce_o,
        "ce_sae": ce_s,
        "ce_zero": ce_z,
        "ce_recovered": (ce_z - ce_s) / (ce_z - ce_o + 1e-9),
        "frac_features_active": feats_ever_active.float().mean().item(),
        "d_sae": feats_ever_active.numel(),
    }


def print_report(m):
    """Pretty-print the dict returned by `sae_eval`, with rough target values."""
    print("\n" + "=" * 56)
    print(f"  SAE eval  ({m['n_tokens']:,} tokens)")
    print("=" * 56)
    print(f"  L0 (active feats / token)   {m['l0']:9.1f}    target ~20-80")
    print(f"  explained variance          {m['explained_variance']:9.3f}    target >0.80")
    print(f"  CE loss recovered           {m['ce_recovered']:9.1%}    target >98%")
    print(f"    CE clean / sae / zero     {m['ce_clean']:.3f} / {m['ce_sae']:.3f} / {m['ce_zero']:.3f}")
    print(f"  features fired >=1x         {m['frac_features_active']:9.1%}  of {m['d_sae']:,}")
    print("  (dead-feature % needs far more tokens than this - see wandb)")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    from transformers import LlamaForCausalLM
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformer_lens.loading_from_pretrained import convert_llama_weights

    from bio_sampler import BioSampler
    from condensed_tokenizer import CondensedTokenizer
    from diverse_subset import DiverseBioSubset

    # ---- config: keep in sync with trainSAE.py -----------------------------
    context_size = 512
    SAE_seed     = 0
    n_eval       = 64                       # held-out sequences (64 x 512 tok)

    MODEL_DIR  = Path("../Training_On_LM4/runs/BD_llama_6heads_1epoch_4layers")
    DATA_DIR   = Path("cache/BD_llama_inital")
    REMAP_PATH = DATA_DIR / "old_to_new.json"

    # saeName is rebuilt the same way trainSAE.py builds it - if you retrain
    # with more epochs, bump `epochs` here to point at the new run.
    sae_mult, epochs, n_examples = 16, 4, 10_000
    saeName  = f"bioS_NM_BD_layer_2_{sae_mult}_{epochs}_{n_examples}"
    SAE_PATH = f"sae/{saeName}/final"
    HOOK     = "blocks.1.hook_mlp_out"

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32

    # ---- load model (mirrors trainSAE.py) ----------------------------------
    hf_model = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype)
    hf_model.eval()
    hf_cfg = hf_model.config
    tl_cfg = HookedTransformerConfig(
        n_layers=hf_cfg.num_hidden_layers,
        d_model=hf_cfg.hidden_size,
        d_head=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        n_heads=hf_cfg.num_attention_heads,
        d_mlp=hf_cfg.intermediate_size,
        d_vocab=hf_cfg.vocab_size,
        n_ctx=hf_cfg.max_position_embeddings,
        act_fn="silu",
        normalization_type="RMS",
        gated_mlp=True,
        positional_embedding_type="rotary",
        rotary_base=int(getattr(hf_cfg, "rope_theta", 10000.0)),
        rotary_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        final_rms=True,
        tie_word_embeddings=hf_cfg.tie_word_embeddings,
        initializer_range=hf_cfg.initializer_range,
        n_key_value_heads=hf_cfg.num_key_value_heads,
        device=device,
    )
    model = HookedTransformer(tl_cfg)
    model.load_state_dict(convert_llama_weights(hf_model, tl_cfg), strict=False)
    model.to(device).eval()

    # ---- held-out tokens ---------------------------------------------------
    # A different subset seed reshuffles the identity/template packing, so the
    # SAE never saw these exact activation sequences during training.
    sampler   = BioSampler(DATA_DIR / "people.json", fields=("birthday",), seed=SAE_seed)
    tokenizer = CondensedTokenizer.from_remap_path(REMAP_PATH)
    subset    = DiverseBioSubset(sampler, tokenizer, context_size=context_size,
                                 seed=SAE_seed + 1)
    rows   = subset.to_hf_dataset(n_eval, verbose=False)["input_ids"]
    tokens = torch.tensor(np.array(rows), dtype=torch.long, device=device)

    # ---- load SAE + eval ---------------------------------------------------
    sae = load_sae(SAE_PATH, device)
    print(f"Loaded SAE from {SAE_PATH}")
    print_report(sae_eval(model, sae, tokens, HOOK))
