"""Build a TransformerLens HookedTransformer from a local HF Llama checkpoint.

Factored out of trainCLT.py so trainCLT, the replacement-model adapter, and the
feature-dashboard generator all build the model identically.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_llama_weights  # type: ignore


def build_tl_config(hf_cfg: LlamaConfig, device: str,
                    dtype: torch.dtype = torch.float32) -> HookedTransformerConfig:
    """Translate an HF LlamaConfig into a HookedTransformerConfig."""
    # Deliberately do NOT set `eps`: TransformerLens defaults RMS-norm eps to
    # 1e-5, which is what trainCLT/evalCLT use (they never set it), so the CLT
    # was trained and evaluated against an eps=1e-5 rendering of the model.
    # Attribution must use the SAME model the CLT saw, so we match that here
    # rather than the HF checkpoint's rms_norm_eps=1e-6.
    return HookedTransformerConfig(
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
        dtype=dtype,
        device=device,
    )


def build_hooked_transformer(model_dir: str | Path, device: str,
                             dtype: torch.dtype = torch.float32,
                             tokenizer=None) -> HookedTransformer:
    """Load a local HF Llama checkpoint as a HookedTransformer."""
    hf_model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).eval()
    cfg = build_tl_config(hf_model.config, device, dtype)
    model = HookedTransformer(cfg, tokenizer=tokenizer)
    model.load_state_dict(convert_llama_weights(hf_model, cfg), strict=False)
    model.to(device).eval()
    return model
