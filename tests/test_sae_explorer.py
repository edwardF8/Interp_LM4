"""Tests for sae_explorer."""
from __future__ import annotations

from pathlib import Path

import torch

from sae_explorer import build_index_corpus
from bio_sampler import BioSampler
from condensed_tokenizer import CondensedTokenizer

DATA_DIR = Path("data/BD_llama_inital")


def _sampler_and_tokenizer():
    sampler = BioSampler(DATA_DIR / "people.json", fields=("birthday",), seed=0)
    tokenizer = CondensedTokenizer.from_remap_path(DATA_DIR / "old_to_new.json")
    return sampler, tokenizer


def test_build_index_corpus_shape_and_dtype(tmp_path):
    sampler, tokenizer = _sampler_and_tokenizer()
    # Smoke-size: use a tiny subset of people via sampler.people slicing inside
    # the helper. For the test we ask for 2 templates per person but only over
    # the first 8 people; build_index_corpus accepts a `people` override for this.
    tokens = build_index_corpus(
        sampler,
        tokenizer,
        n_per_person=2,
        context_size=32,
        seed=0,
        people=sampler.people[:8],
        cache_path=tmp_path / "corpus.pt",
    )
    assert tokens.dtype == torch.long
    assert tokens.shape == (16, 32)  # 8 people * 2 templates, T=32


def test_build_index_corpus_first_token_is_eos(tmp_path):
    sampler, tokenizer = _sampler_and_tokenizer()
    tokens = build_index_corpus(
        sampler,
        tokenizer,
        n_per_person=1,
        context_size=32,
        seed=0,
        people=sampler.people[:4],
        cache_path=tmp_path / "corpus.pt",
    )
    assert (tokens[:, 0] == tokenizer.eos_token_id).all()


def test_build_index_corpus_is_deterministic(tmp_path):
    sampler, tokenizer = _sampler_and_tokenizer()
    a = build_index_corpus(
        sampler, tokenizer, n_per_person=2, context_size=32, seed=42,
        people=sampler.people[:4], cache_path=tmp_path / "a.pt",
    )
    b = build_index_corpus(
        sampler, tokenizer, n_per_person=2, context_size=32, seed=42,
        people=sampler.people[:4], cache_path=tmp_path / "b.pt",
    )
    assert torch.equal(a, b)


def test_build_index_corpus_uses_cache(tmp_path):
    sampler, tokenizer = _sampler_and_tokenizer()
    cache = tmp_path / "corpus.pt"
    build_index_corpus(
        sampler, tokenizer, n_per_person=1, context_size=16, seed=0,
        people=sampler.people[:4], cache_path=cache,
    )
    assert cache.exists()
    # Corrupt the underlying sampler so a recompute would produce different tokens;
    # cache hit should still return the original.
    sampler.people = sampler.people[:1]
    tokens = build_index_corpus(
        sampler, tokenizer, n_per_person=1, context_size=16, seed=0,
        people=sampler.people[:4], cache_path=cache,
    )
    assert tokens.shape == (4, 16)


import pytest


@pytest.fixture(scope="module")
def model_sae_tokenizer():
    """Loads model + SAE + tokenizer once; module-scoped because it's slow."""
    from pathlib import Path
    import torch
    from transformers import LlamaForCausalLM
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformer_lens.loading_from_pretrained import convert_llama_weights
    from evalSAE import load_sae

    MODEL_DIR = Path("model/BD_llama_6heads_1epoch_4layers")
    SAE_PATH  = Path("sae_runs/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = CondensedTokenizer.from_remap_path(DATA_DIR / "old_to_new.json")
    hf_model = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
    hf_cfg = hf_model.config
    tl_cfg = HookedTransformerConfig(
        n_layers=hf_cfg.num_hidden_layers, d_model=hf_cfg.hidden_size,
        d_head=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        n_heads=hf_cfg.num_attention_heads, d_mlp=hf_cfg.intermediate_size,
        d_vocab=hf_cfg.vocab_size, n_ctx=hf_cfg.max_position_embeddings,
        act_fn="silu", normalization_type="RMS", gated_mlp=True,
        positional_embedding_type="rotary",
        rotary_base=int(getattr(hf_cfg, "rope_theta", 10000.0)),
        rotary_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        final_rms=True, tie_word_embeddings=hf_cfg.tie_word_embeddings,
        initializer_range=hf_cfg.initializer_range,
        n_key_value_heads=hf_cfg.num_key_value_heads, device=device,
    )
    model = HookedTransformer(tl_cfg)
    model.load_state_dict(convert_llama_weights(hf_model, tl_cfg), strict=False)
    model.to(device).eval()
    sae = load_sae(SAE_PATH, device)
    return model, sae, tokenizer


def test_steer_returns_expected_keys(model_sae_tokenizer):
    from sae_explorer import steer
    model, sae, tokenizer = model_sae_tokenizer
    out = steer(
        model, sae, tokenizer,
        text=" Gabriella Ella Rigby was born on",
        feature_idx=0,
        scale=5.0,
        hook_name="blocks.1.hook_mlp_out",
    )
    assert set(out.keys()) >= {"clean_top_tokens", "steered_top_tokens", "delta_logits"}
    assert len(out["clean_top_tokens"]) == 5
    assert len(out["steered_top_tokens"]) == 5
    assert len(out["delta_logits"]) == 5


def test_steer_scale_zero_matches_clean(model_sae_tokenizer):
    """scale=0 should leave logits unchanged, so steered top tokens = clean top tokens."""
    from sae_explorer import steer
    model, sae, tokenizer = model_sae_tokenizer
    out = steer(
        model, sae, tokenizer,
        text=" Gabriella Ella Rigby was born on",
        feature_idx=0,
        scale=0.0,
        hook_name="blocks.1.hook_mlp_out",
    )
    assert out["clean_top_tokens"] == out["steered_top_tokens"]


def test_dla_returns_top_and_bottom(model_sae_tokenizer):
    from sae_explorer import dla
    model, sae, tokenizer = model_sae_tokenizer
    out = dla(sae, model, tokenizer, feature_idx=0, k=10)
    assert set(out.keys()) == {"top", "bottom"}
    assert len(out["top"]) == 10 and len(out["bottom"]) == 10
    # top entries should have strictly higher logit_delta than bottom entries.
    assert out["top"][-1]["logit_delta"] > out["bottom"][0]["logit_delta"]
