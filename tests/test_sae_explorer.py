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
