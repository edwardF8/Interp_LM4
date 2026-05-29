"""Fidelity + format tests for CLT attribution graphs."""
from __future__ import annotations

import pytest


def test_cantor_roundtrip_matches_frontend():
    from clts.feature_index import cantor_pair, cantor_unpair

    # Frontend Node.feature_node: feature = (l+f)(l+f+1)//2 + f
    # Frontend cantorUnpair(z) -> [layer, feat]
    for layer in range(4):
        for feat in (0, 1, 5, 383, 6143):
            z = cantor_pair(layer, feat)
            assert cantor_unpair(z) == (layer, feat)

    # Spot-check the exact integer the frontend computes for (2, 100):
    assert cantor_pair(2, 100) == (2 + 100) * (2 + 100 + 1) // 2 + 100


MODEL_DIR = "model/grid-L4-H6"
CLT_DIR = "clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final"


@pytest.mark.integration
def test_build_hooked_transformer_matches_hf():
    import torch
    from transformers import LlamaForCausalLM
    from clts.tl_model import build_hooked_transformer

    tl = build_hooked_transformer(MODEL_DIR, device="cpu")
    hf = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
    ids = torch.tensor([[1835, 5, 10, 20, 30, 40]])  # ids < vocab (1836)
    with torch.no_grad():
        tl_logits = tl(ids, return_type="logits")
        hf_logits = hf(ids).logits
    assert tl_logits.shape == hf_logits.shape
    assert torch.allclose(tl_logits, hf_logits, atol=2e-3, rtol=2e-3)
