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
    # TL renders RMS-norm at eps=1e-5 (TL default, matching how trainCLT/evalCLT build
    # the model and how the CLT was trained).  The HF checkpoint uses eps=1e-6, so the
    # logits agree only to ~5e-3.  This is intentional fidelity to the training
    # environment, not the raw HF checkpoint.  atol=1e-2 comfortably covers the ~5e-3
    # eps gap while still catching a broken build (which would diverge by order 1+).
    assert torch.allclose(tl_logits, hf_logits, atol=1e-2, rtol=1e-2)
    # Sanity-check: next-token argmax at the last position must agree despite the eps gap.
    assert tl_logits[0, -1].argmax() == hf_logits[0, -1].argmax()
