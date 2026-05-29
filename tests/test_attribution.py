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
