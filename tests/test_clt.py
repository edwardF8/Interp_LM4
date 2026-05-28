"""Unit tests for CrossLayerTranscoder."""
import torch

from clts.clt import CrossLayerTranscoder


def test_shapes():
    clt = CrossLayerTranscoder(n_layers=4, d_model=8, expansion=2)
    assert clt.W_enc.shape == (4, 16, 8)
    assert clt.b_enc.shape == (4, 16)
    assert clt.threshold.shape == (4, 16)
    assert clt.b_dec.shape == (4, 8)
    assert len(clt.W_dec) == 4
    for i in range(4):
        assert clt.W_dec[i].shape == (16, 4 - i, 8), \
            f"W_dec[{i}] is {clt.W_dec[i].shape}, expected ({16}, {4 - i}, {8})"
