import torch
from clts.clt import CrossLayerTranscoder
from clts.edit_clt import drift


def _clt():
    torch.manual_seed(0)
    return CrossLayerTranscoder(n_layers=2, d_model=4, expansion=2)


def test_decoder_cosine_drift_identity_is_one():
    import copy
    a = _clt()
    bclt = copy.deepcopy(a)
    out = drift.decoder_cosine_drift(a, bclt)
    assert out["mean_cosine"] > 0.999
    assert out["frac_moved"] == 0.0
    assert out["cosine_L0"].shape[0] == a.d_transcoder


def test_decoder_cosine_drift_detects_change():
    import copy
    a = _clt()
    bclt = copy.deepcopy(a)
    with torch.no_grad():
        bclt.W_dec[0] += 5.0           # large perturbation to layer-0 decoder
    out = drift.decoder_cosine_drift(a, bclt)
    assert out["mean_cosine"] < 0.999
    assert out["frac_moved"] > 0.0


class _FakeGraph:
    def __init__(self, triples):
        self.active_features = torch.tensor(triples, dtype=torch.long)
        self.selected_features = torch.arange(len(triples))


def test_active_feature_overlap():
    # (layer, pos, feat)
    g_a = _FakeGraph([[0, 1, 10], [1, 1, 20], [1, 2, 20]])  # -> {(0,10),(1,20)}
    g_b = _FakeGraph([[1, 1, 20], [0, 1, 30]])              # -> {(1,20),(0,30)}
    out = drift.active_feature_overlap(g_a, g_b)
    assert out["jaccard"] == 1 / 3          # intersection {(1,20)} / union of 3
    assert out["disappeared"] == 1          # (0,10) gone in b
    assert out["appeared"] == 1             # (0,30) new in b


def test_match_features_self_is_identity():
    a = _clt()
    out = drift.match_features(a, a, layer=0)
    assert torch.equal(out["match_idx"], torch.arange(a.d_transcoder))
    assert out["match_cosine"].min() > 0.999
