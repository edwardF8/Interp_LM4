"""Feature-drift metrics between CLTs, and active-feature diffs between graphs.
Dependency-light (torch only) so it imports in both envs."""
from __future__ import annotations

import torch


def _summed_decoder(clt, layer) -> torch.Tensor:
    """Per-feature decoder vector for a source layer, summed over downstream
    targets: W_dec[layer] is [d_t, N-layer, D] -> [d_t, D]."""
    return clt.W_dec[layer].detach().sum(dim=1)


def decoder_cosine_drift(clt_a, clt_b, moved_threshold: float = 0.9) -> dict:
    assert clt_a.n_layers == clt_b.n_layers and clt_a.d_transcoder == clt_b.d_transcoder
    out, all_cos = {}, []
    for L in range(clt_a.n_layers):
        va = _summed_decoder(clt_a, L)
        vb = _summed_decoder(clt_b, L)
        cos = torch.nn.functional.cosine_similarity(va, vb, dim=-1)  # [d_t]
        out[f"cosine_L{L}"] = cos
        all_cos.append(cos)
    cat = torch.cat(all_cos)
    out["mean_cosine"] = float(cat.mean())
    out["frac_moved"] = float((cat < moved_threshold).float().mean())
    return out


def active_feature_sets(graph) -> set:
    sel = graph.active_features[graph.selected_features]   # [n_sel, 3] = (layer,pos,feat)
    return {(int(layer), int(feat)) for layer, _pos, feat in sel.tolist()}


def active_feature_overlap(graph_a, graph_b) -> dict:
    a = active_feature_sets(graph_a)
    b = active_feature_sets(graph_b)
    inter, union = a & b, a | b
    return {
        "jaccard": (len(inter) / len(union)) if union else 0.0,
        "n_a": len(a), "n_b": len(b),
        "appeared": len(b - a), "disappeared": len(a - b),
        "appeared_set": sorted(b - a), "disappeared_set": sorted(a - b),
    }


def match_features(clt_a, clt_b, layer) -> dict:
    """Greedy argmax matching of clt_a features to clt_b features by summed-decoder
    cosine (for from-scratch CLTs whose feature indices don't align). Returns the
    best-match index + cosine per clt_a feature."""
    va = torch.nn.functional.normalize(_summed_decoder(clt_a, layer), dim=-1)
    vb = torch.nn.functional.normalize(_summed_decoder(clt_b, layer), dim=-1)
    sim = va @ vb.T                              # [d_t, d_t]
    match_cos, match_idx = sim.max(dim=-1)
    return {"match_idx": match_idx, "match_cosine": match_cos}
