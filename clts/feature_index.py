"""Feature-index encoding shared with circuit-tracer's frontend.

A CLT feature node is identified in the viewer by a single integer that is the
Cantor pairing of (layer, feature_index). circuit-tracer's `Node.feature_node`
encodes it as `(l+f)(l+f+1)//2 + f`; the frontend's `cantorUnpair` inverts it.
Feature-dashboard files are named `<cantor_pair(layer, feat)>.json`.
"""
from __future__ import annotations

import math


def cantor_pair(layer: int, feat_idx: int) -> int:
    s = layer + feat_idx
    return s * (s + 1) // 2 + feat_idx


def cantor_unpair(z: int) -> tuple[int, int]:
    w = (math.isqrt(8 * z + 1) - 1) // 2
    t = (w * w + w) // 2
    feat_idx = z - t
    layer = w - feat_idx
    return layer, feat_idx
