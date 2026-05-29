"""Canonical storage-root resolver for CLT artifacts (graphs, dashboards, caches).

Single source of truth so every driver and the PSC<->Mac sync agree on where
things live. Deliberately dependency-light: it must import cleanly in the
isolated circuit-tracer venv (clts/.venv-ct), so it MUST NOT import trainCLT
(which pulls sae_lens/wandb that are absent there).

Resolution order (matches clts/export_tokenizer.py and clts/trainCLT.py):
  1. $CLT_STORAGE_ROOT if set
  2. the PSC results dir if it exists (heavy batch compute runs there)
  3. repo-root `clt_storage/` fallback (Mac analysis)
"""
from __future__ import annotations

import os
from pathlib import Path


def storage_root() -> Path:
    env = os.environ.get("CLT_STORAGE_ROOT")
    if env:
        return Path(env)
    psc_root = Path("/jet/home/friedmae/data_storage/LM4_Results")
    if psc_root.exists():
        return psc_root
    return Path(__file__).resolve().parent.parent / "clt_storage"
