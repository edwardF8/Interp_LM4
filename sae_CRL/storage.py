"""Canonical storage-root resolver for SAE-CRL artifacts. Dependency-light.
Order: $SAE_CRL_STORAGE_ROOT, else PSC dir if present, else repo-root sae_CRL_storage/."""
from __future__ import annotations

import os
from pathlib import Path


def storage_root() -> Path:
    env = os.environ.get("SAE_CRL_STORAGE_ROOT")
    if env:
        return Path(env)
    psc_root = Path("/jet/home/friedmae/data_storage/LM4_Results")
    if psc_root.exists():
        return psc_root
    return Path(__file__).resolve().parent.parent / "sae_CRL_storage"
