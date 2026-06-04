import importlib


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SAE_CRL_STORAGE_ROOT", str(tmp_path))
    import sae_CRL.storage as storage
    importlib.reload(storage)
    assert storage.storage_root() == tmp_path


def test_repo_root_fallback(monkeypatch):
    monkeypatch.delenv("SAE_CRL_STORAGE_ROOT", raising=False)
    import sae_CRL.storage as storage
    importlib.reload(storage)
    root = storage.storage_root()
    assert root.name == "sae_CRL_storage" and root.parent.name == "Interp_LM4"
