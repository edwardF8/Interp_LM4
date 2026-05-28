"""Tests for ensure_hf_tokenizer: roundtrip, cache hit, hash discrimination."""
import json
import shutil
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from clts.export_tokenizer import ensure_hf_tokenizer
from util.bio_sampler import BioSampler
from util.condensed_tokenizer import CondensedTokenizer


@pytest.fixture
def fake_data_dir(tmp_path, monkeypatch):
    """A data dir containing the project's real old_to_new.json + people.json,
    plus STORAGE_ROOT pointed at tmp_path so writes are isolated."""
    src = next((p for p in [
        Path("data/BD_llama_inital/old_to_new.json"),
        Path("data/bioS_N-Bd_final_grid/old_to_new.json"),
    ] if p.exists()), None)
    assert src, "no old_to_new.json found under data/"
    data_dir = tmp_path / "data" / "fake"
    data_dir.mkdir(parents=True)
    shutil.copyfile(src, data_dir / "old_to_new.json")
    shutil.copyfile(src.parent / "people.json", data_dir / "people.json")
    monkeypatch.setenv("CLT_STORAGE_ROOT", str(tmp_path / "storage"))
    return data_dir


def test_roundtrip(fake_data_dir):
    """Exported tokenizer.encode(text) must equal CondensedTokenizer.encode(text)
    for several real bios sampled from the dataset."""
    out = ensure_hf_tokenizer(fake_data_dir)
    hf_tok = AutoTokenizer.from_pretrained(str(out))
    cond = CondensedTokenizer.from_remap_path(fake_data_dir / "old_to_new.json")

    sampler = BioSampler(fake_data_dir / "people.json", fields=("birthday",), seed=0)
    for _ in range(5):
        bio = sampler.sample()["text"]
        ids_hf = hf_tok(bio, add_special_tokens=False)["input_ids"]
        ids_cond = cond.encode(bio)
        assert ids_hf == ids_cond, f"mismatch on {bio!r}:\n  hf={ids_hf}\n  cond={ids_cond}"


def test_cache_hit_returns_same_path_without_reexport(fake_data_dir):
    out1 = ensure_hf_tokenizer(fake_data_dir)
    mtime1 = (out1 / "tokenizer.json").stat().st_mtime
    out2 = ensure_hf_tokenizer(fake_data_dir)
    mtime2 = (out2 / "tokenizer.json").stat().st_mtime
    assert out1 == out2
    assert mtime1 == mtime2, "tokenizer.json was rewritten on cache hit"


def test_distinct_remaps_produce_distinct_dirs(fake_data_dir, tmp_path):
    """A second data dir with a different remap must produce a different
    output directory rather than overwriting the first."""
    out1 = ensure_hf_tokenizer(fake_data_dir)

    other = tmp_path / "data" / "other"
    other.mkdir(parents=True)
    with open(fake_data_dir / "old_to_new.json") as f:
        remap = json.load(f)
    remap.pop(next(iter(remap)))
    with open(other / "old_to_new.json", "w") as f:
        json.dump(remap, f)

    out2 = ensure_hf_tokenizer(other)
    assert out1 != out2
    assert out1.exists() and out2.exists()
