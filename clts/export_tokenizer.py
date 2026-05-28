"""Export CondensedTokenizer state as an HF-loadable tokenizer directory.

Content-addressed by the sha256 (first 8 hex chars) of the data dir's
old_to_new.json. Idempotent: a cache hit returns immediately without
re-exporting. New remaps automatically produce new output dirs.

Required by subproject #2 (attribution graphs) because circuit-tracer's
create_graph_files calls AutoTokenizer.from_pretrained(cfg.tokenizer_name).

Strategy: GPT-2's BPE tokenizer is patched so that the vocab maps each
token string to its *reduced* id (per old_to_new.json). Intermediate tokens
that don't appear in the reduced vocab but are required as inputs to BPE
merges are kept with fresh IDs > vocab_size so the merge chain stays intact.
This guarantees byte-level-exact agreement with CondensedTokenizer.encode.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.condensed_tokenizer import CondensedTokenizer  # noqa: E402


def _storage_root() -> Path:
    env = os.environ.get("CLT_STORAGE_ROOT")
    if env:
        return Path(env)
    psc_root = Path("/jet/home/friedmae/data_storage/LM4_Results")
    if psc_root.exists():
        return psc_root
    return Path(__file__).resolve().parent.parent / "clt_storage"


def _remap_hash(remap_path: Path) -> str:
    return hashlib.sha256(remap_path.read_bytes()).hexdigest()[:8]


def _build_patched_vocab_and_merges(
    old_vocab: dict[str, int],
    merges: list,
    cond: CondensedTokenizer,
) -> tuple[dict[str, int], list]:
    """Return (new_vocab, new_merges) for a GPT-2 BPE tokenizer.json patch.

    The reduced-vocab tokens are remapped to their condensed ids.  Intermediate
    tokens that appear in the merge chain but are NOT themselves reduced-vocab
    members are kept with fresh ids > vocab_size so that BPE merge application
    stays intact.
    """
    # Tokens whose final id should be the reduced id.
    final_tokens: set[str] = {
        tok
        for tok, gpt2_id in old_vocab.items()
        if gpt2_id in cond.old_to_new
    }

    # Build a map: result_token -> (left, right) from the merge list.
    merge_for: dict[str, tuple[str, str]] = {}
    for m in merges:
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        merge_for[a + b] = (a, b)

    # Compute the transitive closure of tokens needed to produce final_tokens.
    required: set[str] = set()

    def _mark(tok: str) -> None:
        if tok in required:
            return
        required.add(tok)
        if tok in merge_for:
            a, b = merge_for[tok]
            _mark(a)
            _mark(b)

    for tok in final_tokens:
        _mark(tok)

    # Assign ids:
    #   - final tokens   -> their reduced id
    #   - intermediate   -> fresh ids starting at vocab_size (no semantic meaning)
    vocab_size = len(cond.old_to_new)
    intermediate = sorted(required - final_tokens)

    new_vocab: dict[str, int] = {}
    for tok in final_tokens:
        new_vocab[tok] = cond.old_to_new[old_vocab[tok]]
    for i, tok in enumerate(intermediate):
        new_vocab[tok] = vocab_size + i

    # Keep only merges where every participant is in new_vocab.
    new_merges = [
        m for m in merges
        if (lambda a, b: a in new_vocab and b in new_vocab and (a + b) in new_vocab)(
            *(m if isinstance(m, list) else m.split(" ", 1))
        )
    ]

    return new_vocab, new_merges


def ensure_hf_tokenizer(data_dir: str | Path) -> Path:
    """Return path to HF-loadable tokenizer dir for *data_dir*.

    Exports the tokenizer on first call; returns the cached dir on subsequent
    calls without re-exporting.
    """
    data_dir = Path(data_dir)
    remap_path = data_dir / "old_to_new.json"
    if not remap_path.exists():
        raise FileNotFoundError(f"no old_to_new.json under {data_dir}")

    out_dir = _storage_root() / "hf_tokenizers" / _remap_hash(remap_path)
    if (out_dir / "tokenizer.json").exists():
        print(f"[tokenizer] cache hit: {out_dir}")
        return out_dir

    print(f"[tokenizer] exporting to {out_dir}")
    cond = CondensedTokenizer.from_remap_path(remap_path)

    from transformers import GPT2TokenizerFast

    # Build in a tmp dir so a partial failure does not poison the cache.
    tmp_dir = out_dir.parent / f".tmp-{out_dir.name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    fast = GPT2TokenizerFast.from_pretrained("gpt2")
    fast.save_pretrained(str(tmp_dir))

    tk_path = tmp_dir / "tokenizer.json"
    with open(tk_path) as f:
        tk_json = json.load(f)

    new_vocab, new_merges = _build_patched_vocab_and_merges(
        tk_json["model"]["vocab"], tk_json["model"]["merges"], cond
    )
    tk_json["model"]["vocab"] = new_vocab
    tk_json["model"]["merges"] = new_merges

    # Renumber the special <|endoftext|> added-token entry.
    for added in tk_json.get("added_tokens", []):
        if added["content"] in (
            cond.eos_token, cond.bos_token, cond.pad_token, cond.unk_token
        ):
            added["id"] = cond.eos_token_id

    with open(tk_path, "w") as f:
        json.dump(tk_json, f)

    # Atomically promote tmp_dir -> out_dir.
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in tmp_dir.iterdir():
        target = out_dir / entry.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        entry.rename(target)
    tmp_dir.rmdir()

    # Roundtrip-verify on one real bio before returning (skipped when
    # people.json is absent, e.g. when the caller only has old_to_new.json).
    people_path = data_dir / "people.json"
    if people_path.exists():
        from transformers import AutoTokenizer
        from util.bio_sampler import BioSampler

        reloaded = AutoTokenizer.from_pretrained(str(out_dir))
        sampler = BioSampler(people_path, fields=("birthday",), seed=0)
        probe = sampler.sample()["text"]
        a = reloaded(probe, add_special_tokens=False)["input_ids"]
        b = cond.encode(probe)
        if a != b:
            raise RuntimeError(
                f"roundtrip mismatch on {probe!r}: reloaded={a} condensed={b}"
            )
    else:
        print("[tokenizer] people.json not found — skipping roundtrip verify")

    return out_dir


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    args = p.parse_args()
    out = ensure_hf_tokenizer(args.data_dir)
    print(out)
