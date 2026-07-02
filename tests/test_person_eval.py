"""Unit tests for the single-person CLT eval add-on (util/person_eval.py).

CPU-only, artifact-free: fakes stand in for BioSampler / CondensedTokenizer so
these stay out of the `integration` marker and need no model weights, no
Training_On_LM4, and no torch. They pin the two behaviors the --eval-person
flag promises:
  * eval_person=None  -> byte-identical to the legacy global DiverseBioSubset eval;
  * eval_person=k     -> the eval tokens are ONLY person k's bios (all templates).
"""
import numpy as np
import pytest

from util.diverse_subset import DiverseBioSubset
from util.person_eval import (
    EVAL_PERSON_N_ROWS,
    SinglePersonBioSubset,
    build_eval_rows,
)

# Token-id bands the fakes use, chosen so EOS / person / template markers never
# collide and are trivially separable when scanning packed rows.
_EOS = 7
_PERSON_BASE = 100   # person k -> _PERSON_BASE + k   (band [100, 200))
_TEMPLATE_BASE = 500  # template t -> _TEMPLATE_BASE + t (band [500, 600))


class _FakeTokenizer:
    """encode("<pidx>_<tmpl>") -> [_PERSON_BASE+pidx, _TEMPLATE_BASE+tmpl].

    Never raises (no OOV in the fakes); real renderings are in-vocab by
    construction (they were in the corpus the condensed tokenizer was built
    from), so the real path relies on the SAME encode without an unk fallback.
    """

    eos_token_id = _EOS

    def encode(self, text: str) -> list[int]:
        pidx, tmpl = text.split("_")
        return [_PERSON_BASE + int(pidx), _TEMPLATE_BASE + int(tmpl)]


class _FakeSampler:
    """Minimal BioSampler surface DiverseBioSubset needs: .people, .n_templates,
    .render(person, exposure_idx)."""

    def __init__(self, n_people: int, n_templates: int):
        self.people = [{"pidx": i} for i in range(n_people)]
        self.n_templates = n_templates

    def render(self, person: dict, exposure_idx: int) -> str:
        return f"{person['pidx']}_{exposure_idx % self.n_templates}"


def _person_markers(rows: np.ndarray) -> set[int]:
    flat = rows.reshape(-1)
    return set((flat[(flat >= _PERSON_BASE) & (flat < _PERSON_BASE + 100)]
               - _PERSON_BASE).tolist())


def _template_markers(rows: np.ndarray) -> set[int]:
    flat = rows.reshape(-1)
    return set((flat[(flat >= _TEMPLATE_BASE) & (flat < _TEMPLATE_BASE + 100)]
               - _TEMPLATE_BASE).tolist())


# ---------------------------------------------------------------------------
# eval_person=None  ==  legacy global eval (byte-identical)
# ---------------------------------------------------------------------------

def test_build_eval_rows_none_is_byte_identical_to_legacy_global():
    sampler = _FakeSampler(n_people=5, n_templates=3)
    tok = _FakeTokenizer()

    got = build_eval_rows(
        sampler, tok, context_size=6, seed=1, eval_person=None, n_rows_global=4
    )
    # The exact construction setup() used before this flag existed.
    legacy = np.array(
        DiverseBioSubset(sampler, tok, context_size=6, seed=1)
        .to_hf_dataset(4, verbose=False)["input_ids"]
    )
    assert np.array_equal(got, legacy)


def test_build_eval_rows_none_spreads_over_many_people():
    # A global eval is NOT a single person: distinct identities show up.
    sampler = _FakeSampler(n_people=5, n_templates=3)
    tok = _FakeTokenizer()
    got = build_eval_rows(
        sampler, tok, context_size=6, seed=1, eval_person=None, n_rows_global=4
    )
    assert len(_person_markers(got)) > 1


# ---------------------------------------------------------------------------
# eval_person=k  ==  person k's bios only (all templates)
# ---------------------------------------------------------------------------

def test_single_person_subset_only_renders_person_k():
    sampler = _FakeSampler(n_people=5, n_templates=3)
    tok = _FakeTokenizer()
    k = 2
    subset = SinglePersonBioSubset(sampler, tok, person_idx=k, context_size=6, seed=1)
    rows = np.array(subset.to_hf_dataset(3, verbose=False)["input_ids"])

    assert rows.shape == (3, 6)
    assert _person_markers(rows) == {k}            # ONLY person k
    assert _template_markers(rows) == {0, 1, 2}    # rotates over ALL templates
    assert int((rows.reshape(-1) == tok.eos_token_id).sum()) > 0  # EOS-prefixed


def test_build_eval_rows_person_is_single_person():
    sampler = _FakeSampler(n_people=6, n_templates=4)
    tok = _FakeTokenizer()
    k = 3
    got = build_eval_rows(
        sampler, tok, context_size=8, seed=1, eval_person=k, n_rows_person=5
    )
    assert got.shape == (5, 8)
    assert _person_markers(got) == {k}


def test_build_eval_rows_person_default_row_count():
    sampler = _FakeSampler(n_people=4, n_templates=2)
    tok = _FakeTokenizer()
    got = build_eval_rows(sampler, tok, context_size=4, seed=0, eval_person=1)
    # "several rows" so the per-person ce_recovered is not a single noisy row.
    assert got.shape[0] == EVAL_PERSON_N_ROWS
    assert EVAL_PERSON_N_ROWS > 1


def test_single_person_out_of_range_raises():
    sampler = _FakeSampler(n_people=3, n_templates=2)
    tok = _FakeTokenizer()
    with pytest.raises(IndexError):
        SinglePersonBioSubset(sampler, tok, person_idx=3)   # == n_people
    with pytest.raises(IndexError):
        SinglePersonBioSubset(sampler, tok, person_idx=-1)


def test_single_person_shares_diverse_subset_packing_machinery():
    # Only WHICH person differs from the global path: the EOS-prefix + encode +
    # context_size packing is inherited unchanged from DiverseBioSubset.
    assert issubclass(SinglePersonBioSubset, DiverseBioSubset)
    for meth in ("_token_stream", "sequences", "to_hf_dataset", "n_tokens"):
        assert getattr(SinglePersonBioSubset, meth) is getattr(DiverseBioSubset, meth)
