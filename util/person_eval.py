"""Single-person eval subset for LOCAL per-person CLT evaluation.

The global CLT eval (``DiverseBioSubset`` over all ~50k people, 64 packed rows)
barely moves when a MEMIT birthday edit rewrites ONE person out of 50k, so an
edit-CLT fine-tune's parity/plateau early-stop -- which is driven by
``ce_recovered`` -- fires almost immediately and under-fits the edited person.
This module builds the eval tokens from a SINGLE person's bios instead, rotated
over all birthday templates and packed EXACTLY the way ``DiverseBioSubset``
packs the global eval, so ``ce_recovered`` becomes a LOCAL signal.

Faithfulness (repo invariants):
  * Bio text comes from the SAME ``sampler.render`` (Training_On_LM4's
    ``render_bio``, byte-identical to the training corpus) and the SAME
    ``tokenizer.encode`` that ``DiverseBioSubset`` uses -- only WHICH person
    differs. Nothing is reimplemented here.
  * A real person's rendered birthday bio is in the condensed vocab by
    construction (it was in the corpus the tokenizer was built from), so
    ``encode`` won't ``KeyError``. ``person_idx`` is bounds-checked so an
    out-of-range index fails loudly rather than silently mis-rendering.
"""
from __future__ import annotations

import numpy as np

from util.diverse_subset import DiverseBioSubset

# Rows of ``context_size`` tokens rendered for the per-person eval. One row is a
# noisy ``ce_recovered`` estimate; a handful of rows (each packing the person's
# bios across all templates, cycling as needed) makes it stable without
# materially slowing the periodic eval. At context_size=512 this is ~4k
# next-token positions over the one person's facts.
EVAL_PERSON_N_ROWS = 8


class SinglePersonBioSubset(DiverseBioSubset):
    """``DiverseBioSubset`` variant where every bio is ``person_idx``, rotating
    through all templates. Reuses the parent's EOS-prefix + encode + packing
    unchanged (only ``_bio_text`` is overridden), so the token layout matches
    the global eval exactly -- only which identity appears differs."""

    def __init__(self, sampler, tokenizer, person_idx: int,
                 context_size: int = 512, seed: int = 0):
        super().__init__(sampler, tokenizer, context_size=context_size, seed=seed)
        if not (0 <= person_idx < self.n_people):
            raise IndexError(
                f"eval_person {person_idx} out of range [0, {self.n_people})"
            )
        self.person_idx = person_idx

    def _bio_text(self, i: int) -> str:
        """The i-th bio: always ``person_idx``, template cycling 0..n_templates-1
        so every paraphrase is exercised regardless of seed."""
        person = self.sampler.people[self.person_idx]
        template = i % self.n_templates
        return self.sampler.render(person, template)


def build_eval_rows(sampler, tokenizer, *, context_size: int, seed: int,
                    eval_person: int | None = None,
                    n_rows_global: int = 64,
                    n_rows_person: int = EVAL_PERSON_N_ROWS) -> np.ndarray:
    """Eval token rows ``[n_rows, context_size]`` for the CLT periodic/final eval.

    ``eval_person is None`` -> global held-out ``DiverseBioSubset`` (byte-identical
    to the legacy path: ``n_rows_global`` packed rows at ``seed``). ``eval_person=k``
    -> person k's bios via ``SinglePersonBioSubset`` (``n_rows_person`` rows).
    ``trainCLT.setup()`` wraps the result in a long tensor on-device; keeping this
    device-agnostic (numpy) makes it CPU-testable with fakes.
    """
    if eval_person is None:
        subset = DiverseBioSubset(
            sampler, tokenizer, context_size=context_size, seed=seed
        )
        n_rows = n_rows_global
    else:
        subset = SinglePersonBioSubset(
            sampler, tokenizer, eval_person, context_size=context_size, seed=seed
        )
        n_rows = n_rows_person
    return np.array(subset.to_hf_dataset(n_rows, verbose=False)["input_ids"])
