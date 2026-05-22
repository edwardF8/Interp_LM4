"""Identity-diverse token subset for SAE training.

Slicing the packed `bios_postreduce.bin` gives no control over *which* of the
50,000 identities land in a given 512-token row. This module instead
regenerates bios directly from `people.json` (via `BioSampler`) and
`CondensedTokenizer`, so a small subset is maximally diverse by construction:

  * `person` advances every bio   -> every bio is a distinct identity until all
    50,000 are used, then it cycles with fresh (person, template) pairs;
  * `template` advances every bio -> any prefix also spreads evenly over all
    46 paraphrase templates.

Bios are EOS-*prefixed* and packed into `context_size` sequences exactly the
way `tokenize_and_pack` built the training `.bin` (it prefixes each bio with
`<|endoftext|>`), then exposed as a HuggingFace `Dataset` for SAE Lens's
`override_dataset`.

Usage
-----
    sampler   = BioSampler(DATA_DIR / "people.json", fields=("birthday",), seed=0)
    tokenizer = CondensedTokenizer.from_remap_path(REMAP_PATH)
    subset    = DiverseBioSubset(sampler, tokenizer, context_size=512, seed=0)

    sae_dataset = subset.to_hf_dataset(n_examples=10_000)   # 10k x 512 tokens
"""
from __future__ import annotations

import random

import numpy as np
from datasets import Dataset


class DiverseBioSubset:
    """Generates a fixed number of `context_size` token sequences, packed from
    bios chosen to maximise identity + template diversity.

    `sampler` / `tokenizer` are injected (BioSampler, CondensedTokenizer) so
    this class stays decoupled from how they're built.
    """

    def __init__(self, sampler, tokenizer, context_size: int = 512, seed: int = 0):
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.eos = tokenizer.eos_token_id

        self.n_people = len(sampler.people)
        self.n_templates = sampler.n_templates

        # Shuffle both axes once (seeded) so the subset isn't biased by the
        # order of people.json / the template list. person and template then
        # each advance by one per bio -> any prefix is diverse on both axes.
        rng = random.Random(seed)
        self._people_perm = list(range(self.n_people))
        self._template_perm = list(range(self.n_templates))
        rng.shuffle(self._people_perm)
        rng.shuffle(self._template_perm)

    def _bio_text(self, i: int) -> str:
        """The i-th bio in diverse order: distinct person, rotating template."""
        person = self.sampler.people[self._people_perm[i % self.n_people]]
        template = self._template_perm[i % self.n_templates]
        return self.sampler.render(person, template)

    def _token_stream(self):
        """Infinite token stream: each bio EOS-prefixed, in diverse order
        (mirrors tokenize_and_pack, which prefixes each bio with <|endoftext|>)."""
        i = 0
        while True:
            yield self.eos
            yield from self.tokenizer.encode(self._bio_text(i))
            i += 1

    def sequences(self, n_examples: int):
        """Generator: yield exactly `n_examples` sequences of `context_size`
        token ids each (each a plain list[int])."""
        stream = self._token_stream()
        buf: list[int] = []
        for _ in range(n_examples):
            while len(buf) < self.context_size:
                buf.append(next(stream))
            yield buf[: self.context_size]
            buf = buf[self.context_size :]

    def n_tokens(self, n_examples: int) -> int:
        """Token count of an `n_examples` subset — use to size `training_tokens`."""
        return n_examples * self.context_size

    def to_hf_dataset(self, n_examples: int, verbose: bool = True) -> Dataset:
        """Materialise `n_examples` sequences as a HF `Dataset` with an
        `input_ids` column — ready to pass as SAE Lens `override_dataset`."""
        rows = np.array(list(self.sequences(n_examples)), dtype=np.int32)
        ds = Dataset.from_dict({"input_ids": rows})
        if verbose:
            n_bios = int((rows == self.eos).sum())   # one EOS prefixes each bio
            print(f"DiverseBioSubset: {n_examples:,} x {self.context_size} "
                  f"= {rows.size:,} tokens, ~{n_bios:,} bios")
            print(f"  distinct identities: {min(n_bios, self.n_people):,} / "
                  f"{self.n_people:,}   templates: "
                  f"{min(n_bios, self.n_templates)} / {self.n_templates}   "
                  f"id_range=[{rows.min()}, {rows.max()}]")
        return ds
