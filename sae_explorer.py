"""Local Neuronpedia-style exploration for the custom SAE.

Three public functions:

  build_index_corpus  - assemble + cache the tokenized bio corpus the
                        dashboard indexes against.
  make_dashboard      - wrap sae_dashboard.SaeVisRunner; write per-feature
                        HTML panels + an index page. (Task 4.)
  steer               - causal probe: boost a feature's decoder direction
                        at a hook, report logit deltas. (Task 5.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch


def build_index_corpus(
    sampler,
    tokenizer,
    n_per_person: int = 2,
    context_size: int = 64,
    seed: int = 0,
    people: Sequence[dict] | None = None,
    cache_path: str | Path | None = None,
) -> torch.Tensor:
    """Build the [N, T] long tensor of token ids the dashboard indexes against.

    For each person in `people` (default: sampler.people), draw `n_per_person`
    distinct template indices, render each one, tokenize with the condensed
    tokenizer, prepend the EOS token, and pad/truncate to `context_size`.

    If `cache_path` is given and exists, load + return it without recomputing.
    On a miss, save the result there.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    if people is None:
        people = sampler.people

    rng = torch.Generator().manual_seed(seed)
    n_templates = sampler.n_templates
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    rows: list[list[int]] = []
    for person in people:
        # distinct templates per person (or all of them if n_per_person >= n_templates)
        k = min(n_per_person, n_templates)
        perm = torch.randperm(n_templates, generator=rng).tolist()
        for exposure_idx in perm[:k]:
            text = sampler.render(person, exposure_idx)
            ids = [eos] + tokenizer.encode(text)
            if len(ids) >= context_size:
                ids = ids[:context_size]
            else:
                ids = ids + [pad] * (context_size - len(ids))
            rows.append(ids)

    tokens = torch.tensor(rows, dtype=torch.long)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokens, cache_path)

    return tokens
