"""Span-the-bio windowing (deviation S2) + one-bio-per-row corpus.

Each bio yields ONE window per token position: the window ending at token t holds
the tau+1 activations [t-tau .. t], zero-left-padded where the lookback falls before
the bio start, then transposed to feature-first [x_dim, tau+1] (last col = current t).
Mirrors reference utils.py:gen_window_slicing_batch (window=tau+1, last=current) but
stays within a single bio and zero-pads the start instead of crossing a boundary.
"""
from __future__ import annotations

import random

import torch


def span_windows(acts: torch.Tensor, valid_len: int, tau: int) -> torch.Tensor:
    """acts: [T, x_dim]. Returns [valid_len, x_dim, tau+1]."""
    x_dim = acts.shape[1]
    W = tau + 1
    pad = acts.new_zeros((tau, x_dim))
    padded = torch.cat([pad, acts[:valid_len]], dim=0)            # [tau+valid_len, x_dim]
    idx = torch.arange(valid_len).unsqueeze(1) + torch.arange(W).unsqueeze(0)  # [valid_len, W]
    win = padded[idx]                                            # [valid_len, W, x_dim]
    return win.transpose(1, 2)                                   # [valid_len, x_dim, W]


def windows_for_batch(acts: torch.Tensor, valid_lens: torch.Tensor, tau: int) -> torch.Tensor:
    """acts: [n_bios, T, x_dim]. Returns [total_windows, x_dim, tau+1]."""
    out = [span_windows(acts[b], int(valid_lens[b]), tau) for b in range(acts.shape[0])]
    return torch.cat(out, dim=0)


def pack_bios(token_lists, max_bio_len: int, bos_id: int):
    """[[ids]...] -> ([n, max_bio_len] long, [n] valid_len). Each row = [bos]+ids,
    truncated to max_bio_len, right-padded with bos_id."""
    n = len(token_lists)
    tokens = torch.full((n, max_bio_len), bos_id, dtype=torch.long)
    valid_len = torch.zeros(n, dtype=torch.long)
    for i, toks in enumerate(token_lists):
        row = ([bos_id] + list(toks))[:max_bio_len]
        tokens[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        valid_len[i] = len(row)
    return tokens, valid_len


def build_bio_corpus(sampler, tokenizer, n_bios: int, max_bio_len: int, seed: int):
    rng = random.Random(seed)
    token_lists = [tokenizer.encode(sampler.sample(rng)["text"]) for _ in range(n_bios)]
    return pack_bios(token_lists, max_bio_len, tokenizer.bos_token_id)
