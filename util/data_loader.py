"""DataLoader for the packed reduced-vocab token files.

Mirrors PackedTokenDataset from Training_On_LM4/data/tokenize_pack.py so
the interp notebook can read bios_postreduce.bin directly without pulling
in the training package.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PackedTokenDataset(Dataset):
    """uint16 memmap of reduced-vocab ids → (input_ids, labels) chunks.

    labels == input_ids; the causal-LM shift is handled inside the model's
    loss. Returns int64 because nn.Embedding indices must be int64.
    """

    def __init__(self, path: str | Path, seq_len: int = 512):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_seq = len(self.tokens) // seq_len

    def __len__(self) -> int:
        return self.n_seq

    def __getitem__(self, idx: int) -> dict:
        chunk = self.tokens[idx * self.seq_len : (idx + 1) * self.seq_len]
        ids = torch.from_numpy(chunk.astype(np.int64))
        return {"input_ids": ids, "labels": ids.clone()}


def make_dataloader(
    token_path: str | Path,
    seq_len: int = 512,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    ds = PackedTokenDataset(token_path, seq_len=seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
