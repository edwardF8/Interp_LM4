"""Tokenizer shim for the reduced-vocab Capo (bioS) models.

The trained models use a vocab that is a *subset* of GPT-2's: only the
~1.8k token ids that appeared anywhere in the training bios are kept,
remapped to a contiguous [0, reducedVocabSize) range. The mapping is
arbitrary (sorted-unique enumeration), so the model can't be used at all
without it.

CondensedTokenizer composes (GPT2Tokenizer, old_to_new, new_to_old) into
a single object that quacks like a HF tokenizer for the purposes TL uses:
encode, decode, __call__, bos/eos/pad ids, vocab_size.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import GPT2Tokenizer


def load_remap(remap_path: str | Path) -> tuple[dict[int, int], dict[int, int]]:
    """Load the old_to_new.json saved by Training_On_LM4/main.py.

    The file stores {str(gpt2_id): reduced_id}; JSON forces string keys, so
    we cast back to int here. Returns both directions.
    """
    with open(remap_path) as f:
        raw = json.load(f)
    old_to_new = {int(k): int(v) for k, v in raw.items()}
    new_to_old = {v: k for k, v in old_to_new.items()}
    return old_to_new, new_to_old


class CondensedTokenizer:
    def __init__(
        self,
        gpt2_tokenizer: GPT2Tokenizer,
        old_to_new: dict[int, int],
        new_to_old: dict[int, int],
    ):
        self.gpt2 = gpt2_tokenizer
        self.old_to_new = old_to_new
        self.new_to_old = new_to_old

        self.vocab_size = len(old_to_new)
        self.eos_token_id = old_to_new[int(gpt2_tokenizer.eos_token_id)]
        self.bos_token_id = self.eos_token_id
        self.pad_token_id = self.eos_token_id

        self.eos_token = gpt2_tokenizer.eos_token
        self.bos_token = gpt2_tokenizer.eos_token
        self.pad_token = gpt2_tokenizer.eos_token
        self.unk_token = gpt2_tokenizer.eos_token
        self.unk_token_id = self.eos_token_id

        # TL touches these in to_tokens / to_string / __init__; default them
        # to what GPT-2 uses.
        self.padding_side = "right"
        self.truncation_side = "right"
        self.model_max_length = gpt2_tokenizer.model_max_length
        self.add_bos_token = False
        self.name_or_path = "condensed-gpt2"

    @classmethod
    def from_remap_path(cls, remap_path: str | Path) -> "CondensedTokenizer":
        old_to_new, new_to_old = load_remap(remap_path)
        return cls(GPT2Tokenizer.from_pretrained("gpt2"), old_to_new, new_to_old)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        raw = self.gpt2(text, add_special_tokens=add_special_tokens)["input_ids"]
        try:
            return [self.old_to_new[int(t)] for t in raw]
        except KeyError as e:
            raise KeyError(
                f"GPT-2 token id {e.args[0]} not in reduced vocab — this text "
                "contains a token the model wasn't trained on."
            ) from e

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        gpt2_ids = [self.new_to_old[int(i)] for i in ids]
        return self.gpt2.decode(gpt2_ids, skip_special_tokens=skip_special_tokens)

    def __call__(
        self,
        text: str | list[str],
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
        padding: bool | str = False,
    ) -> dict:
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        batch = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

        if padding:
            max_len = max(len(ids) for ids in batch)
            attn = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch]
            batch = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in batch]
        else:
            attn = [[1] * len(ids) for ids in batch]

        if return_tensors == "pt":
            input_ids = torch.tensor(batch, dtype=torch.long)
            attention_mask = torch.tensor(attn, dtype=torch.long)
            if single and not padding:
                input_ids = input_ids[0]
                attention_mask = attention_mask[0]
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        if single:
            return {"input_ids": batch[0], "attention_mask": attn[0]}
        return {"input_ids": batch, "attention_mask": attn}

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]

    def __len__(self) -> int:
        return self.vocab_size
