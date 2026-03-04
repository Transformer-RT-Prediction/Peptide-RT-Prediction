from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

PAD = 0

PROSIT_CORE = "ACDEFGHIKLMNPQRSTVWY"
PROSIT_OX = "ACDEFGHIKLMNPQRSTVWYo"
DP_ALPHABET = "ACDEFGHIKLMNPQRSTVWY1234*"


def infer_alphabet(seqs: Iterable[str]) -> str:
    """
    Infer which alphabet to use based on the presence of
    DP-specific characters or oxidation flag.
    """
    seq_list: List[str] = [str(s) for s in seqs]
    if any(any(c in s for c in "1234*") for s in seq_list):
        return DP_ALPHABET
    if any("o" in s for s in seq_list):
        return PROSIT_OX
    return PROSIT_CORE


def build_tokenizer(alphabet: str) -> Dict[str, int]:
    tok = {c: i + 1 for i, c in enumerate(alphabet)}
    tok["[CLS]"] = len(alphabet) + 1
    return tok


def encode_sequence(seq: str, tok: Dict[str, int], max_len: int) -> np.ndarray:
    """
    Encode a peptide sequence into integer token IDs, with a [CLS] token
    at the front and padding up to max_len+1.
    """
    ids = [tok["[CLS]"]] + [tok[c] for c in str(seq) if c in tok]
    if len(ids) > max_len + 1:
        ids = ids[: max_len + 1]
    padded = ids + [PAD] * ((max_len + 1) - len(ids))
    return np.asarray(padded, dtype=np.int32)


__all__ = [
    "PAD",
    "PROSIT_CORE",
    "PROSIT_OX",
    "DP_ALPHABET",
    "infer_alphabet",
    "build_tokenizer",
    "encode_sequence",
]

