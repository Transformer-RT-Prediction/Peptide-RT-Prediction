from __future__ import annotations

"""
Data loading and preprocessing utilities for RT prediction.

- unmodified: TSV loader for unmodified Hela data
- modified: CSV/TSV loader with modification-aware encoding
"""

from .unmodified import load_tsv_unmodified
from .modified import (
    MOD_CHAR_POOL,
    build_mod_type_mapping_from_df,
    encode_sequences_with_mods,
    load_tsv_modified,
)

__all__ = [
    "load_tsv_unmodified",
    "MOD_CHAR_POOL",
    "build_mod_type_mapping_from_df",
    "encode_sequences_with_mods",
    "load_tsv_modified",
]

