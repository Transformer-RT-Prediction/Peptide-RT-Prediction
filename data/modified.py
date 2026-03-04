from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


MOD_CHAR_POOL = "123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ*"


def build_mod_type_mapping_from_df(
    df: pd.DataFrame, mods_col: str | None
) -> Dict[Tuple[str, str | None], str]:
    """
    Scan the Modifications column and assign a unique character to each
    (mod_name, aa) combination using MOD_CHAR_POOL.
    """
    mod_type_to_char: Dict[Tuple[str, str | None], str] = {}
    if mods_col is None or mods_col not in df.columns:
        return mod_type_to_char

    char_idx = 0

    for s in df[mods_col]:
        if not isinstance(s, str):
            continue
        s_clean = s.strip()
        if not s_clean or s_clean.lower() == "unmodified":
            continue

        for _, name, inside in re.findall(r"(\\d+)x([A-Za-z]+)\\s*\\[([^\\]]+)\\]", s_clean):
            name = name.strip()
            for aa, pos in re.findall(r"([A-Z]?)(\\d+)", inside):
                aa = aa or None
                key = (name, aa)
                if key not in mod_type_to_char:
                    if char_idx < len(MOD_CHAR_POOL):
                        mod_type_to_char[key] = MOD_CHAR_POOL[char_idx]
                        char_idx += 1
                    else:
                        mod_type_to_char[key] = MOD_CHAR_POOL[-1]

    return mod_type_to_char


def encode_sequences_with_mods(
    df: pd.DataFrame,
    seq_col: str,
    mods_col: str | None,
    mod_type_to_char: Dict[Tuple[str, str | None], str],
) -> list[str]:
    """
    For each peptide, replace modified residues by specific characters
    corresponding to their (mod_name, aa) type, preserving length.
    """
    if not mod_type_to_char or mods_col is None or mods_col not in df.columns:
        return (
            df[seq_col]
            .astype(str)
            .str.strip()
            .str.replace(r"\\s+", "", regex=True)
            .tolist()
        )

    new_seqs: list[str] = []

    for seq, mods_str in zip(df[seq_col], df[mods_col]):
        seq = str(seq).strip().replace(" ", "")
        L = len(seq)
        pos_mod_char = [None] * L

        if isinstance(mods_str, str):
            s_clean = mods_str.strip()
            if s_clean and s_clean.lower() != "unmodified":
                for _, name, inside in re.findall(
                    r"(\\d+)x([A-Za-z]+)\\s*\\[([^\\]]+)\\]", s_clean
                ):
                    name = name.strip()
                    for aa, pos in re.findall(r"([A-Z]?)(\\d+)", inside):
                        aa = aa or None
                        pos_idx = int(pos) - 1
                        if pos_idx < 0 or pos_idx >= L:
                            continue
                        key = (name, aa)
                        ch = mod_type_to_char.get(key)
                        if ch is None:
                            ch = MOD_CHAR_POOL[-1]
                        pos_mod_char[pos_idx] = ch

        chars: list[str] = []
        for aa, mch in zip(seq, pos_mod_char):
            if mch is None:
                chars.append(aa)
            else:
                chars.append(mch)

        new_seqs.append("".join(chars))

    return new_seqs


def load_tsv_modified(path: Path) -> pd.DataFrame:
    """
    Unified loader for CSV/TSV files with columns like:
      - Sequence, Modifications, RT
    """
    df = pd.read_csv(path, sep=None, engine="python")
    cols_lower = {c.lower(): c for c in df.columns}

    seq_col = None
    for key in ["sequence", "peptide sequence", "peptide"]:
        for lc, orig in cols_lower.items():
            if key == lc or key in lc:
                seq_col = orig
                break
        if seq_col:
            break

    rt_col = None
    for key in ["rt", "retention time", "retention_time", "tr"]:
        for lc, orig in cols_lower.items():
            if key == lc or key in lc:
                rt_col = orig
                break
        if rt_col:
            break

    mods_col = None
    for lc, orig in cols_lower.items():
        if "mod" in lc:
            mods_col = orig
            break

    if seq_col is None or rt_col is None:
        raise ValueError(
            f"sequence/rt columns not found in {path}. Columns: {list(df.columns)}"
        )

    mod_mapping = build_mod_type_mapping_from_df(df, mods_col)
    if mod_mapping:
        print(
            f"[Data] Detected {len(mod_mapping)} modification types in "
            f"{os.path.basename(path)}"
        )

    seqs_encoded = encode_sequences_with_mods(df, seq_col, mods_col, mod_mapping)

    df_out = pd.DataFrame(
        {
            "sequence": seqs_encoded,
            "rt": pd.to_numeric(df[rt_col], errors="coerce"),
        }
    )

    df_out["sequence"] = (
        df_out["sequence"]
        .astype(str)
        .str.strip()
        .str.replace(r"\\s+", "", regex=True)
    )
    df_out = df_out.dropna(subset=["rt"])
    df_out = df_out[df_out["sequence"].str.len() > 0].reset_index(drop=True)

    return df_out


__all__ = [
    "MOD_CHAR_POOL",
    "build_mod_type_mapping_from_df",
    "encode_sequences_with_mods",
    "load_tsv_modified",
]

