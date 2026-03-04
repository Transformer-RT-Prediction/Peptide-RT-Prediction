from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_tsv_unmodified(path: Path) -> pd.DataFrame:
    """
    Loader for the unmodified Hela text file.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}
    seq_col = cols.get("sequence")
    rt_col = cols.get("rt")

    if seq_col is None or rt_col is None:
        raise ValueError(
            f"sequence/rt columns not found in {path}. Columns: {list(df.columns)}"
        )

    df = df[[seq_col, rt_col]].rename(columns={seq_col: "sequence", rt_col: "rt"})
    df["sequence"] = (
        df["sequence"].astype(str).str.strip().str.replace(r"\\s+", "", regex=True)
    )
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


__all__ = ["load_tsv_unmodified"]

