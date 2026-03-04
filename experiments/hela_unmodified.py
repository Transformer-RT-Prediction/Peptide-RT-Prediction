from __future__ import annotations

from pathlib import Path

from rt_pred.config import hela_unmodified_config
from rt_pred.data import load_tsv_unmodified
from rt_pred.training import run_experiment


def main(root: str | Path = "/home/eemslab/rt_pred_unmodified/exp_hela/"):
    """
    Entry point for the unmodified Hela experiment.
    Mirrors the behavior of `rt_pred_exp_hela1 (2).ipynb`.
    """
    cfg = hela_unmodified_config(root)
    run_experiment(cfg, load_df=load_tsv_unmodified)


if __name__ == "__main__":
    main()

