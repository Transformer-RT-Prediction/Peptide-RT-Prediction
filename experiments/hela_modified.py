from __future__ import annotations

from pathlib import Path

from rt_pred.config import hela_modified_config
from rt_pred.data import load_tsv_modified
from rt_pred.training import run_experiment


def main(root: str | Path = "/home/eemslab/rt_pred_modified/exp_hela/"):
    """
    Entry point for the modified Hela experiment.
    Mirrors the behavior of `rt_pred_exp_mod_hela1.ipynb`.
    """
    cfg = hela_modified_config(root)
    run_experiment(cfg, load_df=load_tsv_modified)


if __name__ == "__main__":
    main()

