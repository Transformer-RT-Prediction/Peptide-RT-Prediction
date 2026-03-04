from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class HyperParams:
    name: str
    D_MODEL: int
    N_LAYERS: int
    N_HEADS: int
    D_FF: int
    DROPOUT: float
    BASE_LR: float


DEFAULT_HP_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "small_d192_l8",
        "D_MODEL": 192,
        "N_LAYERS": 8,
        "N_HEADS": 4,
        "D_FF": 768,
        "DROPOUT": 0.10,
        "BASE_LR": 2e-3,
    },
    {
        "name": "baseline_d256_l8",
        "D_MODEL": 256,
        "N_LAYERS": 8,
        "N_HEADS": 8,
        "D_FF": 1024,
        "DROPOUT": 0.10,
        "BASE_LR": 2e-3,
    },
    {
        "name": "deep_d256_l12",
        "D_MODEL": 256,
        "N_LAYERS": 12,
        "N_HEADS": 8,
        "D_FF": 1024,
        "DROPOUT": 0.15,
        "BASE_LR": 1.5e-3,
    },
    {
        "name": "wide_d320_l12",
        "D_MODEL": 320,
        "N_LAYERS": 12,
        "N_HEADS": 8,
        "D_FF": 1280,
        "DROPOUT": 0.15,
        "BASE_LR": 1.5e-3,
    },
]


@dataclass
class ExperimentConfig:
    """
    High-level configuration for an RT prediction experiment.
    """

    root: Path
    files: List[str]

    unit_mode: str = "auto"
    sec_to_min_threshold: float = 600.0

    epochs: int = 500
    batch: int = 256
    d_model: int = 256
    n_layers: int = 16
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.10
    conv_k: int = 9
    huber_delta: float = 1.0
    weight_decay: float = 1e-4
    warmup_steps: int = 4000
    min_lr: float = 1e-5
    base_lr: float = 2e-3

    hp_search: bool = True
    epochs_tune: int = 500
    hp_configs: List[Dict[str, Any]] = field(
        default_factory=lambda: list(DEFAULT_HP_CONFIGS)
    )


def hela_unmodified_config(root: str | Path) -> ExperimentConfig:
    return ExperimentConfig(
        root=Path(root),
        files=["hela_unmodified.txt"],
    )


def hela_modified_config(root: str | Path) -> ExperimentConfig:
    return ExperimentConfig(
        root=Path(root),
        files=["mod_hela_exp.csv"],
    )


__all__ = [
    "HyperParams",
    "ExperimentConfig",
    "DEFAULT_HP_CONFIGS",
    "hela_unmodified_config",
    "hela_modified_config",
]

