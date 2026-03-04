from __future__ import annotations

"""
Model subpackage for the Conformer-like RT prediction network.

This package is split into:
- schedules: learning-rate schedules
- layers: encoder blocks and pooling layers
- builder: high-level Keras Model factory
- utils: small model-related utilities (e.g. seeding)
"""

from .schedules import WarmupCosine
from .layers import (
    StripMask,
    PositionalEmbedding,
    ConvModule,
    GEGLUFFN,
    EncoderBlock,
    TransformerEncoder,
    MaskedMeanMax,
    AttnPool,
)
from .builder import build_model_from_hp
from .utils import set_seed

__all__ = [
    "WarmupCosine",
    "StripMask",
    "PositionalEmbedding",
    "ConvModule",
    "GEGLUFFN",
    "EncoderBlock",
    "TransformerEncoder",
    "MaskedMeanMax",
    "AttnPool",
    "build_model_from_hp",
    "set_seed",
]

