from __future__ import annotations

import random as _random

import numpy as _np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """
    Set global random seed for TensorFlow, NumPy and Python's random.
    """
    tf.keras.utils.set_random_seed(seed)
    _np.random.seed(seed)
    _random.seed(seed)


__all__ = ["set_seed"]

