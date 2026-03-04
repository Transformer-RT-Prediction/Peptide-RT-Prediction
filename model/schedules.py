from __future__ import annotations

import numpy as np
import tensorflow as tf


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Warmup + cosine decay learning rate schedule.
    """

    def __init__(self, base_lr, warmup_steps, min_lr=1e-5, total_steps=200_000):
        super().__init__()
        self.base_lr = tf.cast(base_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)
        self.total = tf.cast(total_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        warm = self.base_lr * tf.minimum(1.0, step / tf.maximum(1.0, self.warmup_steps))

        progress = tf.clip_by_value(
            (step - self.warmup_steps) / tf.maximum(1.0, self.total - self.warmup_steps),
            0.0,
            1.0,
        )
        cosine = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(np.pi * progress)
        )

        return tf.where(step < self.warmup_steps, warm, cosine)


__all__ = ["WarmupCosine"]

