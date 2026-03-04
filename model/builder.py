from __future__ import annotations

import tensorflow as tf

from ..config import ExperimentConfig
from .layers import TransformerEncoder, MaskedMeanMax, AttnPool
from .schedules import WarmupCosine


def build_model_from_hp(
    hp: dict,
    max_len_with_cls: int,
    vocab_size: int,
    steps_per_epoch: int,
    epochs: int,
    cfg: ExperimentConfig,
) -> tf.keras.Model:
    """
    Build a Conformer-lite model using a hyperparameter dict `hp`.
    """
    d_model = hp["D_MODEL"]
    n_layers = hp["N_LAYERS"]
    n_heads = hp["N_HEADS"]
    d_ff = hp["D_FF"]
    dropout = hp.get("DROPOUT", cfg.dropout)
    base_lr = hp.get("BASE_LR", cfg.base_lr)

    inp = tf.keras.Input(
        shape=(max_len_with_cls,),
        dtype=tf.int32,
        name="tokens",
    )

    enc = TransformerEncoder(
        vocab_size=vocab_size,
        max_len_with_cls=max_len_with_cls,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        conv_k=cfg.conv_k,
    )

    x, key_mask = enc(inp)

    cls_tok = x[:, 0, :]
    mean_p, max_p = MaskedMeanMax()([x, key_mask])

    attn_pool = AttnPool(d_model=d_model, name="attn_pool")
    attn_vec = attn_pool([x, key_mask])

    feat = tf.keras.layers.Concatenate()([cls_tok, mean_p, max_p, attn_vec])

    feat = tf.keras.layers.LayerNormalization(epsilon=1e-6)(feat)
    feat = tf.keras.layers.Dropout(dropout)(feat)
    feat = tf.keras.layers.Dense(
        d_model * 2,
        activation=tf.keras.activations.gelu,
    )(feat)
    feat = tf.keras.layers.Dropout(dropout)(feat)

    out = tf.keras.layers.Dense(
        1,
        activation="linear",
        dtype="float32",
    )(feat)

    model = tf.keras.Model(inp, out, name=f"rt_conformer_lite_{hp.get('name','hp')}")

    total_steps = max(1, steps_per_epoch * epochs)
    lr_sched = WarmupCosine(
        base_lr,
        warmup_steps=cfg.warmup_steps,
        min_lr=cfg.min_lr,
        total_steps=total_steps,
    )

    opt = tf.keras.optimizers.AdamW(
        learning_rate=lr_sched,
        weight_decay=cfg.weight_decay,
        epsilon=1e-8,
        global_clipnorm=1.0,
    )

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.Huber(delta=cfg.huber_delta),
    )

    return model


__all__ = ["build_model_from_hp"]

