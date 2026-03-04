from __future__ import annotations

import tensorflow as tf


class StripMask(tf.keras.layers.Layer):
    """Stops Keras mask propagation to silence harmless warnings."""

    def __init__(self):
        super().__init__()
        self.supports_masking = True

    def call(self, x):
        return x

    def compute_mask(self, inputs, mask=None):
        return None


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len_with_cls, d_model, dropout):
        super().__init__()
        self.supports_masking = True
        self.pos = tf.keras.layers.Embedding(
            max_len_with_cls, d_model, name="positional_embedding"
        )
        self.do = tf.keras.layers.Dropout(dropout)

    def call(self, token_emb, training=None):
        L = tf.shape(token_emb)[1]
        pos_ids = tf.range(L)
        x = token_emb + self.pos(pos_ids)[None, ...]
        return self.do(x, training=training)

    def compute_mask(self, inputs, mask=None):
        return mask


class ConvModule(tf.keras.layers.Layer):
    """Pointwise (GLU) → DepthwiseConv1D → BN → SiLU → Pointwise."""

    def __init__(self, d_model, kernel_size=9, dropout=0.1):
        super().__init__()
        self.pw1 = tf.keras.layers.Dense(2 * d_model)
        self.dw = tf.keras.layers.DepthwiseConv1D(kernel_size, padding="same")
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.act = tf.keras.layers.Activation(tf.nn.silu)
        self.pw2 = tf.keras.layers.Dense(d_model)
        self.do = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=None):
        u, g = tf.split(self.pw1(x), 2, axis=-1)
        x = u * tf.keras.activations.sigmoid(g)
        x = self.dw(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        x = self.pw2(x)
        return self.do(x, training=training)


class GEGLUFFN(tf.keras.layers.Layer):
    """GEGLU feed-forward: Dense(2*d_ff) -> GEGLU -> Dropout -> Dense(d_model)."""

    def __init__(self, d_ff, d_model, dropout):
        super().__init__()
        self.pre = tf.keras.layers.Dense(2 * d_ff)
        self.do = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(d_model)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        x = self.norm(x)
        h = self.pre(x)
        a, b = tf.split(h, 2, axis=-1)
        geglu = tf.keras.activations.gelu(a) * b
        geglu = self.do(geglu, training=training)
        return self.proj(geglu)


class EncoderBlock(tf.keras.layers.Layer):
    """
    Conformer-style macaron block:
      0.5*FFN1 -> MHSA -> ConvModule -> 0.5*FFN2
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, conv_k):
        super().__init__()

        self.ffn1 = GEGLUFFN(d_ff=d_ff, d_model=d_model, dropout=dropout)
        self.ffn2 = GEGLUFFN(d_ff=d_ff, d_model=d_model, dropout=dropout)

        self.norm_attn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
        )
        self.do_attn = tf.keras.layers.Dropout(dropout)

        self.conv_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv = ConvModule(d_model, kernel_size=conv_k, dropout=dropout)

    def call(self, x, attn_mask, training=None):
        x = x + 0.5 * self.ffn1(x, training=training)

        y = self.mha(
            self.norm_attn(x),
            self.norm_attn(x),
            attention_mask=attn_mask,
            training=training,
        )
        x = x + self.do_attn(y, training=training)

        y = self.conv(self.conv_norm(x), training=training)
        x = x + y

        x = x + 0.5 * self.ffn2(x, training=training)

        return x


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        max_len_with_cls,
        d_model,
        d_ff,
        n_layers,
        n_heads,
        dropout,
        conv_k,
    ):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(
            vocab_size,
            d_model,
            mask_zero=True,
            name="aa_embedding",
        )
        self.pos = PositionalEmbedding(max_len_with_cls, d_model, dropout)
        self.strip = StripMask()
        self.blocks = [
            EncoderBlock(d_model, n_heads, d_ff, dropout, conv_k)
            for _ in range(n_layers)
        ]
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, token_ids, training=None):
        key_padding_mask = tf.not_equal(token_ids, 0)

        x = self.embed(token_ids)
        x = self.pos(x, training=training)
        x = self.strip(x)

        attn_mask = tf.cast(key_padding_mask[:, None, :], tf.bool)

        for blk in self.blocks:
            x = blk(x, attn_mask, training=training)

        return self.final_norm(x), key_padding_mask


class MaskedMeanMax(tf.keras.layers.Layer):
    """Pools [B,L,D] with mask [B,L] → returns [mean, max], each [B,D]."""

    def call(self, inputs):
        x, mask = inputs
        mask = tf.cast(mask, x.dtype)[:, :, None]

        sum_x = tf.reduce_sum(x * mask, axis=1)
        length = tf.reduce_sum(mask, axis=1)
        length = tf.maximum(length, tf.constant(1.0, x.dtype))
        mean = sum_x / length

        very_neg = tf.cast(-1e4, x.dtype)
        x_masked = tf.where(tf.cast(mask, tf.bool), x, very_neg)
        maxp = tf.reduce_max(x_masked, axis=1)

        return [mean, maxp]


class AttnPool(tf.keras.layers.Layer):
    """
    Single-head learned attention pooling over sequence.
    Input:  x [B,L,D], mask [B,L] (bool)
    Output: pooled [B,D]
    """

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.query = self.add_weight(
            name="attn_query",
            shape=(1, 1, d_model),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        x, mask = inputs
        B = tf.shape(x)[0]
        D = tf.shape(x)[-1]

        q = tf.cast(self.query, x.dtype)
        q = tf.tile(q, [B, 1, 1])

        scale = tf.math.sqrt(tf.cast(D, x.dtype))
        scores = tf.matmul(q, x, transpose_b=True) / scale

        mask_f = tf.cast(mask[:, None, :], x.dtype)
        scores = scores + (1.0 - mask_f) * tf.cast(-1e4, x.dtype)

        attn = tf.nn.softmax(scores, axis=-1)
        ctx = tf.matmul(attn, x)
        return ctx[:, 0, :]


__all__ = [
    "StripMask",
    "PositionalEmbedding",
    "ConvModule",
    "GEGLUFFN",
    "EncoderBlock",
    "TransformerEncoder",
    "MaskedMeanMax",
    "AttnPool",
]

