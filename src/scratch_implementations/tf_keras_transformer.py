import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class CausalSelfAttentionTF(layers.Layer):
    def __init__(self, d_model=768, n_head=12, max_len=1024, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_head = n_head
        
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5

        # 1. PROJECTION (General -> Q, K, V)
        # Project d_model (768) -> 3 * d_model (2304).
        self.c_attn = layers.Dense(3 * d_model, name="c_attn")
        
        # 2. OUTPUT PROJECTION (Heads -> General)
        # Project d_model (768) -> d_model (768).
        self.c_proj = layers.Dense(d_model, name="c_proj")

        # 3. CAUSAL MASK
        mask = 1 - tf.linalg.band_part(tf.ones((max_len, max_len)), -1, 0)
        self.bias = tf.reshape(mask, (1, 1, max_len, max_len)) # for broatcasting 