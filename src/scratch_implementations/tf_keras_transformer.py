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
        self.bias = tf.reshape(mask, (1, 1, max_len, max_len)) # for broadcasting


    def split_heads(self, x, batch_size):
        # Input 'x' shape is (Batch, Seq_Len, d_model)

        # Reshape to (Batch, Seq, Heads, Head_Dim)
        x = tf.reshape(x, (batch_size, -1, self.n_head, self.head_dim))
        
        # Transpose to (Batch, Heads, Seq, Head_Dim)
        return tf.transpose(x, perm=[0, 2, 1, 3])



    def call(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        # Seq_Len can be 4, 10, or 1024 (dynamic)

        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # 1. CALCULATE Q, K, V
        # ---------------------------------------------------------
        # Run Linear Layer -> (Batch, Seq, 3 * d_model)
        qkv = self.c_attn(x)
        
        # Split into 3 tensors: Q, K, V
        # Each is (Batch, Seq, d_model)
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1) 

        # 2. SPLIT HEADS
        # ---------------------------------------------------------
        # Transform (Batch, Seq, d_model) -> (Batch, Heads, Seq, Head_Dim)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 3. ATTENTION SCORES (The Dot Product)
        # ---------------------------------------------------------
        # Equation: Q @ K_Transpose
        # Shapes: (Batch, Head, Seq, Head_Dim) @ (Batch, Head, Head_Dim, Seq) -> (Batch, Head, Seq, Seq) 
        att = tf.matmul(q, k, transpose_b=True) * self.scale

        # 4. CAUSAL MASKING
        # ---------------------------------------------------------
        current_mask = self.bias[:, :, :seq_len, :seq_len]
        # Add -1e9 to future positions so Softmax crushes them to zero.
        att += (current_mask * -1e9)

        # 5. SOFTMAX & AGGREGATE
        # ---------------------------------------------------------
        # Normalize rows to sum to 1.0
        att = tf.nn.softmax(att, axis=-1)

        # Weighted Sum of Values
        # (Batch, Head, Seq, Seq) @ (Batch, Head, Seq, Head_Dim) -> (B, h, Seq, Head_Dim)
        y = tf.matmul(att, v)

        # 6. REASSEMBLE
        # ---------------------------------------------------------
        # Transpose back: (Batch, Head, Seq, Head_Dim) -> (Batch, Seq, Head, Head_Dim)
        # We move the 'Heads' dimension next to 'Head_Size' so we can merge them.
        y = tf.transpose(y, perm=[0, 2, 1, 3])
        
        # Reshape: (Batch, Seq, Head * Head_Dim) -> (Batch, Seq, d_model)
        y = tf.reshape(y, (batch_size, seq_len, self.d_model))

        # 7. OUTPUT PROJECTION
        # Mixes insights from different heads together.
        return self.c_proj(y)


class MLPTF(layers.Layer):
    def __init__(self, d_model=768, **kwargs):
        super().__init__(**kwargs)
        
        self.d_ff = 4 * d_model
        
        # 1. Expand (Up-Project)
        # Projects from Small (768) -> Big (3072)
        self.c_fc = layers.Dense(self.d_ff, name="c_fc")
        
        # 2. Contract (Down-Project)
        # Projects from Big (3072) -> Small (768)
        self.c_proj = layers.Dense(d_model, name="c_proj")

    def call(self, x):
        
        # LINEAR EXPANSION (Up-Project)
        # x shape: (Batch, Seq, d_model) -> (Batch, Seq, d_model*4)
        x = self.c_fc(x)
        
        # NON-LINEARITY
        x = tf.nn.gelu(x, approximate=True)
        
        # LINEAR CONTRACTION (Down-Project)
        # x shape: (Batch, Seq, d_model*4) -> (Batch, Seq, d_model) 
        x = self.c_proj(x)
        
        return x


class TransformerBlockTF(layers.Layer):
 
    def __init__(self, d_model=768, n_head=12, **kwargs):
        super().__init__(**kwargs)
        self.attn = CausalSelfAttentionTF(d_model, n_head)
        self.mlp = MLPTF(d_model)

        self.ln1 = layers.LayerNormalization(epsilon=1e-5, name="ln_1")
        self.ln2 = layers.LayerNormalization(epsilon=1e-5, name="ln_2")

    def call(self, x):

        # 1. Attention (PRE-NORM instead of POST-NORM; to preserve the gradient highway through the skip connection)
        input_copy = x              
        x_norm = self.ln1(x)         
        attn_out = self.attn(x_norm) 
        x = input_copy + attn_out    

        # 2. MLP (PRE-NORM instead of POST-NORM)
        input_copy = x              
        x_norm = self.ln2(x)         
        mlp_out = self.mlp(x_norm)   
        x = input_copy + mlp_out    
        
        return x