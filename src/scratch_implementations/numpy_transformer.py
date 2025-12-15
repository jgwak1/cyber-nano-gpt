import numpy as np

class CausalSelfAttentionNumPy:
    def __init__(self, d_model=512, n_head=8, max_len=512):
        self.d_model = d_model 
        self.n_head = n_head   
        self.head_dim = d_model // n_head 
        self.scale = self.head_dim ** -0.5 

        # 1. Attention Weights (Merged Q, K, V)
        # Shape: [Input_Dim, Output_Dim] -> [d_model, 3 * d_model]
        self.c_attn = np.random.normal(scale=0.02, size=(d_model, 3 * d_model)) 
        self.b_attn = np.zeros(3 * d_model)

        # 2. Output Projection Weights
        # Projects FROM "Concatenated Heads" TO "Residual Stream".
        # Input Dim:  n_head * head_dim
        # Output Dim: d_model
        # Shape:      [n_head * head_dim, d_model]
        self.c_proj = np.random.normal(scale=0.02, size=(n_head * self.head_dim, d_model)) 
        self.b_proj = np.zeros(d_model)

    def softmax(self, x):
        # Numerical stability: subtract max to prevent overflow (e^x -> Inf)
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, x):
        B, T, C = x.shape 
        
        # 1. LINEAR PROJECTION
        # x: [B, T, C] @ c_attn: [C, 3*C] -> [B, T, 3*C]
        qkv = x @ self.c_attn + self.b_attn
        
        # 2. SPLIT Q, K, V 
        # Split last dim into 3 parts. Result: 3 arrays of [B, T, C]
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # 3. RESHAPE & TRANSPOSE
        # Reshape: [B, T, n_head * head_dim] -> [B, T, n_head, head_dim]
        # Transpose: Swap Time(1) and Head(2) -> [B, n_head, T, head_dim]
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        # 4. SCALED DOT-PRODUCT ATTENTION
        # (B, h, T, hs) @ (B, h, hs, T) -> (B, h, T, T)
        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        # 5. CAUSAL MASKING
        # Set upper triangle (future) to -inf. Softmax(-inf) -> 0.0
        mask = np.tril(np.ones((T, T)))
        att = np.where(mask == 0, -1e9, att)
        att = self.softmax(att)
        
        # 6. AGGREGATE for Attention Outputs
        # Weighted sum of values.
        # (B, h, T, T) @ (B, h, T, hs) -> (B, h, T, hs)
        y = att @ v
        
        # 7. REASSEMBLE HEADS for Output Projection
        # Transpose: [B, h, T, hs] -> [B, T, h, hs]
        # Reshape:   [B, T, h, hs] -> [B, T, n_head * head_dim]
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # 8. OUTPUT PROJECTION
        # Axis of Collapse (MatMul):
        # Input 'y': [Batch, Time, (n_head*head_dim)]  <-- Axis 2: Input Features
        # Weights:                [(n_head*head_dim), d_model]
        #                          ^-- Axis 0: Input Features (Matches Input Axis 2)
        # Result:    [Batch, Time, d_model]
        return (y @ self.c_proj) + self.b_proj