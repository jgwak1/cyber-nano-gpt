import torch.nn as nn
import torch.nn.functional as F
from .attention import CausalSelfAttentionPyTorch

class MLPPyTorch(nn.Module):
    
    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model
        self.d_ff = 4 * d_model

        # 1. First Linear Layer
        # Shape: [d_model, 4*d_model]
        self.c_fc = nn.Linear(d_model, self.d_ff)

        # 2. Second Linear Layer
        # Shape: [4*d_model, d_model]
        self.c_proj = nn.Linear(self.d_ff, d_model)

    def forward(self, x):
        # x shape: [Batch, Time, d_model]

        # 1. Expand (Linear)
        x = self.c_fc(x)

        # 2. Activate (GELU)
        x = F.gelu(x, approximate='tanh')

        # 3. Project (Linear)
        x = self.c_proj(x)
        return x

    

class TransformerBlockPyTorch(nn.Module):

    def __init__(self, d_model=512, n_head=8):
        super().__init__()
        
        # 1. Attention
        self.attn = CausalSelfAttentionPyTorch(d_model, n_head)
        
        # 2. LayerNorm 1 (Stabilize after Attention)
        self.ln1 = nn.LayerNorm(d_model)
        
        # 3. Feed Forward
        self.mlp = MLPPyTorch(d_model)
        
        # 4. LayerNorm 2 (Stabilize after MLP)
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x):
        # x shape: [Batch, Time, d_model]       
        # 1. Attention
        input_copy = x
        x = self.attn(x)
        x = x + input_copy
        x = self.ln1(x)

        # 2. MLP
        input_copy = x
        x = self.mlp(x)
        x = x + input_copy
        x = self.ln2(x)

        return x
