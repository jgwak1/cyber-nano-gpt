import torch.nn as nn
import torch.nn.functional as F
from .attention import CausalSelfAttentionPyTorch

class MLPPyTorch(nn.Module):
    
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = 4 * d_model

        # 1. First Linear Layer
        # Shape: [d_model, 4*d_model]
        self.c_fc = nn.Linear(d_model, self.d_ff)

        # 2. Second Linear Layer
        # Shape: [4*d_model, d_model]
        self.c_proj = nn.Linear(self.d_ff, d_model)

        # 3. Dropout layer
        # Randomly zero out neurons to prevent overfitting.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [Batch, Time, d_model]

        # 1. Expand (Linear)
        x = self.c_fc(x)

        # 2. Activate (GELU)
        x = F.gelu(x, approximate='tanh')

        # 3. Project (Linear)
        x = self.c_proj(x)

        # 4. Dropout 
        #    Regularizes the MLP output to improve generalization, before they are added to the residual stream in TransformerBlock
        x = self.dropout(x)

        return x

    

class TransformerBlockPyTorch(nn.Module):

    def __init__(self, d_model=512, n_head=8, dropout=0.1):
        super().__init__()
        
        # 1. Attention
        self.attn = CausalSelfAttentionPyTorch(d_model, n_head, dropout= dropout)
        
        # 2. LayerNorm 1 (Stabilize after Attention)
        self.ln1 = nn.LayerNorm(d_model)
        
        # 3. Feed Forward
        self.mlp = MLPPyTorch(d_model, dropout=dropout)
        
        # 4. LayerNorm 2 (Stabilize after MLP)
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x):
        # x shape: [Batch, Time, d_model]       
        # 1. Attention
        input_copy = x
        x = self.ln1(x)   # Pre-Norm
        x = self.attn(x)
        x = x + input_copy

        # 2. MLP
        input_copy = x
        x = self.ln2(x)   # Pre-Norm 
        x = self.mlp(x)
        x = x + input_copy

        return x
