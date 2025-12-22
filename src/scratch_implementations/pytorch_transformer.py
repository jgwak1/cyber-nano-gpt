import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttentionPyTorch(nn.Module):
    
    def __init__(self, d_model=512, n_head=8, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5

        # 1. Attention Weights (Merged Q, K, V)
        self.c_attn = nn.Linear(d_model, 3 * d_model)

        # 2. Output Projection Weights
        self.c_proj = nn.Linear(d_model, d_model)


    # TODO
    def forward(self, x):
       pass