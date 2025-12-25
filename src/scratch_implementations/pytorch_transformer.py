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

    def forward(self, x):
        # x shape: [Batch, Time, Channel] (B, T, C)
        B, T, C = x.size()

        # 1. LINEAR PROJECTION
        # [B, T, C] @ [C, 3*C] -> [B, T, 3*C]
        qkv = self.c_attn(x)

        # 2. SPLIT Q, K, V
        # Split last dim into 3 parts. Result: 3 tensors of [B, T, C]
        q, k, v = qkv.split(self.d_model, dim=2)

        # 3. RESHAPE & TRANSPOSE (The "Permute" in PyTorch)
        # Reshape: [B, T, n_head * head_dim] -> [B, T, n_head, head_dim]
        # Permute: Swap Time(1) and Head(2)  -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        # 4. SCALED DOT-PRODUCT ATTENTION
        # (B, h, T, hs) @ (B, h, hs, T) -> (B, h, T, T)
        att = (q @ k.transpose(-2, -1)) * self.scale

        # 5. CAUSAL MASKING
        mask = torch.tril(torch.ones(T, T, device=x.device))
        
        # Apply Mask: Where mask is 0, set position to -inf.
        att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)

        # 6. AGGREGATE for Attention Outputs
        # (B, h, T, T) @ (B, h, T, hs) -> (B, h, T, hs)
        y = att @ v

        # 7. REASSEMBLE HEADS
        # Permute: [B, h, T, hs] -> [B, T, h, hs]
        # Reshape: [B, T, h, hs] -> [B, T, C] (where C = n_head * head_dim)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # 8. OUTPUT PROJECTION
        # [B, T, C] @ [C, C] -> [B, T, C]
        return self.c_proj(y)



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

    