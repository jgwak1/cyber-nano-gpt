import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttentionPyTorch(nn.Module):
    
    def __init__(self, d_model=512, n_head=8, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5

        # 1. Attention Weights (Merged Q, K, V)
        self.c_attn = nn.Linear(d_model, 3 * d_model)

        # 2. Output Projection Weights
        self.c_proj = nn.Linear(d_model, d_model)

        # 3. Dropout Layers
        
        # attn_dropout: Drops connections in the "attention(i.e., who looks at whom)" matrix.
        #               i.e., Randomly "blinds" tokens from seeing specific past tokens. Prevents the model from memorizing simple patterns like "Token A always follows Token B."
        self.attn_dropout = nn.Dropout(dropout)

        # resid_dropout: Drops the final output features before adding to the residual.
        #                i.e., Randomly drops information before it enters the main residual highway. This forces the network to distribute information across many neurons rather than relying on a single "super neuron."
        self.resid_dropout = nn.Dropout(dropout)

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
        # TODO: Could utilize "F.scaled_dot_product_attention"
        #       for FlashAttention speedup on RTX 4000. 
        # (B, h, T, hs) @ (B, h, hs, T) -> (B, h, T, T)
        att = (q @ k.transpose(-2, -1)) * self.scale

        # 5. CAUSAL MASKING
        mask = torch.tril(torch.ones(T, T, device=x.device))
        
        # Apply Mask: Where mask is 0, set position to -inf.
        att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)

        # 6. Apply Attention Dropout
        # Drop values right after softmax ( Randomly zeroes out attention scores (i.e., edges in the graph).)
        # This prevents the model from being "too sure" about which previous word matters.
        # (e.g., Prevents the model from just looking at "France" to predict "Paris", forcing it to also attend to "capital" and "The".)
        att = self.attn_dropout(att)

        # 7. AGGREGATE for Attention Outputs
        # (B, h, T, T) @ (B, h, T, hs) -> (B, h, T, hs)
        y = att @ v

        # 8. REASSEMBLE HEADS
        # Permute: [B, h, T, hs] -> [B, T, h, hs]
        # Reshape: [B, T, h, hs] -> [B, T, C] (where C = n_head * head_dim)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # 9. OUTPUT PROJECTION
        # [B, T, C] @ [C, C] -> [B, T, C]
        y = self.c_proj(y)
      
        # 10. Drops the final output features before adding to the residual
        y = self.resid_dropout(y)

        return y
