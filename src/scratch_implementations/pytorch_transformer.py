import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken
from transformers import GPT2LMHeadModel

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



class GPT_Inference(nn.Module):
    
    def __init__(self, vocab_size=50257, 
                       d_model=768, 
                       n_layer=12, 
                       n_head=12, 
                       block_size=1024):
        
        super().__init__()

        # block_size = Context Window (Max Sequence Length)
        self.block_size = block_size

        # 1. TOKEN EMBEDDINGS (wte = Word Token Embeddings)
        #    Since this class if for Inference-only, 
        #    overwrite these with OpenAI's pre-trained weights later.
        self.wte = nn.Embedding(vocab_size, d_model)

        # 2. POSITION EMBEDDINGS (wpe = Word Position Embeddings)
        # - Learned positions instead of using sine/cosine waves (unlike original 2017 paper).
        # - A unique vector learned for every single slot in the context window.
        # Also overwrite these with OpenAI's pre-trained weights later.
        self.wpe = nn.Embedding(block_size, d_model)

        # 3. STACKED BLOCKS 
        self.blocks = nn.ModuleList(
            [TransformerBlockPyTorch(d_model, n_head) for _ in range(n_layer)]
        )

        # 4. FINAL LAYERNORM
        self.ln_f = nn.LayerNorm(d_model)

        # 5. LANGUAGE MODEL HEAD 
        # - Compare final embedding against every single column in the library to see which one it matches best.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        # idx: [Batch, Time] (Integer indices provided by the Tokenizer)
        device = idx.device
        b, t = idx.size()

        # 1. Word Meaning
        tok_emb = self.wte(idx) # [Batch, Time, d_model]

        # 2. Position Embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.wpe(pos) # [Time, d_model]

        # 3. Position-Aware Embeddings
        # (Token Embeddings + Position Embeddings)
        x = tok_emb + pos_emb

        # TRANSFORMER BLOCKS (The Thinking)
        for block in self.blocks:
            x = block(x)

        # FINAL PREDICTION (The Dot Product)
        # 1. Final Norm (First Cleanup)
        x = self.ln_f(x)

        # 2. Calculate Logits (Alignment Scores)
        logits = self.lm_head(x)

        return logits


def load_pretrained_weights(my_model, model_type='gpt2'):
    """
    Downloads official OpenAI weights and maps them to our PyTorch model.
    """
  
    print("Loading weights from HuggingFace...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_type)
    sd = hf_model.state_dict()

    with torch.no_grad():    # just initializing memory here, not training.

        # 1. Embeddings
        my_model.wte.weight.copy_(sd['transformer.wte.weight'])
        my_model.wpe.weight.copy_(sd['transformer.wpe.weight'])

        # 2. Blocks
        for i, block in enumerate(my_model.blocks):
            prefix = f'transformer.h.{i}'

            # Layer Norms
            block.ln1.weight.copy_(sd[f'{prefix}.ln_1.weight'])
            block.ln1.bias.copy_(sd[f'{prefix}.ln_1.bias'])
            block.ln2.weight.copy_(sd[f'{prefix}.ln_2.weight'])
            block.ln2.bias.copy_(sd[f'{prefix}.ln_2.bias'])

            # Attention
            #    
            # PyTorch nn.Linear weights are [Out, In].
            # HF GPT2 Conv1D weights are [In, Out].
            # So we must TRANSPOSE (.t()) weights when copying from HF to our nn.Linear.
            
            # c_attn weight; transposed
            block.attn.c_attn.weight.copy_(sd[f'{prefix}.attn.c_attn.weight'].t())
            block.attn.c_attn.bias.copy_(sd[f'{prefix}.attn.c_attn.bias'])
            
            # c_proj weight; transposed
            block.attn.c_proj.weight.copy_(sd[f'{prefix}.attn.c_proj.weight'].t())
            block.attn.c_proj.bias.copy_(sd[f'{prefix}.attn.c_proj.bias'])

            # MLP
            # c_fc weight; transposed
            block.mlp.c_fc.weight.copy_(sd[f'{prefix}.mlp.c_fc.weight'].t())
            block.mlp.c_fc.bias.copy_(sd[f'{prefix}.mlp.c_fc.bias'])

            # c_proj weight; transposed
            block.mlp.c_proj.weight.copy_(sd[f'{prefix}.mlp.c_proj.weight'].t())
            block.mlp.c_proj.bias.copy_(sd[f'{prefix}.mlp.c_proj.bias'])

        # 3. Final Norm
        my_model.ln_f.weight.copy_(sd['transformer.ln_f.weight'])
        my_model.ln_f.bias.copy_(sd['transformer.ln_f.bias'])

        # 4. LM Head
        # Note: HF often ties weights (wte.weight == lm_head.weight).
        # Explicitly copy them here to be safe.
        my_model.lm_head.weight.copy_(sd['lm_head.weight'])

    return my_model


def autoregressive_decoding(model, idx, max_new_tokens, temperature=1.0):
    # idx: [Batch, Time] array of integer indices
    # This loop runs 'max_new_tokens' times. Each iteration generates ONE new word.
    
    model.eval() # Set to evaluation mode
    with torch.no_grad():

        for _ in range(max_new_tokens):
            # 1. Crop Context
            idx_cond = idx[:, -model.block_size:]

            # 2. Forward Pass
            logits = model(idx_cond)

            # 3. Focus on the LAST word
            last_logits = logits[:, -1, :] # [Batch, Vocab]

            # 4. Apply Temperature
            scaled_logits = last_logits / temperature

            # 5. Softmax -> Probabilities
            probs = F.softmax(scaled_logits, dim=-1)

            # 6. Sample (The Choice)
            # torch.multinomial handles the sampling logic efficiently
            idx_next = torch.multinomial(probs, num_samples=1) # [Batch, 1]

            # 7. Update Sequence
            idx = torch.cat((idx, idx_next), dim=1)

    return idx