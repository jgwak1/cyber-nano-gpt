import numpy as np
import tiktoken  
from transformers import GPT2LMHeadModel

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
    



class MLPNumPy:
    def __init__(self, d_model=512):
        self.d_model = d_model
        # The hidden layer is typically 4 times the size of the embedding
        self.d_ff = 4 * d_model 
        
        # 1. First Linear Layer (Expansion)
        # Shape: [d_model, 4*d_model]
        self.c_fc = np.random.normal(scale=0.02, size=(d_model, self.d_ff))
        self.b_fc = np.zeros(self.d_ff)
        
        # 2. Second Linear Layer (Projection)
        # Shape: [4*d_model, d_model]
        self.c_proj = np.random.normal(scale=0.02, size=(self.d_ff, d_model))
        self.b_proj = np.zeros(d_model)

    def gelu(self, x):
        """
        Gaussian Error Linear Unit (GELU).
        """
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def forward(self, x):
        # x shape: [Batch, Time, d_model]
        
        # 1. Expand (Linear)
        # Use @ for safer batch matrix multiplication
        # [B, T, d_model] @ [d_model, 4*d_model] -> [B, T, 4*d_model]
        x = (x @ self.c_fc) + self.b_fc
        
        # 2. Activate (GELU)
        x = self.gelu(x)
        
        # 3. Project (Linear)
        # Shape: [Batch, Time, 4*d_model] @ [4*d_model, d_model] -> [Batch, Time, d_model]
        x = (x @ self.c_proj) + self.b_proj
        
        return x
    

class LayerNormNumPy:
    def __init__(self, d_model=512, eps=1e-5):
        self.eps = eps
        
        # Learnable Parameters (Gamma and Beta)
        # Gamma (Scale): Starts at 1.0 (do nothing)
        self.gamma = np.ones(d_model)
        # Beta (Shift): Starts at 0.0 (do nothing)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        # x shape: [Batch, Time, d_model]
        
        # 1. Calculate Mean
        mean = np.mean(x, axis=-1, keepdims=True)
        
        # 2. Calculate Variance
        variance = np.var(x, axis=-1, keepdims=True)
        
        # 3. Normalize
        # Subtract mean, divide by standard deviation.
        # (x - mean): Centers the data (Broadcasting: [B,T,D] - [B,T,1])
        # / sqrt(var): Scales the spread to 1.
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        
        # 4. Scale and Shift (Learnable)
        # The model might decide "actually, I want this word to be slightly positive".
        # Gamma stretches it; Beta shifts it.
        output = self.gamma * x_norm + self.beta
        
        return output
    


class TransformerBlockNumPy:
    def __init__(self, d_model=512, n_head=8):
        
        # 1. Attention (The Router)
        self.attn = CausalSelfAttentionNumPy(d_model, n_head)
        
        # 2. LayerNorm 1 (The Stabilizer after Attention)
        self.ln1 = LayerNormNumPy(d_model)
        
        # 3. Feed Forward (The Thinker)
        self.mlp = MLPNumPy(d_model)
        
        # 4. LayerNorm 2 (The Stabilizer after MLP)
        self.ln2 = LayerNormNumPy(d_model)

    def forward(self, x):
        # x shape: [Batch, Time, d_model]

        # ATTENTION
        # 1. Save the Input (The Identity)
        input_copy = x
        # 2. Calculate the Updates (The Residual)
        x = self.attn.forward(x)
        # 3. Add & Norm
        x = x + input_copy        
        # Normalize the result
        x = self.ln1.forward(x)
        
        # MLP
        # 4. Save the Input again
        input_copy = x
        # 5. Calculate the Updates
        x = self.mlp.forward(x)        
        # 6. Add & Norm
        # Same logic: Backprop gradient flows through the '1' in (1 + dF/dx).
        x = x + input_copy        
        # Normalize
        x = self.ln2.forward(x)
        
        return x



class GPT_Inference:
    def __init__(self, vocab_size=50257, d_model=768, n_layer=12, n_head=12, block_size=1024):
        # block_size = Context Window (Max Sequence Length)
        self.block_size = block_size 
        
        # 1. TOKEN EMBEDDINGS (wte = Word Token Embeddings)
        #    Since this class if for Inference-only, 
        #    overwrite these with OpenAI's pre-trained weights later.
        self.wte = np.random.normal(scale=0.02, size=(vocab_size, d_model))
        
        # 2. POSITION EMBEDDINGS (wpe = Word Position Embeddings)
        # - Learned positions instead of using sine/cosine waves (unlike original 2017 paper).
        # - A unique vector learned for every single slot in the context window.
        # Also overwrite these with OpenAI's pre-trained weights later.
        self.wpe = np.random.normal(scale=0.02, size=(block_size, d_model))
        
        # 3. STACKED BLOCKS 
        self.blocks = [TransformerBlockNumPy(d_model, n_head) for _ in range(n_layer)]
        
        # 4. FINAL LAYERNORM
        self.ln_f = LayerNormNumPy(d_model)
        
        # 5. LANGUAGE MODEL HEAD 
        # - Compare final embedding against every single
        # - column in the library to see which one it matches best.
        self.lm_head = np.random.normal(scale=0.02, size=(d_model, vocab_size))

    def forward(self, idx):
        # idx: [Batch, Time] (Integer indices provided by the Tokenizer)
        batch, time = idx.shape
        
        # 1. Word Meaning
        tok_emb = self.wte[idx] # Shape: [Batch, Time, 768]
        
        # 2. Position Meaning
        pos_emb = self.wpe[np.arange(time)] # Shape: [Time, 768]
        
        # 3. Position-Aware Embeddings
        # (Token Embeddings + Position Embeddings)
        x = tok_emb + pos_emb
        
        # TRANSFORMER BLOCKS (The Thinking)
        for block in self.blocks:
            x = block.forward(x)
            
        # FINAL PREDICTION (The Dot Product)
        # 1. First Cleanup
        x = self.ln_f.forward(x)
        
        # 2. Calculate Logits (Alignment Scores)
        logits = x @ self.lm_head
        
        return logits
    


def load_pretrained_weights(gpt_numpy_model, model_type='gpt2'):
    """
    Downloads the official pre-trained weights from OpenAI (via HuggingFace)
    and maps them into our manual NumPy class structure.
    """
    
    # 1. Downloads about ~500MB of data (for GPT-2 Small).
    #    'hf_model' is a PyTorch object containing all the learned numbers.
    hf_model = GPT2LMHeadModel.from_pretrained(model_type)
    
    sd = hf_model.state_dict()
    
    print("Injecting weights into NumPy model...")
    
    # HELPERs for PyTorch -> NumPy
    def get_w(key): 
        return sd[key].detach().numpy().T 
    def get_b(key): 
        return sd[key].detach().numpy()

    # WTE (Word Token Embeddings)
    gpt_numpy_model.wte = sd['transformer.wte.weight'].detach().numpy()
    
    # WPE (Word Position Embeddings)
    gpt_numpy_model.wpe = sd['transformer.wpe.weight'].detach().numpy()
    
    for i, block in enumerate(gpt_numpy_model.blocks):

        prefix = f'transformer.h.{i}'        

        # --- Layer Norms ---
        block.ln1.gamma = get_b(f'{prefix}.ln_1.weight')
        block.ln1.beta  = get_b(f'{prefix}.ln_1.bias')
        block.ln2.gamma = get_b(f'{prefix}.ln_2.weight')
        block.ln2.beta  = get_b(f'{prefix}.ln_2.bias')
        
        # --- Attention ---
        block.attn.c_attn.w = sd[f'{prefix}.attn.c_attn.weight'].detach().numpy()
        block.attn.c_attn.b = get_b(f'{prefix}.attn.c_attn.bias')
        
        # Output Projection (c_proj): Mixing the head results back together
        block.attn.c_proj.w = sd[f'{prefix}.attn.c_proj.weight'].detach().numpy()
        block.attn.c_proj.b = get_b(f'{prefix}.attn.c_proj.bias')
        
        # --- MLP (Feed Forward) ---
        block.mlp.c_fc.w = sd[f'{prefix}.mlp.c_fc.weight'].detach().numpy()
        block.mlp.c_fc.b = get_b(f'{prefix}.mlp.c_fc.bias')
        
        block.mlp.c_proj.w = sd[f'{prefix}.mlp.c_proj.weight'].detach().numpy()
        block.mlp.c_proj.b = get_b(f'{prefix}.mlp.c_proj.bias')

    # Final Layer Norm weights
    gpt_numpy_model.ln_f.gamma = get_b('transformer.ln_f.weight')
    gpt_numpy_model.ln_f.beta  = get_b('transformer.ln_f.bias')
    
    # LM Head (The Vocabulary Projection)
    gpt_numpy_model.lm_head    = get_w('lm_head.weight')

    return gpt_numpy_model


# =============================================================================
# 3. THE DRIVER (The Functions you asked for)
# =============================================================================
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def generate(model, idx, max_new_tokens, temperature=1.0):
    # idx: [Batch, Time] array of integer indices
    # This loop runs 'max_new_tokens' times. Each iteration generates ONE new word.    
    
    for _ in range(max_new_tokens):
        # 1. Crop Context
        idx_cond = idx[:, -model.block_size:]
        
        # 2. Forward Pass
        logits = model.forward(idx_cond)
        
        # 3. Focus on the LAST word
        last_time_step_logits = logits[:, -1, :] # Shape: [Batch, VocabSize]
        
        # 4. Apply Temperature
        scaled_logits = last_time_step_logits / temperature
        
        # 5. Apply Softmax
        # Turn raw scores (logits) into Probabilities (percentages).
        probs = softmax(scaled_logits)
        
        idx_next = []
        for i in range(idx.shape[0]):
            # i = The current sentence index in the batch
            # probs[i] = The probability list for THIS specific sentence
            next_token = np.random.choice(len(probs[i]), 
                                          p=probs[i])
            idx_next.append(next_token)
            
        # 7. Update Sequence
        idx_next = np.array(idx_next).reshape(-1, 1)
        idx = np.concatenate((idx, idx_next), axis=1)
        
    return idx

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    # 1. Tokenizer (The Dictionary)
    enc = tiktoken.get_encoding("gpt2")
    
    # 2. Setup Model (The Engine)
    # Create the empty structure (random noise)
    model = GPT_Inference()
    # Download and inject the learned weights (OpenAI weights)
    model = load_pretrained_weights(model)
    
    # 3. Encode Input
    input_text = "The scientist discovered"
    input_ids = enc.encode(input_text)
    
    # Add Batch Dimension
    # The model expects [Batch, Time].
    # We have [Time]. So we wrap it in an extra list: [[Time]]
    idx = np.array([input_ids]) 
    
    # 4. Generate
    output_ids = generate(model, idx, max_new_tokens=20)
    
    # 5. Decode Output using Tokenizer
    output_text = enc.decode(output_ids[0].tolist())
    
    print(f"\nInput:  {input_text}")
    print(f"Output: {output_text}")