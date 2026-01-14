import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
from transformers import TFGPT2LMHeadModel
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
    

class GPT_Inference_TF(Model):
    def __init__(self, vocab_size=50257, d_model=768, n_layer=12, n_head=12, max_len=1024):
        super().__init__()
        
        self.d_model = d_model

        # block_size = Context Window (Max Sequence Length)
        self.block_size = max_len

        # 1. TOKEN EMBEDDINGS (wte = Word Token Embeddings)
        #    Since this class if for Inference-only, 
        #    overwrite these with OpenAI's pre-trained weights later.
        self.wte = layers.Embedding(vocab_size, d_model, name="wte")

        # 2. POSITION EMBEDDINGS (wpe = Word Position Embeddings)
        # - Learned positions instead of using sine/cosine waves (unlike original 2017 paper).
        # - A unique vector learned for every single slot in the context window.
        # Also overwrite these with OpenAI's pre-trained weights later.
        self.wpe = layers.Embedding(max_len, d_model, name="wpe")

        # 3. STACKED BLOCKS 
        self.blocks_list = [
            TransformerBlockTF(d_model, n_head, name=f"h_{i}") 
            for i in range(n_layer)
        ]

        # 4. FINAL LAYERNORM
        self.ln_f = layers.LayerNormalization(epsilon=1e-5, name="ln_f")
        
        # 5. LANGUAGE MODEL HEAD 
        # - In most classic Transformers, "self.lm_head.weight = self.wte.weight" for "Semantic Consistency" and "Parameter Efficiency" 
        # - Compare final embedding against every single column in the library to see which one it matches best.
        self.lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

    def call(self, idx):
        # idx: [Batch, Time] (Integer indices provided by the Tokenizer)

        batch_size = tf.shape(idx)[0]
        t = tf.shape(idx)[1]
        
        # Word Embeddings
        tok_emb = self.wte(idx)

        # Position Embeddings
        pos = tf.range(0, t, dtype=tf.int32) # POSITION INDICES
        pos_emb = self.wpe(pos)
        
        # Position-Aware Word Embeddings
        # (Token Embeddings + Position Embeddings)
        x = tok_emb + pos_emb
        
        # TRANSFORMER BLOCKS (The Thinking)
        for block in self.blocks_list:
            x = block(x)
            
        # FINAL PREDICTION (The Dot Product) 
        # 1. Final Norm (First Cleanup)
        x = self.ln_f(x)
        
        # 2. Calculate Logits (Alignment Scores)
        #     x: (B, T, 768)
        #     lm_head: (768, 50257)
        #     Result: (B, T, 50257)
        #     
        #     we compare our contextualized thought vectors against the 
        #     static dictionary definitions to get raw scores (logits).
        logits = self.lm_head(x)
        
        return logits


def load_pretrained_weights_tf(my_model, model_type='gpt2'):
    """
    Downloads official OpenAI weights (via HuggingFace) and maps them to our TF model.
    """
    print(f"Loading weights from HuggingFace ({model_type})...")
    
    hf_model = TFGPT2LMHeadModel.from_pretrained(model_type)
    
    my_model.wte.set_weights(hf_model.transformer.wte.get_weights())
    my_model.wpe.set_weights(hf_model.transformer.wpe.get_weights())

    for i, block in enumerate(my_model.blocks_list):
        hf_block = hf_model.transformer.h[i]

        block.ln1.set_weights(hf_block.ln_1.get_weights())
        block.ln2.set_weights(hf_block.ln_2.get_weights())

        block.attn.c_attn.set_weights(hf_block.attn.c_attn.get_weights())
        block.attn.c_proj.set_weights(hf_block.attn.c_proj.get_weights())

        block.mlp.c_fc.set_weights(hf_block.mlp.c_fc.get_weights())
        block.mlp.c_proj.set_weights(hf_block.mlp.c_proj.get_weights())

    my_model.ln_f.set_weights(hf_model.transformer.ln_f.get_weights())

    try:

        if hasattr(hf_model, 'lm_head'):
             my_model.lm_head.set_weights(hf_model.lm_head.get_weights())
        else:
             print("Loading tied weights from WTE to LM Head (Transposing)...")
             wte_weights = hf_model.transformer.wte.get_weights()[0] # [Vocab, Dim]
             my_model.lm_head.set_weights([wte_weights.T]) # [Dim, Vocab]
             
    except Exception as e:
        print(f"Warning: Could not load LM Head weights directly: {e}")
        print("Using WTE weights (Weight Tying) as fallback.")
        wte_weights = my_model.wte.get_weights()[0] # [Vocab, Dim]
        # Dense layer expects [Dim, Vocab], so we Transpose.
        my_model.lm_head.set_weights([wte_weights.T])

    print("Weights loaded successfully!")
    return my_model



def autoregressive_decoding_tf(model, idx, max_new_tokens, temperature=1.0):
    # idx: [Batch, Time] array of integer indices
    # This loop runs 'max_new_tokens' times. Each iteration generates ONE new word.

    for _ in range(max_new_tokens):
        
        # 1. Crop Context (Sliding Window)
        idx_cond = idx[:, -model.block_size:]
        
        # 2. Forward Pass
        logits = model(idx_cond)
        
        # 3. Focus on the LAST token
        last_logits = logits[:, -1, :]
        
        # 4. Scale by Temperature
        scaled_logits = last_logits / temperature
        
        # 5. Sample (Internal-Softmax -> Probabilities -> Sampling)
        idx_next = tf.random.categorical(scaled_logits, num_samples=1, dtype=tf.int32)
        
        # 6. Update Sequence
        idx = tf.concat([idx, idx_next], axis=1)
        
    return idx

@tf.function # compiles function into static C++ graph for faster execution
def train_step(model, inputs, targets, optimizer):
    """
    Performs one step of Backpropagation (Training).
    """

    with tf.GradientTape() as tape:
        
        logits = model(inputs, training=True) 

        shift_logits = logits[:, :-1, :]
        shift_labels = targets[:, 1:]

        logits_flat = tf.reshape(shift_logits, (-1, tf.shape(shift_logits)[-1]))
        labels_flat = tf.reshape(shift_labels, (-1,))

        loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
        
        loss = loss_fn(labels_flat, logits_flat)

    grads = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def train_model(model, train_data, epochs=3):
    """
    Runs the training loop for a set number of epochs.
    """
    print("\n=== STARTING TRAINING ===")
    optimizer = optimizers.AdamW(learning_rate=3e-4)

    for epoch in range(epochs):
        loss_val = train_step(model, train_data, train_data, optimizer)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {float(loss_val):.4f}")
        
    print("=== TRAINING COMPLETE ===\n")
    return model





