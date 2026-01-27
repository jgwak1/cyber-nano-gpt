import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TransformerBlockPyTorch

class GPT(nn.Module):
    
   def __init__(self, vocab_size=50257, 
                      d_model=768, 
                      n_layer=12, 
                      n_head=12, 
                      block_size=1024,
                      dropout=0.1):
        
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

        # 3. EMBEDDING DROPOUT
        # Randomly mask out some input tokens completely.
        self.drop = nn.Dropout(dropout)


        # 4. STACKED BLOCKS 
        self.blocks = nn.ModuleList(
            [TransformerBlockPyTorch(d_model, n_head, dropout=dropout) for _ in range(n_layer)]
        )

        # 5. FINAL LAYERNORM
        self.ln_f = nn.LayerNorm(d_model)

        # 6. LANGUAGE MODEL HEAD 
        # - Compare final embedding against every single column in the library to see which one it matches best.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 7. WEIGHT TYING
        # - Point the output head to the same memory as the input embeddings, which saves parameters and typically improves performance.
        self.lm_head.weight = self.wte.weight

        # 8. WEIGHT INITIALIZATION
        # Apply special GPT-2 initialization logic to all sub-modules.
        self.apply(self._init_weights)

        # 9. RESIDUAL SCALING
        # Scale weights of c_proj layers by 1/sqrt(2 * n_layers) to prevent variance explosion in deep residual streams.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layer)**0.5)


   def _init_weights(self, module):
        """
        Standard GPT-2 initialization.
        - Linear/Embedding weights: Normal dist (mean=0, std=0.02)
        - Biases: 0
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


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

        # 4. Dropout (Randomly zeroes out specific elements in the embedding vectors)
        #    - Regularizes the input representation to prevent overfitting.
        #      (e.g., If the feature representing "securityness" is zeroed out in the embedding of token "SSH", 
        #       the model must rely on other features like "protocol-type" to identify it.)
        x = self.drop(x)


        # TRANSFORMER BLOCKS (The Thinking)
        for block in self.blocks:
            x = block(x)

        # FINAL PREDICTION (The Dot Product)
        # 1. Final Norm (First Cleanup)
        x = self.ln_f(x)

        # 2. Calculate Logits (Alignment Scores)
        logits = self.lm_head(x)

        return logits

   
   def generate(self, idx, max_new_tokens, temperature=1.0):
       # autoregressive_decoding
       # idx: [Batch, Time] array of integer indices
       # This loop runs 'max_new_tokens' times. Each iteration generates ONE new word.
      
       self.eval() # Set to evaluation mode
       with torch.no_grad():

         for _ in range(max_new_tokens):
            # 1. Crop Context
            idx_cond = idx[:, -self.block_size:]

            # 2. Forward Pass
            logits = self(idx_cond)

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


   @classmethod
   def load_pretrained_weights(cls, model_type='gpt2'):
      """
      Downloads official OpenAI weights and maps them to our PyTorch model.
      """
      from transformers import GPT2LMHeadModel


      print("Loading weights from HuggingFace...")
      hf_model = GPT2LMHeadModel.from_pretrained(model_type)
      sd = hf_model.state_dict()

      # model = cls(config)
      model = cls()


      with torch.no_grad():    # just initializing memory here, not training.

         # 1. Embeddings
         model.wte.weight.copy_(sd['transformer.wte.weight'])
         model.wpe.weight.copy_(sd['transformer.wpe.weight'])

         # 2. Blocks
         for i, block in enumerate(model.blocks):
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
         model.ln_f.weight.copy_(sd['transformer.ln_f.weight'])
         model.ln_f.bias.copy_(sd['transformer.ln_f.bias'])

         # 4. LM Head
         # Note: HF often ties weights (wte.weight == lm_head.weight).
         # Explicitly copy them here to be safe.
         model.lm_head.weight.copy_(sd['lm_head.weight'])

      return model      