import torch
import tiktoken
from src.model import GPT
from src.training import train_model

if __name__ == "__main__":
    
    # 1. SETUP
    # ---------------------------------------------------------
    # Tokenizer
    enc = tiktoken.get_encoding("gpt2")
    gpt2_vocab_size = enc.n_vocab  # Vocabulary Size of GPT-2


    # Simple Device Detection (NVIDIA or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. MODEL INITIALIZATION 
    # ---------------------------------------------------------
    USE_PRETRAINED_WEIGHTS = True 

    if USE_PRETRAINED_WEIGHTS:
        model = GPT.load_pretrained_weights() 
        print("Loaded OpenAI GPT-2 weights.")
    else:
        print("Initialized with random weights (Training from scratch).")
        model = GPT(vocab_size= gpt2_vocab_size)

    model.to(device)


    # 3. PREPARE DATA (Fake Dataset: 10 sentences, 64 tokens each)
    # ---------------------------------------------------------
    train_data = torch.randint(0, gpt2_vocab_size, (10, 64)).to(device)


    # 4. TRAINING 
    # ---------------------------------------------------------
    model = train_model(model, train_data, epochs=5)

    # 5. INFERENCE demo
    # ---------------------------------------------------------
    print("\n--- INFERENCE PHASE ---")

    model.eval() 
    input_text = "The scientist discovered"
    input_ids = enc.encode(input_text)
    idx = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output_ids = model.generate(idx, max_new_tokens=20)

    output_text = enc.decode(output_ids[0].cpu().tolist())

    print(f"Input:  {input_text}")
    print(f"Output: {output_text}")

