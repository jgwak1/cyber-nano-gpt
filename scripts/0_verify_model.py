import sys
import os
import json
import torch

# Add root to path to import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.gpt import GPT

def verify_architecture():
    # 1. Load the Config JSON
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.json')
    
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        conf = json.load(f)

    print(f"Loaded Config: {conf['n_layer']} layers, {conf['n_embd']} dim")

    # 2. Instantiate Model)
    # ---------------------------------------------------------
    try:
        model = GPT(
            vocab_size=conf['vocab_size'],
            d_model=conf['n_embd'],     
            n_layer=conf['n_layer'],
            n_head=conf['n_head'],
            block_size=conf['n_positions']
        )
        print("Model instantiated successfully.")
    except TypeError as e:
        print(f"Model instantiation Failed: {e}")
        print("Tip: Check if your GPT class arguments match these mapped keys.")
        return

    # 3. Parameter Count Check
    # ---------------------------------------------------------
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params/1e6:.2f}M")
    
    # Check if it fits 60M parameters (nano-scale)
    if 50 < params/1e6 < 70:
        print("   -> Perfect fit for 'Nano' (Goal: 60M)")
    else:
        print("   -> Note: Slightly off from 60M goal, but valid.")

    # 4. Dummy Forward Pass
    # ---------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on: {device}")
    model.to(device)
    
    # Create fake input: Batch=2, Seq_Len=128
    dummy_idx = torch.randint(0, conf['vocab_size'], (2, 128)).to(device)
    
    try:
        logits = model(dummy_idx)
        print(f"Forward pass successful. Output shape: {logits.shape}")
        
        # Verify shape is [Batch, Seq, Vocab]
        assert logits.shape == (2, 128, conf['vocab_size'])
        print("   -> Output shape matches expectation.")
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    verify_architecture()