import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.model.gpt import GPT
from scripts._5_dataset_loader import get_dataloader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # 1. Load Configuration
    with open("config/config.json", "r") as f:
        config = json.load(f)
    
    # 2. Initialize Custom GPT Model
    model = GPT(
        vocab_size=config['vocab_size'],
        d_model=config['n_embd'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        block_size=config['n_positions'],
        dropout=config['resid_pdrop']
    ).to(device)
    
    model.train()
    print(f">>> Model Initialized. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # 3. Initialize DataLoader
    DATA_DIR = r"data\processed\nano_gpt_sequences"
    VOCAB_PATH = r"data\processed\vocab_only_benign.json"
    CONFIG_PATH = r"config\config.json"
    
    train_loader = get_dataloader(DATA_DIR, VOCAB_PATH, CONFIG_PATH, batch_size=32)

    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    # 5. Training Loop

    # Define training limit (None for infinite/full dataset)
    max_steps = 100

    print(">>> Starting Baseline Training...")
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Forward Pass (Returns Logits: [Batch, Time, Vocab])
        logits = model(x)
        
        # Calculate Loss Externally
        # PyTorch cross_entropy expects: logits [N, C], targets [N]
        # B = Batch (32), T = Time (256), C = Vocab (35037)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step:04d} | Loss: {loss.item():.4f}")
            
        # Terminate training after reaching max_steps defined for this run
        if max_steps is not None and step >= max_steps:
            break

    # 6. Checkpointing
    save_path = "models/checkpoints/baseline_custom_gpt.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f">>> Training session completed. Saved to {save_path}")

if __name__ == "__main__":
    train()