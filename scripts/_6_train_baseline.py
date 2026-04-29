import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import glob
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.model.gpt import GPT
from scripts._5_dataset_loader import get_dataloader
from scripts._5_dataset_loader import CyberDataset




@torch.no_grad()
def estimate_loss(model, dataloader, eval_iters=20):

    """
        Statistically estimates the loss trajectory over the entire dataset by sampling random batches.
        
        This prevents the physical time bottleneck that occurs when running a forward pass 
        over the entire large-scale validation set. By iterating only over a limited number 
        of batches defined by `eval_iters`, it ensures that GPU compute resources are focused 
        on training rather than evaluation, while still allowing for meaningful overfitting monitoring.
        
        Args:
            model (GPT): The PyTorch model to evaluate.
            dataloader (DataLoader): The target DataLoader to sample from (Train or Validation).
            eval_iters (int): The number of batches to use for the estimation (default: 20).
            
        Returns:
            float: The averaged loss value across the sampled batches.
    """

    model.eval() # Disable dropout and batchnorm during inference
    losses = torch.zeros(eval_iters)
    data_iter = iter(dataloader)

    device = next(model.parameters()).device

    for k in range(eval_iters):
        try:
            x,y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
        losses[k] = loss.item()

    model.train() # restore training mode
    return losses.mean().item()


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

    # 3. Initialize DataLoader and Split (90/10)
    DATA_DIR = r"data\processed\nano_gpt_sequences"
    VOCAB_PATH = r"data\processed\vocab_only_benign.json"
    CONFIG_PATH = r"config\config.json"
    
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "part-*.csv")))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    

    # Each file is an independent shard (no cross-file sequences).
    # Shuffling prevents 'Train=Past / Val=Future' bias from sequential file numbering.
    # This ensures the model trains across the entire timeline, not just the first 90%.
    # (Task: Next-Token Prediction; not time-series forecasting on the final 10%.)
    random.seed(42)
    random.shuffle(all_files)

    split_idx = int(0.9 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f">>> File Split: {len(train_files)} Train files, {len(val_files)} Validation files.")

    # train_dataset = CyberDataset(train_files, VOCAB_PATH, CONFIG_PATH)
    # val_dataset = CyberDataset(val_files, VOCAB_PATH, CONFIG_PATH)
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    train_loader = get_dataloader(train_files, VOCAB_PATH, CONFIG_PATH)
    val_loader = get_dataloader(val_files, VOCAB_PATH, CONFIG_PATH)



    # full_loader = get_dataloader(DATA_DIR, VOCAB_PATH, CONFIG_PATH, batch_size=32)
    # dataset = full_loader.dataset

    # train_size = int(0.9 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split( dataset, [train_size, val_size] )

    # # TODO: Refactor dataset/loader separation later
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle= False)    
    # print(f">>> Dataset Split: {train_size} Train samples, {val_size} Validation samples.")


    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
        # 5. Training Loop


    # 4.5 Checkpoint Resume Logic (Stateful Training)    
    checkpoint_dir = "models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True) # Ensure directory exists
    start_step = 0
    
    if os.path.exists(checkpoint_dir):
        list_of_files = glob.glob(f"{checkpoint_dir}/*.pt")
        if list_of_files:
            latest_ckpt = max(list_of_files, key=os.path.getctime)
            print(f">>> Found checkpoint: {latest_ckpt}. Restoring state...")
            
            # checkpoint = torch.load(latest_ckpt, map_location=device)
            # model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # start_step = checkpoint['step'] + 1
            
            # Load checkpoint with flexibility for different save formats
            checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
            
            # TODO: Refactor -----------
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Case 1: Full training state dictionary exists
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_step = checkpoint.get('step', 0) + 1
                print(f">>> Full state restored. Resuming from step {start_step - 1}.")
            else:
                # Case 2: File contains only raw state_dict (weights)
                model.load_state_dict(checkpoint)
                start_step = 0 # Cannot resume step or optimizer from raw weights
                print(">>> Raw weights loaded. Optimizer and step count reset.")
            #  -----------

            print(f">>> Successfully resumed from step {start_step - 1}")
        else:
            print(">>> Checkpoint directory exists, but no valid .pt files found. Starting from scratch.")
    else:
        print(">>> No checkpoint directory found. Starting from scratch.")
    # =========================================================================

    # Define training limit (None for infinite/full dataset)
    max_steps = 1000
    eval_interval = 50 # Perform validation and checkpointing every 500 steps

    print(">>> Starting Long-run Baseline Training...")
    for step, (x, y) in enumerate(train_loader):

        # Fast-foward to the resumed step if recovering from a checkpoint
        if step < start_step:
            continue

        # --- Validation & Checkpointing ---

        if step > 0 and step % eval_interval == 0:
            val_loss = estimate_loss(model, val_loader)
            train_loss_est = estimate_loss(model, train_loader)

            print(f"Step {step:05d} | Train Loss: {train_loss_est:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save Checkpoint
            ckpt_path = f"{checkpoint_dir}/baseline_step_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_est,
                'val_loss': val_loss
            }, ckpt_path)

        
        # ----------------------------------

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