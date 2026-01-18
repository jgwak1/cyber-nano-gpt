import torch.nn.functional as F
import torch.optim as optim

def train_step(model, inputs, targets, optimizer):
    """
    Performs one step of Backpropagation (Training).
    """
    
    model.train() 
    
    optimizer.zero_grad()
    
    logits = model(inputs) 
    
    shift_logits = logits[..., :-1, :].contiguous() # for .view()
    shift_labels = targets[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )
    
    loss.backward()
    
    optimizer.step()
    
    return loss.item()


def train_model(model, train_data, device, epochs=3):
    """
    Runs the training loop for a set number of epochs.
    """
    print("\n=== STARTING TRAINING ===")
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    for epoch in range(epochs):
        loss_val = train_step(model, train_data, train_data, optimizer)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_val:.4f}")
        
    print("=== TRAINING COMPLETE ===\n")
    return model