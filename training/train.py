import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.preprocessing import normalize_sentinel2, to_tensor

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    # Progress bar for the hackathon vibes
    pbar = tqdm(train_loader, desc="Training")
    
    for i, (lr, hr) in enumerate(pbar):
        lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(lr)
        loss = criterion(outputs, hr)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        # For a hackathon demo, stop after 100 steps per epoch to save time
        if i >= 100: 
            break
            
    return running_loss / 100