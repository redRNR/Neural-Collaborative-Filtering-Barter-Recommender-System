import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import copy
from loss import BPRLoss

def train_ncf(model, train_data, social_adj, num_users, num_items, 
              epochs=10, batch_size=1024, lr=0.001, use_time=True):
    device = model.user_embedding.weight.device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = BPRLoss(social_adj=None)
    
    print("Training with BPR loss")
    
    best_model_state = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    nan_free_epochs = 0
    
    for epoch in range(epochs):
        model.train()
        losses = []
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        epoch_had_nan = False

        for batch in tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch = batch.to(device)
            
            # Extract indices from batch
            idx = 0
            giver_indices = batch[:, idx] if batch.size(1) >= 3 else None
            if giver_indices is not None:
                idx += 1
            user_indices = batch[:, idx]
            pos_item_indices = batch[:, idx + 1]
            time_values = batch[:, idx + 2] if use_time and batch.size(1) > idx + 2 else None

            # Handle social indices
            if social_adj is None:
                social_indices = user_indices
            else:
                social_indices = []
                for u in user_indices.cpu().numpy():
                    if u < len(social_adj) and len(social_adj[u]) > 0:
                        social_u = int(np.random.choice(social_adj[u]))
                    else:
                        social_u = u
                    social_indices.append(social_u)
                social_indices = torch.LongTensor(social_indices).to(device)

            # Generate negative samples
            neg_item_indices = torch.randint(0, num_items, size=pos_item_indices.size(), device=device)
            neg_giver_indices = torch.randint(0, num_users, size=giver_indices.size(), device=device) if giver_indices is not None else None

            optimizer.zero_grad()
            
            try:
                loss = loss_fn(
                    model,
                    user_indices,
                    pos_item_indices,
                    neg_item_indices,
                    social_indices,
                    giver_indices,
                    neg_giver_indices,
                    time_values
                )
                
                if torch.isnan(loss):
                    print("\nWarning: NaN loss detected")
                    epoch_had_nan = True
                    break
                    
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
            except RuntimeError as e:
                print(f"\nError in training step: {str(e)}")
                epoch_had_nan = True
                break

        if epoch_had_nan:
            print(f"\nNaN detected in epoch {epoch+1}. Reverting to best model state.")
            print(f"Best model was from epoch {nan_free_epochs}")
            model.load_state_dict(best_model_state)
            break
            
        if losses:
            avg_loss = np.mean(losses)
            print(f'Epoch {epoch+1}/{epochs}, BPR Loss: {avg_loss:.4f}')
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(model.state_dict())
                nan_free_epochs = epoch + 1
        else:
            print(f'Epoch {epoch+1}/{epochs}, Loss: No valid batches')
            epoch_had_nan = True
            break

    if epoch_had_nan:
        model.load_state_dict(best_model_state)
        
    return nan_free_epochs