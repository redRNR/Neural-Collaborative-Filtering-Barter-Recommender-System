# train_ncf.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def train_ncf(model, train_data, social_adj, num_users, num_items, 
              epochs=10, batch_size=1024, lr=0.001):
    device = model.user_embedding.weight.device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        losses = []
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        for batch in tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch = batch.to(device)  # Move batch to device
            
            idx = 0
            giver_indices = batch[:, idx] if batch.size(1) >= 3 else None
            if giver_indices is not None:
                idx += 1
            user_indices = batch[:, idx]
            idx += 1
            pos_item_indices = batch[:, idx]

            # Move social indices to device
            social_indices = []
            for u in user_indices.cpu().numpy():
                friends = social_adj[u]
                if len(friends) > 0:
                    social_u = int(np.random.choice(friends))
                else:
                    social_u = u
                social_indices.append(social_u)
            social_indices = torch.LongTensor(social_indices).to(device)

            # Generate negative samples directly on device
            neg_item_indices = torch.randint(0, num_items, 
                                           size=pos_item_indices.size(), 
                                           device=device)
            
            if giver_indices is not None:
                neg_giver_indices = torch.randint(0, num_users, 
                                                size=giver_indices.size(), 
                                                device=device)
            else:
                neg_giver_indices = None

            optimizer.zero_grad()
            pos_scores = model(user_indices, pos_item_indices, 
                             social_indices, giver_indices)
            neg_scores = model(user_indices, neg_item_indices, 
                             social_indices, neg_giver_indices)
            loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
