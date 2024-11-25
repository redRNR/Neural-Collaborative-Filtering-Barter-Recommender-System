# train_ncf.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def train_ncf(model, train_data, social_adj, num_users, num_items, epochs=10, batch_size=1024, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    user_item_set = set(zip(train_data[:, 1].numpy(), train_data[:, 2].numpy()))

    giver_present = False
    time_present = False

    # Determine if 'giver_indices' and 'time_indices' are present based on train_data shape
    num_columns = train_data.size(1)
    if num_columns >= 3:
        giver_present = True
    if num_columns >= 4:
        time_present = True

    for epoch in range(epochs):
        model.train()
        losses = []
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        for batch in tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            idx = 0
            giver_indices = batch[:, idx] if giver_present else None
            if giver_present:
                idx += 1
            user_indices = batch[:, idx]
            idx += 1
            pos_item_indices = batch[:, idx]
            idx += 1
            if time_present:
                time_indices = batch[:, idx]
                idx += 1

            # Prepare social indices
            social_indices = []
            for u in user_indices:
                u = u.item()
                friends = social_adj[u]
                if len(friends) > 0:
                    social_u = int(np.random.choice(friends))
                else:
                    social_u = u
                social_indices.append(social_u)
            social_indices = torch.LongTensor(social_indices)

            # Sample negative items
            neg_item_indices = torch.randint(0, num_items, size=pos_item_indices.size())
            for i in range(len(user_indices)):
                while (user_indices[i].item(), neg_item_indices[i].item()) in user_item_set:
                    neg_item_indices[i] = torch.randint(0, num_items, (1,))

            # Sample negative givers if giver_present
            if giver_present:
                neg_giver_indices = torch.randint(0, num_users, size=giver_indices.size())
                for i in range(len(giver_indices)):
                    while neg_giver_indices[i].item() == giver_indices[i].item():
                        neg_giver_indices[i] = torch.randint(0, num_users, (1,))
            else:
                neg_giver_indices = giver_indices  # None

            optimizer.zero_grad()
            # Pass giver_indices to the model
            pos_scores = model(user_indices, pos_item_indices, social_indices, giver_indices)
            neg_scores = model(user_indices, neg_item_indices, social_indices, neg_giver_indices)
            loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
