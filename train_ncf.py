import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from ncf_model import NeuralCollaborativeFiltering
from dataset_loader import RateBeerDataset
from evaluate_ncf import auc_score
import pandas as pd
import numpy as np


def train_ncf_model(model, train_loader, num_epochs=10, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for users, items, labels in train_loader:
            users, items, labels = users.view(-1).to(device), items.view(-1).to(device), labels.view(-1).to(device)

            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # Load the RateBeer dataset
    data_path = 'data/ratebeer/transac_dense.csv'  # Update this path if needed
    interactions_df = pd.read_csv(data_path, header=None)
    interactions_df.columns = ['user', 'item', 'rating', 'timestamp']
    
    # Create mappings for user and item IDs
    unique_users = interactions_df['user'].unique()
    unique_items = interactions_df['item'].unique()
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    # Apply mappings
    interactions_df['user'] = interactions_df['user'].map(user_id_map)
    interactions_df['item'] = interactions_df['item'].map(item_id_map)
    
    # Number of users and items after reindexing
    num_users = len(unique_users)
    num_items = len(unique_items)

    # Prepare the interactions list
    user_item_pairs = interactions_df[['user', 'item']].values
    interactions = [(int(row[0]), int(row[1])) for row in user_item_pairs]
    
    # Prepare DataLoader
    dataset = RateBeerDataset(interactions, num_users, num_items)
    train_loader = DataLoader(dataset, batch_size=16384, shuffle=True, pin_memory=True)

    # Initialize and train the model
    model = NeuralCollaborativeFiltering(num_users=num_users, num_items=num_items)
    train_ncf_model(model, train_loader, num_epochs=1000, lr=0.001, device='cuda')

    # Prepare test data for AUC evaluation
    test_data = [
        (user, item, 1) for user, item in interactions[:1000]
    ]
    test_data += [(user, np.random.randint(num_items), 0) for user, _ in interactions[:1000]]

    # Evaluate AUC on the test set
    auc_score(model, test_data, device='cuda')
