import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from ncf_model import NeuralCollaborativeFiltering
from dataset_loader import RateBeerDataset
from evaluate_ncf import auc_score
import pandas as pd
import numpy as np

def train_ncf_model(model, train_loader, num_epochs=100, lr=0.001, device='cpu'):
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
    data_path = 'data/ratebeer/transac_dense.csv'
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

    # Create train/test split (10% like MATLAB version)
    num_interactions = len(interactions_df)
    num_test = int(np.floor(num_interactions * 0.1))
    
    # Random permutation for splitting
    indices = np.random.permutation(num_interactions)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    # Create train and test datasets
    train_interactions = [
        (int(user), int(item)) 
        for user, item in interactions_df.iloc[train_indices][['user', 'item']].values
    ]
    test_interactions = [
        (int(user), int(item)) 
        for user, item in interactions_df.iloc[test_indices][['user', 'item']].values
    ]

    # Prepare train DataLoader
    train_dataset = RateBeerDataset(train_interactions, num_users, num_items)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=True)

    # Initialize and train the model
    model = NeuralCollaborativeFiltering(num_users=num_users, num_items=num_items)
    train_ncf_model(model, train_loader, num_epochs=1000, lr=0.001, device='cuda')

    # Prepare test data for AUC evaluation
    test_positives = [(int(user), int(item), 1) for user, item in test_interactions]
    
    # Create negative samples for testing
    test_negatives = []
    for user, _ in test_interactions:
        while True:
            neg_item = np.random.randint(num_items)
            # Check if this is a real interaction
            if not interactions_df[(interactions_df['user'] == user) & 
                                 (interactions_df['item'] == neg_item)].shape[0] > 0:
                test_negatives.append((int(user), int(neg_item), 0))
                break
    
    # Combine positive and negative test samples
    test_data = test_positives + test_negatives
    print(f"Number of test samples: {len(test_data)}")
    
    # Evaluate AUC on the test set
    auc = auc_score(model, test_data, device='cuda')
    print(f"Final AUC: {auc}")