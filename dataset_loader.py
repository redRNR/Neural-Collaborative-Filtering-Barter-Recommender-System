
import numpy as np
import torch
from torch.utils.data import Dataset

class RateBeerDataset(Dataset):
    def __init__(self, interactions, num_users, num_items, num_negatives=4):
        self.interactions = interactions  # List of (user, item) tuples for positive interactions
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_item_set = set(interactions)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, pos_item = self.interactions[idx]
        
        # Positive sample
        users = [user]
        items = [pos_item]
        labels = [1]

        # Generate negative samples
        for _ in range(self.num_negatives):
            neg_item = np.random.randint(self.num_items)
            while (user, neg_item) in self.user_item_set:
                neg_item = np.random.randint(self.num_items)
            users.append(user)
            items.append(neg_item)
            labels.append(0)

        return torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(labels)
