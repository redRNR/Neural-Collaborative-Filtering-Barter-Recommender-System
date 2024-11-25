# ncf.py

import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[64, 32, 16, 8]):
        super(NCF, self).__init__()
        # Add extra embedding dimension for padding
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=num_users)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=num_items)
        self.social_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=num_users)
        self.giver_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=num_users)

        input_size = embedding_dim * 4
        mlp_layers = []
        for hidden_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            input_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)
        self.output = nn.Linear(hidden_layers[-1], 1)

    def forward(self, user_indices, item_indices, social_indices, giver_indices):
        # Handle out-of-range indices
        user_indices = torch.clamp(user_indices, 0, self.user_embedding.num_embeddings - 1)
        item_indices = torch.clamp(item_indices, 0, self.item_embedding.num_embeddings - 1)
        social_indices = torch.clamp(social_indices, 0, self.social_embedding.num_embeddings - 1)
        giver_indices = torch.clamp(giver_indices, 0, self.giver_embedding.num_embeddings - 1)

        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        social_embedding = self.social_embedding(social_indices)
        giver_embedding = self.giver_embedding(giver_indices)
        
        vector = torch.cat([user_embedding, item_embedding, social_embedding, giver_embedding], dim=-1)
        mlp_output = self.mlp(vector)
        prediction = self.output(mlp_output)
        return prediction.view(-1)