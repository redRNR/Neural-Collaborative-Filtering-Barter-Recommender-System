import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[64, 32, 16, 8], use_time=False):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=num_users)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=num_users)
        self.social_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=num_users)
        self.giver_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=num_users)
        self.use_time = use_time
        
        # Only create time layer if we're using time
        if use_time:
            self.time_linear = nn.Linear(1, embedding_dim)
            input_size = embedding_dim * 5 
        else:
            self.time_linear = None
            input_size = embedding_dim * 4 
        
        mlp_layers = []
        for hidden_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            input_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)
        self.output = nn.Linear(hidden_layers[-1], 1)

    def forward(self, user_indices, item_indices, social_indices=None, giver_indices=None, time_values=None):
        device = self.user_embedding.weight.device
        user_indices = user_indices.to(device)
        item_indices = item_indices.to(device)
        
        if social_indices is None:
            social_indices = user_indices
        if giver_indices is None:
            giver_indices = user_indices
        
        social_indices = social_indices.to(device)
        giver_indices = giver_indices.to(device)
        if time_values is not None:
            time_values = time_values.to(device)

        # Clamp indices to valid ranges
        user_indices = torch.clamp(user_indices, 0, self.user_embedding.num_embeddings - 1)
        item_indices = torch.clamp(item_indices, 0, self.item_embedding.num_embeddings - 1)
        social_indices = torch.clamp(social_indices, 0, self.social_embedding.num_embeddings - 1)
        giver_indices = torch.clamp(giver_indices, 0, self.giver_embedding.num_embeddings - 1)

        # Compute embeddings
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        social_embedding = self.social_embedding(social_indices)
        giver_embedding = self.giver_embedding(giver_indices)
        
        # Handle time if we're using it
        if self.use_time and time_values is not None:
            time_values = time_values.float().unsqueeze(-1)
            time_embedding = self.time_linear(time_values)
            vector = torch.cat([user_embedding, item_embedding, social_embedding, giver_embedding, time_embedding], dim=-1)
        else:
            vector = torch.cat([user_embedding, item_embedding, social_embedding, giver_embedding], dim=-1)
            
        mlp_output = self.mlp(vector)
        prediction = self.output(mlp_output)
        return prediction.view(-1)