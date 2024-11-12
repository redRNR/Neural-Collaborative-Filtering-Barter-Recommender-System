
import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=64, layers=[128, 64, 32, 16], dropout=0.2):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        # Neural network layers
        layer_sizes = [latent_dim * 2] + layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(len(layers))
        ])
        self.output = nn.Linear(layer_sizes[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # ncf_model.py, inside the forward method
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)

        x = torch.cat([user_embedding, item_embedding], dim=-1)

        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.output(x)).squeeze()
