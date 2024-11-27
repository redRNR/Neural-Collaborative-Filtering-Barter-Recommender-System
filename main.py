# main.py

import torch
from load_data import load_data
from ncf import NCF
from train_ncf import train_ncf

path = 'data/ratebeer/'
print(f'Loading data from {path}')
use_social = True
use_time = True
giver = True

# Load data
train_data, num_users, num_items, test_data, social_adj, C = load_data(
    path, use_social=use_social, use_time=use_time, giver=giver
)
print('Data loaded')

# Instantiate the NCF model
embedding_dim = 64
model = NCF(num_users, num_items, embedding_dim=embedding_dim)
print('Model instantiated')

# Training using Neural Collaborative Filtering
epochs = 10
batch_size = 1024
learning_rate = 0.001

train_ncf(model, train_data, social_adj, num_users, num_items, epochs=epochs, batch_size=batch_size, lr=learning_rate)
print('Training completed')

# Prepare user-item set for evaluation
user_item_set = set(zip(train_data[:, 1].numpy(), train_data[:, 2].numpy()))

# Evaluate the model
from evaluate import evaluate
evaluate(model, test_data, num_users, num_items, social_adj, user_item_set)
