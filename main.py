# main.py
import torch
from load_data import load_data
from ncf import NCF
from train_ncf import train_ncf

# Add device selection at the start
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

path = 'data/ratebeer/'
print(f'Loading data from {path}')
use_social = True
use_time = True
giver = True

# Load data remains the same
train_data, num_users, num_items, test_data, social_adj, C = load_data(
    path, use_social=use_social, use_time=use_time, giver=giver
)
print('Data loaded')

# Move data to device after loading
train_data = train_data.to(device)
test_data = test_data.to(device)

# Instantiate and move model to device
for embedding_dims in [50]:
    model = NCF(num_users, num_items, embedding_dim=embedding_dims)
    model = model.to(device)
    print(f'Model instantiated with embedding_dims={embedding_dims} on {device}')

    # Rest remains the same
    epochs = 10
    batch_size = 256
    learning_rate = 0.001
    batch_sizes = [512]
    for batch_size in batch_sizes:
        print(f'Training with batch_size={batch_size}')
        train_ncf(model, train_data, social_adj, num_users, num_items, 
                 epochs=epochs, batch_size=batch_size, lr=learning_rate)
        print('Training completed')

        user_item_set = set(zip(train_data[:, 1].cpu().numpy(), 
                              train_data[:, 2].cpu().numpy()))

        from evaluate import evaluate
        evaluate(model, test_data, num_users, num_items, social_adj, user_item_set)
