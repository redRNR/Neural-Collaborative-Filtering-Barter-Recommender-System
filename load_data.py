import numpy as np
import pandas as pd
import torch

def load_data(path, use_social=True, use_time=True, giver=True):
    # Load pairs and wish data
    pairs = pd.read_csv(f'{path}/pairs_dense.csv', header=None).values
    wish = pd.read_csv(f'{path}/wish_dense.csv', header=None).values
    
    # Handle transaction data
    try:
        transac = pd.read_csv(f'{path}/transac_dense.csv', header=None).values
        have = pd.read_csv(f'{path}/have_dense.csv', header=None).values
    except FileNotFoundError:
        pairs = pairs + 1
        wish = wish + 1
        pairs = np.hstack((pairs[:, [0, 1, 2, 3, 4, 4]]))
        transac = np.vstack((pairs[:, [0, 2, 1, 4]], pairs[:, [2, 0, 3, 4]]))
        have = np.empty((0, 2))

    # Find minimum indices for normalization
    min_user_idx = min(
        np.min(pairs[:, [0, 2]]),
        np.min(transac[:, [0, 1]]),
        np.min(wish[:, 0]),
        np.min(have[:, 0]) if have.size > 0 else np.inf
    )
    min_item_idx = min(
        np.min(pairs[:, [1, 3]]),
        np.min(transac[:, 2]),
        np.min(wish[:, 1]),
        np.min(have[:, 1]) if have.size > 0 else np.inf
    )

    # Normalize indices
    pairs[:, [0, 2]] -= min_user_idx
    pairs[:, [1, 3]] -= min_item_idx
    transac[:, [0, 1]] -= min_user_idx
    transac[:, 2] -= min_item_idx
    wish[:, 0] -= min_user_idx
    wish[:, 1] -= min_item_idx
    if have.size > 0:
        have[:, 0] -= min_user_idx
        have[:, 1] -= min_item_idx

    # Calculate dimensions
    num_users = int(max(
        np.max(pairs[:, [0, 2]]),
        np.max(transac[:, [0, 1]]),
        np.max(wish[:, 0]),
        np.max(have[:, 0]) if have.size > 0 else 0
    )) + 1
    num_items = int(max(
        np.max(pairs[:, [1, 3]]),
        np.max(transac[:, 2]),
        np.max(wish[:, 1]),
        np.max(have[:, 1]) if have.size > 0 else 0
    )) + 1

    # Split into train and test sets
    num_test = int(np.floor(len(pairs) / 10))
    perms = np.random.permutation(len(pairs))
    test_pairs = pairs[perms[:num_test]]
    train_pairs = pairs[perms[num_test:]]

    # Process wish data
    if wish.shape[1] != 2:
        raise ValueError("Expected 'wish' to have exactly 2 columns.")
    wish_extended = np.hstack((
        -np.ones((len(wish), 1)),
        wish[:, [0]],
        wish[:, [1]],
        -np.ones((len(wish), 1))
    ))

    # Combine train data and filter test interactions
    train_data = np.vstack((transac, wish_extended))
    test_interactions = {(row[0], row[1]) for row in test_pairs} | {(row[2], row[3]) for row in test_pairs}
    train_data = np.array([row for row in train_data if (int(row[1]), int(row[2])) not in test_interactions])

    # Create social interaction matrix if needed
    if use_social:
        C = np.zeros((num_users, num_users), dtype=np.float32)
        for giver, receiver, *_ in transac:
            if giver != -1:
                C[int(receiver), int(giver)] += 1
        social_adj = [np.nonzero(C[u])[0].astype(int).tolist() for u in range(num_users)]
    else:
        C = None
        social_adj = None

    # Prepare column indices for tensor conversion
    columns = []
    idx = 0
    if giver:
        columns.append(idx)  # Giver
        idx += 1
    columns.extend([idx, idx + 1])  # User, Item
    if use_time:
        columns.append(idx + 2)  # Time

    # Convert to tensors
    train_data_tensor = torch.LongTensor(train_data[:, columns])
    test_data_tensor = torch.LongTensor(test_pairs[:, columns])

    # Handle giver indices
    if giver:
        train_data_tensor[:, 0][train_data_tensor[:, 0] == -1] = num_users - 1
        test_data_tensor[:, 0][test_data_tensor[:, 0] == -1] = num_users

    return train_data_tensor, num_users, num_items, test_data_tensor, social_adj, C