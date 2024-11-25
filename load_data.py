# load_data.py

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

def load_data(path, use_social=True, use_time=True, giver=True):
    # Load pairs and wish data
    pairs = pd.read_csv(f'{path}/pairs_dense.csv', header=None).values
    wish = pd.read_csv(f'{path}/wish_dense.csv', header=None).values

    # Try to load transaction data; if not available, process accordingly
    try:
        transac = pd.read_csv(f'{path}/transac_dense.csv', header=None).values
        have = pd.read_csv(f'{path}/have_dense.csv', header=None).values
    except FileNotFoundError:
        # Handle case where only wish and pairs data are available
        pairs = pairs + 1  # Adjust if needed
        wish = wish + 1
        pairs = np.hstack((pairs[:, [0, 1, 2, 3, 4, 4]]))
        transac = np.vstack((pairs[:, [0, 2, 1, 4]], pairs[:, [2, 0, 3, 4]]))
        have = np.empty((0, 2))

    transac_out = transac.copy()

    # Adjust indices to start from 0
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

    pairs[:, [0, 2]] -= min_user_idx
    pairs[:, [1, 3]] -= min_item_idx
    transac[:, [0, 1]] -= min_user_idx
    transac[:, 2] -= min_item_idx
    wish[:, 0] -= min_user_idx
    wish[:, 1] -= min_item_idx
    if have.size > 0:
        have[:, 0] -= min_user_idx
        have[:, 1] -= min_item_idx

    # Recalculate num_users and num_items after adjustment
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

    # Split test and train sets
    num_test = int(np.floor(len(pairs) / 10))
    perms = np.random.permutation(len(pairs))
    test_rows = perms[:num_test]
    train_rows = perms[num_test:]
    test_pairs = pairs[test_rows, :]
    train_pairs = pairs[train_rows, :]

    # Prepare training data
    # Giver - Receiver - Item - Time (if available)
    # Assuming transac has columns: Giver, Receiver, Item, Time
    train_transac = transac.copy()

    # Ensure that wish has two columns: User and Item
    if wish.shape[1] != 2:
        raise ValueError("Expected 'wish' to have exactly 2 columns.")

    # For wish data, set Giver to -1 and Time to -1
    num_wish = len(wish)
    wish_extended = np.hstack((
        -np.ones((num_wish, 1)),  # Giver as -1
        wish[:, [0]],             # Receiver (User)
        wish[:, [1]],             # Item
        -np.ones((num_wish, 1))   # Time as -1
    ))

    # Combine transaction and wish data
    train_data = np.vstack((train_transac, wish_extended))

    # Remove test pairs from training data
    # Create a set of test interactions to filter out
    test_interactions = set()
    for row in test_pairs:
        test_interactions.add((row[0], row[1]))
        test_interactions.add((row[2], row[3]))

    train_interactions = []
    for row in train_data:
        user = int(row[1])
        item = int(row[2])
        if (user, item) not in test_interactions:
            train_interactions.append(row)
    train_data = np.array(train_interactions)

    # Build the social interaction matrix C from transaction data
    # C[u, v] represents the number of times user u received items from user v
    C = np.zeros((num_users, num_users), dtype=np.float32)
    for row in transac:
        giver = int(row[0])
        receiver = int(row[1])
        if giver != -1:
            C[receiver, giver] += 1

    # Prepare train_data tensor
    # Depending on whether 'giver' and 'use_time' are True, include those columns
    columns = []
    idx = 0
    if giver:
        columns.append(idx)  # Giver
        idx += 1
    columns.append(idx)  # User
    idx += 1
    columns.append(idx)  # Item
    idx += 1
    if use_time:
        columns.append(idx)  # Time

    train_data_tensor = torch.LongTensor(train_data[:, columns])

    # Replace -1 in giver_indices with num_users - 1
    if giver:
        giver_idx_in_train = 0  # Giver is at position 0
        giver_indices_train = train_data_tensor[:, giver_idx_in_train]
        giver_indices_train[giver_indices_train == -1] = num_users - 1
        train_data_tensor[:, giver_idx_in_train] = giver_indices_train

    # If use_social is True, create social adjacency list from C matrix
    if use_social:
        social_adj = [[] for _ in range(num_users)]
        for u in range(num_users):
            friends = np.nonzero(C[u])[0].astype(int).tolist()
            social_adj[u].extend(friends)
    else:
        social_adj = None

    # Prepare test_data tensor
    test_data_columns = []
    idx = 0
    if giver:
        test_data_columns.append(idx)  # Giver
        idx += 1
    test_data_columns.append(idx)  # User
    idx += 1
    test_data_columns.append(idx)  # Item
    idx += 1
    if use_time:
        test_data_columns.append(idx)  # Time

    test_data_tensor = torch.LongTensor(test_pairs[:, test_data_columns])

    # Replace -1 in giver_indices with num_users in test_data_tensor
    if giver:
        giver_idx_in_test = 0  # Giver is at position 0
        giver_indices_test = test_data_tensor[:, giver_idx_in_test]
        giver_indices_test[giver_indices_test == -1] = num_users
        test_data_tensor[:, giver_idx_in_test] = giver_indices_test

    return train_data_tensor, num_users, num_items, test_data_tensor, social_adj, C
