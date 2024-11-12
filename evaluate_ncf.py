import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def auc_score(model, interactions_df, device='cuda'):
    """
    Calculate the AUC for the model's predictions on test data using a percentage split.
    
    Parameters:
    - model: Trained NCF model
    - interactions_df: Full DataFrame of interactions
    - device: Device to perform computations on
    """
    model.eval()
    y_true = []
    y_scores = []
    
    # Calculate number of test samples (10% like MATLAB version)
    num_interactions = len(interactions_df)
    num_test = int(np.floor(num_interactions * 0.1))
    
    # Create test set using random permutation
    indices = np.random.permutation(num_interactions)
    test_indices = indices[:num_test]
    test_interactions = interactions_df.iloc[test_indices]
    
    # Create positive test examples from actual interactions
    with torch.no_grad():
        for _, row in test_interactions.iterrows():
            user = int(row['user'])
            item = int(row['item'])
            
            # Positive example
            user_tensor = torch.LongTensor([user]).to(device)
            item_tensor = torch.LongTensor([item]).to(device)
            score = model(user_tensor, item_tensor).item()
            
            y_true.append(1)
            y_scores.append(score)
            
            # Generate negative example for same user
            neg_item = np.random.randint(model.item_embedding.num_embeddings)
            while interactions_df[(interactions_df['user'] == user) & 
                                (interactions_df['item'] == neg_item)].shape[0] > 0:
                neg_item = np.random.randint(model.item_embedding.num_embeddings)
                
            item_tensor = torch.LongTensor([neg_item]).to(device)
            neg_score = model(user_tensor, item_tensor).item()
            
            y_true.append(0)
            y_scores.append(neg_score)
    
    # Calculate AUC score
    auc = roc_auc_score(y_true, y_scores)
    print(f"AUC: {auc:.4f}")
    print(f"Number of test samples: {len(y_true)}")
    return auc

def hit_rate_at_k(model, interactions_df, k=10, device='cuda'):
    """
    Calculate Hit Rate@K using percentage split.
    """
    model.eval()
    hits = 0
    num_test = int(np.floor(len(interactions_df) * 0.1))
    test_interactions = interactions_df.sample(n=num_test)
    
    with torch.no_grad():
        for _, row in test_interactions.iterrows():
            user = int(row['user'])
            true_item = int(row['item'])
            
            user_tensor = torch.LongTensor([user] * model.item_embedding.num_embeddings).to(device)
            item_tensor = torch.arange(model.item_embedding.num_embeddings).to(device)
            predictions = model(user_tensor, item_tensor)
            
            # Get top-k items
            top_k_items = torch.topk(predictions, k).indices.cpu().numpy()
            if true_item in top_k_items:
                hits += 1
    
    hit_rate = hits / num_test
    print(f"Hit Rate@{k}: {hit_rate:.4f}")
    print(f"Number of test samples: {num_test}")
    return hit_rate