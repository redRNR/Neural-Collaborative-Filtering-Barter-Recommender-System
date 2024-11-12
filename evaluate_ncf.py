import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def hit_rate_at_k(model, test_data, k=10, device='cuda'):
    model.eval()
    hits = 0
    with torch.no_grad():
        for user, true_item in test_data:
            user_tensor = torch.LongTensor([user] * model.item_embedding.num_embeddings).to(device)
            item_tensor = torch.arange(model.item_embedding.num_embeddings).to(device)
            predictions = model(user_tensor, item_tensor)

            # Get top-k items
            top_k_items = torch.topk(predictions, k).indices.cpu().numpy()
            if true_item in top_k_items:
                hits += 1
    hit_rate = hits / len(test_data)
    print(f"Hit Rate@{k}: {hit_rate:.4f}")
    return hit_rate

def auc_score(model, test_data, device='cuda'):
    """
    Calculate the AUC for the model's predictions on test data.
    
    Parameters:
    - model: Trained NCF model
    - test_data: List of (user, item, label) where label is 1 for positive and 0 for negative interactions
    - device: Device to perform computations on (e.g., 'cuda' or 'cpu')
    
    Returns:
    - AUC score for the model on test data.
    """
    model.eval()
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for user, item, label in test_data:
            user_tensor = torch.LongTensor([user]).to(device)
            item_tensor = torch.LongTensor([item]).to(device)
            
            # Prediction score for the (user, item) pair
            score = model(user_tensor, item_tensor).item()
            y_true.append(label)
            y_scores.append(score)
    
    # Calculate AUC score
    auc = roc_auc_score(y_true, y_scores)
    print(f"AUC: {auc:.4f}")
    return auc
