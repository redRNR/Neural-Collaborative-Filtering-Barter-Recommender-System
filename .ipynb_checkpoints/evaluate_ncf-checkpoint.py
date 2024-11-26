
import torch
import numpy as np

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
