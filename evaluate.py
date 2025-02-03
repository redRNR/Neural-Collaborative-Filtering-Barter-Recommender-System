import torch
import numpy as np
from tqdm import tqdm

def evaluate(model, test_data, num_users, num_items, social_adj, user_item_set, use_giver=True, use_time=True):
    device = model.user_embedding.weight.device
    model.eval()
    all_pos_scores = []
    all_neg_scores = []
    
    auc_scores = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_data))):
            try:
                idx_data = test_data[idx].to(device)
                idx_offset = 0
                if use_giver:
                    giver = int(idx_data[idx_offset])
                    idx_offset += 1
                else:
                    giver = None
                user = int(idx_data[idx_offset])
                idx_offset += 1
                pos_item = int(idx_data[idx_offset])
                idx_offset += 1
                if use_time:
                    time_value = float(idx_data[idx_offset]) if idx_data.size(0) > idx_offset else None
                else:
                    time_value = None

                if social_adj is not None and user < len(social_adj) and len(social_adj[user]) > 0:
                    social_u = int(np.random.choice(social_adj[user]))
                else:
                    social_u = user

                user_tensor = torch.LongTensor([user]).to(device)
                pos_item_tensor = torch.LongTensor([pos_item]).to(device)
                social_tensor = torch.LongTensor([social_u]).to(device)
                giver_tensor = torch.LongTensor([giver]).to(device) if giver is not None else None
                time_tensor = torch.FloatTensor([time_value]).to(device) if time_value is not None else None

                # Generate negative samples
                neg_items = torch.LongTensor(
                    np.random.choice(num_items, size=100, replace=False)
                ).to(device)
                
                if use_giver:
                    neg_givers = torch.LongTensor(
                        np.random.choice(num_users, size=len(neg_items), replace=True)
                    ).to(device)
                else:
                    neg_givers = None

                # Evaluate positive sample
                pos_score = model(user_tensor, pos_item_tensor, 
                                  social_tensor, giver_tensor,
                                  time_tensor).cpu().item()
                
                if np.isnan(pos_score):
                    continue
                    
                # Evaluate negative samples
                neg_scores = model(
                    user_tensor.repeat(len(neg_items)),
                    neg_items,
                    social_tensor.repeat(len(neg_items)),
                    neg_givers,
                    time_tensor.repeat(len(neg_items)) if time_tensor is not None else None
                ).cpu().detach().numpy()
                
                # Skip if we got NaN values
                if np.any(np.isnan(neg_scores)):
                    continue

                all_pos_scores.append(pos_score)
                all_neg_scores.extend(neg_scores)

                # AUC calculation
                auc = np.mean([1 if pos_score > neg_score else 0.5 if pos_score == neg_score else 0 for neg_score in neg_scores])
                auc_scores.append(auc)

            except Exception as e:
                print(f"Error processing test item {idx}: {str(e)}")
                continue

    if not all_pos_scores or not all_neg_scores:
        raise ValueError("No valid scores were generated during evaluation")

    average_auc = np.mean(auc_scores)

    print(f'AUC: {average_auc:.4f}')

    return average_auc
