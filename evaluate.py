import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as roc_auc_score

def evaluate(model, test_data, num_users, num_items, social_adj, user_item_set):
    model.eval()
    all_pos_scores = []
    all_neg_scores = []
    auc_scores = []

    for idx in tqdm(range(len(test_data))):
        idx_data = test_data[idx]
        idx = 0
        giver = int(idx_data[idx]) if idx_data.size(0) >= 3 else None
        if giver is not None:
            idx += 1
        user = int(idx_data[idx])
        idx += 1
        pos_item = int(idx_data[idx])
        idx += 1

        if user < len(social_adj) and len(social_adj[user]) > 0:
            social_u = int(np.random.choice(social_adj[user]))
        else:
            social_u = user

        user_tensor = torch.LongTensor([user])
        pos_item_tensor = torch.LongTensor([pos_item])
        social_tensor = torch.LongTensor([social_u])
        giver_tensor = torch.LongTensor([giver]) if giver is not None else None

        neg_items = np.random.choice(num_items, size=100, replace=False)
        neg_items = [int(item) for item in neg_items if (user, item) not in user_item_set]
        neg_items_tensor = torch.LongTensor(neg_items)

        if giver is not None:
            neg_givers = np.random.choice(num_users, size=len(neg_items), replace=True)
            neg_givers_tensor = torch.LongTensor(neg_givers)
        else:
            neg_givers_tensor = None

        pos_score = model(user_tensor, pos_item_tensor, social_tensor, giver_tensor).item()
        neg_scores = model(
            user_tensor.repeat(len(neg_items_tensor)),
            neg_items_tensor,
            social_tensor.repeat(len(neg_items_tensor)),
            neg_givers_tensor
        ).detach().numpy()

        all_pos_scores.append(pos_score)
        all_neg_scores.extend(neg_scores)

        auc = np.mean([1 if pos_score > neg_score else 0 for neg_score in neg_scores])
        auc_scores.append(auc)

    average_auc = np.mean(auc_scores)
    print(f'AUC: {average_auc:.4f}')

    y_true = np.array([1] * len(all_pos_scores) + [0] * len(all_neg_scores))
    y_scores = np.array(all_pos_scores + all_neg_scores)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {average_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve{average_auc:.4f}.png')
    plt.close()

    return average_auc