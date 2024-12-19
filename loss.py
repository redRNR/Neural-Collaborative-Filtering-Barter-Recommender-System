import torch
import torch.nn.functional as F
from typing import Optional

class BPRLoss:
    def __init__(self, social_adj=None):
        self.social_adj = social_adj

    def __call__(self, model, user_indices, pos_item_indices, neg_item_indices, 
                 social_indices, giver_indices=None, neg_giver_indices=None, time_values=None):
        # Get positive predictions
        pos_pred = model(user_indices, pos_item_indices, social_indices, giver_indices, time_values)
        
        # Get negative predictions
        neg_pred = model(user_indices, neg_item_indices, social_indices, 
                        neg_giver_indices if giver_indices is not None else None,
                        time_values)

        # Compute BPR loss
        return -(pos_pred - neg_pred).sigmoid().log().mean()