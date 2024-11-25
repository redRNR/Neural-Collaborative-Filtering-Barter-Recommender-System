import torch

def bpr_loss(model, user_indices, pos_item_indices, neg_item_indices):
    pos_scores = model(user_indices, pos_item_indices)
    neg_scores = model(user_indices, neg_item_indices)
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
    return loss
