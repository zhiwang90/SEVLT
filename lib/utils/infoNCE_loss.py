import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def infonce_loss(feat_q, feat_k, temperature=0.02):

    B, L, D = feat_q.shape

    feat_q = feat_q.mean(dim=1)
    feat_k = feat_k.mean(dim=1)
    feat_q = F.normalize(feat_q, dim=-1)
    feat_k = F.normalize(feat_k, dim=-1)

    logits = torch.matmul(feat_q, feat_k.t())
    logits /= temperature

    labels = torch.arange(B, device=logits.device)
    loss = F.cross_entropy(logits, labels)


    return loss