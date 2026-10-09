import torch
import torch.nn.functional as F

def kl_loss(pred_logits, target_logits, temperature=1.0, eps=1e-8):

        pred_log_prob = F.log_softmax(pred_logits / temperature, dim=-1)
        target_prob = F.softmax(target_logits / temperature, dim=-1).clamp(min=eps)
        loss = F.kl_div(pred_log_prob, target_prob, reduction='batchmean')


        return loss
