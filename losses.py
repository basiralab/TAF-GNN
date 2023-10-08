import torch.nn.functional as F
import torch.nn as nn


def alignment_loss(target, predicted):
    kl_loss = nn.KLDivLoss(reduction='sum', log_target=True)
    inp = F.log_softmax(predicted, dim=1)
    target = F.log_softmax(target, dim=1)
    loss = kl_loss(inp, target)
    return loss
