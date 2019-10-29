import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cf


def pred_loss(predictions, labels, weight=1.0):
    criterion = nn.CrossEntropyLoss()
    cost = criterion(predictions, labels)

    # softmax_result = F.log_softmax(predictions, dim=1)
    # cost = torch.mean(-1*softmax_result[labels==1])

    cost *= weight
    return cost

def consistency_loss(coarse_outputs, t_k, weight=1.0):
    B_ik = torch.mean(coarse_outputs, 0)
    cost = torch.sum((t_k-B_ik)**2)*weight
    # cost = torch.sqrt(torch.sum(torch.abs(t_k - B_ik)))*weight

    return cost