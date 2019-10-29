import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cf


def pred_loss(predictions, labels, weight=1.0, mode='focus'):
    # if mode == 'focus':
    #     softmax_result = F.log_softmax(predictions, dim=1)
    #     cost = torch.mean(-1*softmax_result[labels==1])
    # else:
    #     assert mode == 'full'
    #     softmax_result = F.softmax(predictions, dim=1)
    #     cost = -1 * torch.mean(torch.cat((torch.log(softmax_result[labels == 1]),torch.log(1-softmax_result[labels == 0])), 0))

    # softmax_result = F.log_softmax(predictions, dim=1)
    # cost = torch.mean(-1*softmax_result[labels==1])
    # softmax_result = F.softmax(predictions, dim=1)
    softmax_result = predictions
    cost = -1 * torch.mean(torch.cat((torch.log(softmax_result[labels == 1]), torch.log(1 - softmax_result[labels == 0])), 0))

    cost *= weight
    return cost

def fine_tuning_loss(predictions, labels, t_k, B_o_ik, args,lam=20, weight=1.0):
    for k in range (args.num_superclass):
        for i in range (np.shape(predictions)[0]):
            t_k[k] -= B_o_ik[i,k]/(np.shape(predictions)[0])
        t_k[k] = (t_k[k]*t_k[k])*(lam/2)
    softmax_result = predictions
    criterion = nn.CrossEntropyLoss()
    cost = criterion(predictions, labels)
    cost += torch.sum(t_k)
    cost *= weight
    return cost


def recon_loss(reconstructions, raw_data, weight=1.0):
    cost = F.mse_loss(reconstructions, raw_data.float(), reduction='sum')
    cost *= weight
    return cost


def diverter_loss(express_vec, centers, weight=1.0): #fea_vec, center
    cost = F.mse_loss(express_vec, centers, reduction='sum')
    cost *= weight
    return cost


def center_loss(express_vec, centers, weight=1.0):
    cost = F.mse_loss(express_vec, centers, reduction='sum')
    cost *= weight
    return cost


def triplet_loss(express_vec, labels, weight=1.0):
    _, labels = torch.max(labels, 1)
    anchor, positives, negatives = get_triplets(express_vec, labels)
    num_samples = anchor.shape[0]
    y = torch.ones((num_samples, 1)).view(-1)
    if anchor.is_cuda: y = y.cuda()
    ap_dist = torch.norm(anchor - positives, 2, dim=1).view(-1)
    an_dist = torch.norm(anchor - negatives, 2, dim=1).view(-1)
    cost = F.soft_margin_loss(an_dist - ap_dist, y)
    cost *= weight
    return cost

def get_triplets(embeds, labels):
    dist_mtx = pdist(embeds, embeds).detach().cpu().numpy()
    labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
    num = labels.shape[0]
    dia_inds = np.diag_indices(num)
    lb_eqs = labels == labels.T
    lb_eqs[dia_inds] = False
    dist_same = dist_mtx.copy()
    dist_same[lb_eqs == False] = -np.inf
    pos_idxs = np.argmax(dist_same, axis = 1)
    dist_diff = dist_mtx.copy()
    lb_eqs[dia_inds] = True
    dist_diff[lb_eqs == True] = np.inf
    neg_idxs = np.argmin(dist_diff, axis = 1)
    pos = embeds[pos_idxs].contiguous().view(num, -1)
    neg = embeds[neg_idxs].contiguous().view(num, -1)
    return embeds, pos, neg

def pdist(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx