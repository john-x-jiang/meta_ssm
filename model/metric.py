import time
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from model.loss import nll_loss, kl_div


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def vpt(output, target):
    epsilon = 0.025
    B, T = target.shape[0], target.shape[1]
    W, H = target.shape[2], target.shape[3]
    mse = F.mse_loss(output, target, reduction='none')
    mse_m = mse.sum(axis=[2, 3]) / (W * H)
    vpt = torch.zeros(B).to(mse.device)
    for i in range(B):
        vpt[i] = torch.min(torch.where(mse_m[i, :] < epsilon)[0])
    vpt = vpt / T
    return vpt


def bce(output, target):
    bce = F.binary_cross_entropy(output, target, reduction='none')
    return bce


def mse(output, target):
    mse = F.mse_loss(output, target, reduction='none')
    return mse
