import time
import scipy.stats as stats
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


def cc(output, target):
    B, T, N = target.shape
    res = []
    for i in range(B):
        correlation_sum = 0
        for j in range(N):
            a = output[i, :, j]
            b = target[i, :, j]
            if (a == a[0]).all() or (b == b[0]).all():
                correlation_sum = correlation_sum + 0
            else:
                correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
        correlation_sum = correlation_sum / n
        res.append(correlation_sum)
    res = np.array(res)
    return res


def vpt(output, target):
    epsilon = 0.02
    B, T = target.shape[0], target.shape[1]
    W, H = target.shape[2], target.shape[3]
    mse = F.mse_loss(output, target, reduction='none')
    mse_m = mse.sum(axis=[2, 3]) / (W * H)
    vpt = torch.zeros(B).to(mse.device)
    for i in range(B):
        idx = torch.where(mse_m[i, :] >= epsilon)[0]
        if idx.shape[0] > 0:
            vpt[i] = torch.min(idx)
        else:
            vpt[i] = 0
    vpt = vpt / T
    return vpt


def bce(output, target):
    bce = F.binary_cross_entropy(output, target, reduction='none')
    return bce


def mse(output, target):
    mse = F.mse_loss(output, target, reduction='none')
    return mse
