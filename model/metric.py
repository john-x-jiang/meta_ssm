import time
import scipy.stats as stats
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from model.loss import nll_loss, kl_div
from skimage.filters import threshold_otsu
import numpy as np


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
            vpt[i] = T
    vpt = vpt / T
    return vpt


def bce(output, target):
    bce = F.binary_cross_entropy(output, target, reduction='none')
    return bce


def mse(output, target):
    mse = F.mse_loss(output, target, reduction='none')
    return mse


def dst(output, target):
    N, T = target.shape[0], target.shape[1]
    output, target = thresholding(output, target)
    results = np.zeros([N, T])
    for n in range(N):
        for t in range(T):
            a = output[n, t]
            b = target[n, t]
            pos_a = np.where(a == 1)
            pos_b = np.where(b == 1)

            if pos_b[0].shape[0] == 0:
                results[n, t] = 0
                continue
            center_b = [pos_b[0].mean(), pos_b[1].mean()]
            if pos_a[0].shape[0] != 0:
                center_a = [pos_a[0].mean(), pos_a[1].mean()]
            else:
                center_a = [0, 0]
            center_a = np.array(center_a)
            center_b = np.array(center_b)
            dist = np.sum((center_a - center_b)**2)
            dist = np.sqrt(dist)
            results[n, t] = dist
    return results


def vpd(output, target):
    epsilon = 10
    mses = dst(output, target)
    B, T = mses.shape
    vpt = np.zeros(B)
    for i in range(B):
        idx = np.where(mses[i, :] >= epsilon)[0]
        if idx.shape[0] > 0:
            vpt[i] = np.min(idx)
        else:
            vpt[i] = T
    vpt = vpt / T
    return vpt


def thresholding(output, target):
    target[:, :, 0, :] = 0
    target[:, :, :, 0] = 0
    output[:, :, 0, :] = 0
    output[:, :, :, 0] = 0
    N, T = target.shape[0], target.shape[1]
    res = np.zeros_like(output)
    for n in range(N):
        for t in range(T):
            img = output[n, t]
            otsu_th = np.max([0.27, threshold_otsu(img)])
            res[n, t] = (img > otsu_th).astype(np.float32)
    return res, target
