import scipy.stats as stats
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from skimage.filters import threshold_otsu


def mse(output, target):
    mse = F.mse_loss(output, target, reduction='none')
    return mse


def tcc(u, x):
    m, n, w = u.shape
    res = []
    for i in range(m):
        correlation_sum = 0
        count = 0
        for j in range(n):
            a = u[i, j, :]
            b = x[i, j, :]
            if (a == a[0]).all() or (b == b[0]).all():
                count += 1
                continue
            correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
        correlation_sum = correlation_sum / (n - count)
        res.append(correlation_sum)
    res = np.array(res)
    return res


def scc(u, x):
    m, n, w = u.shape
    res = []
    for i in range(m):
        correlation_sum = 0
        count = 0
        for j in range(w):
            a = u[i, :, j]
            b = x[i, :, j]
            if (a == a[0]).all() or (b == b[0]).all():
                count += 1
                continue
            correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
        correlation_sum = correlation_sum / (w - count)
        res.append(correlation_sum)
    res = np.array(res)
    return res


def dcc(u, x):
    m, n, w = u.shape
    dice_cc = []

    # u_apd = np.sum(u > 0.03, axis=2)
    # u_scar = u_apd > 0.25 * w

    # x_apd = np.sum(x > 0.04, axis=2)
    # x_scar = x_apd > 0.25 * w

    for i in range(m):
        u_row = u[i, :, 50]
        x_row = x[i, :, 50]

        thresh_u = threshold_otsu(u_row)
        u_scar_idx = np.where(u_row >= thresh_u)[0]
        thresh_x = threshold_otsu(x_row)
        x_scar_idx = np.where(x_row >= thresh_x)[0]

        intersect = set(u_scar_idx) & set(x_scar_idx)
        dice_cc.append(2 * len(intersect) / float(len(set(u_scar_idx)) + len(set(x_scar_idx))))

    dice_cc = np.array(dice_cc)
    return dice_cc
