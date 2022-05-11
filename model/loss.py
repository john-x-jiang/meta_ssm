import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def kl_div(mu1, var1, mu2=None, var2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if var2 is None:
        var2 = torch.zeros_like(mu1)

    # return 0.5 * (
    #     var2.log() - var1.log() + (
    #         var1 + (mu1 - mu2).pow(2)
    #     ) / var2 - 1)
    return 0.5 * (
        var2 - var1 + (
            torch.exp(var1) + (mu1 - mu2).pow(2)
        ) / torch.exp(var2) - 1)


def kl_div_stn(mu, logvar):
    return 0.5 * (
        mu.pow(2) + torch.exp(logvar) - logvar - 1
    )


def kl_div_with_log(log1, log2):
    # return log1 - log2
    q = log1.exp()
    return q * (log1 - log2)


def nll_loss(x_hat, x, loss_type='bce'):
    assert x_hat.dim() == x.dim()
    assert x.size() == x_hat.size()
    if loss_type == 'bce':
        return nn.BCELoss(reduction='none')(x_hat, x)
    elif loss_type == 'mse':
        return nn.MSELoss(reduction='none')(x_hat, x)
    else:
        raise NotImplemented


def meta_loss(x, x_, mu_c, var_c, mu_t, var_t, mu_0, var_0, kl_factor, loss_type='mse', obs_len=10, r1=1, r2=1, r3=1):
    # likelihood
    B, T = x.shape[0], x.shape[1]
    nll_raw = nll_loss(x_, x, loss_type)
    nll_0 = nll_raw[:, 0, :].sum() / B
    nll_r = nll_raw[:, 1:obs_len, :].sum() / B / (obs_len - 1)
    # if obs_len < T:
    #     nll_g = nll_raw[:, obs_len:, :].sum() / B / (T - obs_len)
    # else:
    #     nll_g = torch.zeros_like(nll_0)
    nll_m = nll_0 + nll_r

    likelihood = nll_m

    # domain condition
    kl_raw_c = kl_div_stn(mu_c, var_c)
    kl_m_c = kl_raw_c.sum() / B

    kl_raw_c_t = kl_div(mu_c, var_c, mu_t, var_t)
    kl_m_c_t = kl_raw_c_t.sum() / B

    # initial condition
    kl_raw_0 = kl_div_stn(mu_0, var_0)
    kl_m_0 = kl_raw_0.sum() / B

    kl_initial = kl_m_0

    loss = (r1 * kl_initial + r2 * kl_m_c + r3 * kl_m_c_t) * kl_factor + likelihood

    return kl_m_c, kl_m_c_t, kl_initial, likelihood, loss


def dmm_loss(x, x_, mu_0, var_0, mu_c, var_c, kl_factor, loss_type='mse', r1=1, r2=1):
    # likelihood
    B, T = x.shape[0], x.shape[1]
    nll_raw = nll_loss(x_, x, loss_type)
    likelihood = nll_raw[:, 0, :].sum() / B + nll_raw[:, 1:, :].sum() / B / (T - 1)

    # initial condition
    kl_raw_0 = kl_div_stn(mu_0, var_0)
    kl_initial = kl_raw_0.sum() / B

    # domain condition
    kl_raw_c = kl_div_stn(mu_c, var_c)
    kl_m_c = kl_raw_c.sum() / B

    loss = (r1 * kl_initial + r2 * kl_m_c) * kl_factor + likelihood

    return kl_m_c, kl_initial, likelihood, loss
