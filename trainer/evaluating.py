import os
import sys
import random
import numpy as np

import scipy.io as sio
import scipy.stats as stats
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import util
from data_loader.seq_util import reverse_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_driver(model, data_loader, metrics, hparams, exp_dir, data_tag):
    eval_config = hparams.evaluating
    # reconstruction
    evaluate_epoch(model, data_loader, metrics, exp_dir, hparams, eval_config, data_tag)


def evaluate_epoch(model, data_loader, metrics, exp_dir, hparams, eval_config, data_tag):
    model.eval()
    total_len = eval_config.get('total_len')
    reverse = eval_config.get('reverse')
    domain = eval_config.get('domain')
    n_steps = 0
    bces, mses, vpts = None, None, None

    recons = None
    grdths = None
    labels = None

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            x, D, x_state, D_state, label = batch
            if total_len is not None:
                x = x[:, :total_len]
                D = D[:, :, :total_len]

            B, T = x.shape[0], x.shape[1]
            K = D.shape[1]
            x = x.to(device)
            D = D.to(device)
            if reverse:
                seq_length = total_len * torch.ones(B).int()
                x_reversed = reverse_sequence(x, seq_length)
                x_reversed = x_reversed.to(device)
                inputs = x_reversed

                inputs_D = torch.zeros_like(D)
                for i in range(K):
                    inputs_D[:, i, :] = reverse_sequence(D[:, i, :], seq_length)
            else:
                inputs = x
                inputs_D = D
            
            if domain:
                x_, D_, mu_c, var_c, mu_t, var_t, mu_0, var_0, D_mu0, D_var0 = model(inputs, inputs_D)
            else:
                x_, mu_0, var_0, mu_c, var_c = model(inputs)
            n_steps += 1
            
            if torch.isnan(x_).any():
                x_ = torch.nan_to_num(x_)

            if idx == 0:
                recons = tensor2np(x_)
                grdths = tensor2np(x)
                labels = tensor2np(label)                    
            else:
                recons = np.concatenate((recons, tensor2np(x_)), axis=0)
                grdths = np.concatenate((grdths, tensor2np(x)), axis=0)
                labels = np.concatenate((labels, tensor2np(label)), axis=0)
            
            for met in metrics:
                if met.__name__ == 'bce':
                    # reconstruction
                    bce = met(x_, x)
                    bce = tensor2np(bce)
                    if idx == 0:
                        bces = bce
                    else:
                        bces = np.concatenate((bces, bce), axis=0)
                if met.__name__ == 'mse':
                    # reconstruction
                    mse = met(x_, x)
                    mse = tensor2np(mse)
                    if idx == 0:
                        mses = mse
                    else:
                        mses = np.concatenate((mses, mse), axis=0)
                if met.__name__ == 'vpt':
                    # reconstruction
                    vpt = met(x_, x)
                    vpt = tensor2np(vpt)
                    if idx == 0:
                        vpts = vpt
                    else:
                        vpts = np.concatenate((vpts, vpt), axis=0)
    for met in metrics:
        if met.__name__ == 'bce':
            print_results(exp_dir, 'bce', bces)
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'vpt':
            print_results(exp_dir, 'vpt', vpts)

    save_result(exp_dir, recons, grdths, labels, data_tag)


def prediction_driver(model, eval_data_loader, pred_data_loader, metrics, hparams, exp_dir, data_tag):
    eval_config = hparams.evaluating
    # reconstruction
    prediction_epoch(model, eval_data_loader, pred_data_loader, metrics, exp_dir, hparams, eval_config, data_tag)


def prediction_epoch(model, eval_data_loader, pred_data_loader, metrics, exp_dir, hparams, eval_config, data_tag):
    model.eval()
    total_len = eval_config.get('total_len')
    reverse = eval_config.get('reverse')
    domain = eval_config.get('domain')
    n_steps = 0
    bces, mses, vpts = None, None, None

    recons = None
    grdths = None
    labels = None

    with torch.no_grad():
        data_iterator = iter(eval_data_loader)
        for idx, batch in enumerate(pred_data_loader):
            x, _, x_state, _, label = batch

            try:
                eval_batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(eval_data_loader)
                eval_batch = next(data_iterator)
            _, D, _, D_state, label = eval_batch

            if total_len is not None:
                x = x[:, :total_len]
                D = D[:, :, :total_len]

            B, T = x.shape[0], x.shape[1]
            K = D.shape[1]
            x = x.to(device)
            D = D.to(device)
            if reverse:
                seq_length = total_len * torch.ones(B).int()
                x_reversed = reverse_sequence(x, seq_length)
                x_reversed = x_reversed.to(device)
                inputs = x_reversed

                inputs_D = torch.zeros_like(D)
                for i in range(K):
                    inputs_D[:, i, :] = reverse_sequence(D[:, i, :], seq_length)
            else:
                inputs = x
                inputs_D = D
            
            if domain:
                x_, D_, mu_c, var_c, mu_t, var_t, mu_0, var_0, D_mu0, D_var0 = model(inputs, inputs_D)
            else:
                x_, mu_0, var_0, mu_c, var_c = model(inputs)
            n_steps += 1
            
            if torch.isnan(x_).any():
                x_ = torch.nan_to_num(x_)

            if idx == 0:
                recons = tensor2np(x_)
                grdths = tensor2np(x)
                labels = tensor2np(label)                    
            else:
                recons = np.concatenate((recons, tensor2np(x_)), axis=0)
                grdths = np.concatenate((grdths, tensor2np(x)), axis=0)
                labels = np.concatenate((labels, tensor2np(label)), axis=0)
            
            for met in metrics:
                if met.__name__ == 'bce':
                    # reconstruction
                    bce = met(x_, x)
                    bce = tensor2np(bce)
                    if idx == 0:
                        bces = bce
                    else:
                        bces = np.concatenate((bces, bce), axis=0)
                if met.__name__ == 'mse':
                    # reconstruction
                    mse = met(x_, x)
                    mse = tensor2np(mse)
                    if idx == 0:
                        mses = mse
                    else:
                        mses = np.concatenate((mses, mse), axis=0)
                if met.__name__ == 'vpt':
                    # reconstruction
                    vpt = met(x_, x)
                    vpt = tensor2np(vpt)
                    if idx == 0:
                        vpts = vpt
                    else:
                        vpts = np.concatenate((vpts, vpt), axis=0)
    for met in metrics:
        if met.__name__ == 'bce':
            print_results(exp_dir, 'bce', bces)
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'vpt':
            print_results(exp_dir, 'vpt', vpts)

    save_result(exp_dir, recons, grdths, labels, data_tag)


def print_results(exp_dir, met_name, mets):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')
    
    print('{} for seq = {:05.5f}'.format(met_name, mets.mean()))
    with open(os.path.join(exp_dir, 'data/metric.txt'), 'a+') as f:
        f.write('{} for seq = {}\n'.format(met_name, mets.mean()))


def save_result(exp_dir, recons, inputs, labels, data_tag):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')

    sio.savemat(os.path.join(exp_dir, 'data/{}.mat'.format(data_tag)), {'recons': recons, 'inputs': inputs, 'label': labels})


def tensor2np(t):
    return t.cpu().detach().numpy()
