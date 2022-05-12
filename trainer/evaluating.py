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

            B, T = x.shape[0], x.shape[1]
            x = x.to(device)
            
            if domain:
                if total_len is not None:
                    D = D[:, :, :total_len]
                D = D.to(device)
                x_ = model.prediction(x, D)
            else:
                x_, mu_0, var_0, mu_c, var_c = model(x)
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
    domain = eval_config.get('domain')
    batch_size = eval_config.get('batch_size')
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

            B, T = x.shape[0], x.shape[1]
            x = x.to(device)
            
            if domain:
                if total_len is not None:
                    D = D[:, :, :total_len]
                D = D.to(device)
                if x.shape[0] < batch_size:
                    D = D[:x.shape[0], :]
                x_ = model.prediction(x, D)
            else:
                x_, mu_0, var_0, mu_c, var_c = model(D)
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
