import os
import sys
import random
import numpy as np
import time

import scipy.io as sio
import scipy.stats as stats
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_driver(model, data_loaders, metrics, hparams, exp_dir, data_tags):
    eval_config = hparams.evaluating
    loss_func = hparams.loss

    evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, data_tags, eval_config, loss_func=loss_func)


def evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, data_tags, eval_config, loss_func=None):
    torso_len = eval_config['torso_len']
    signal_source = eval_config['signal_source']
    omit = eval_config['omit']
    window = eval_config.get('window')
    k_shot = eval_config.get('k_shot')
    changable = eval_config.get('changable')
    data_scaler = eval_config.get('data_scaler')
    model.eval()
    n_data = 0
    data_idx = 0
    total_time = 0
    mses = None
    tccs = None
    sccs = None
    dccs = None

    with torch.no_grad():
        data_names = list(data_loaders.keys())
        for data_name in data_names:
            data_loader_tag = data_loaders[data_name]
            eval_tags = list(data_loader_tag.keys())

            for tag_idx, eval_tag in enumerate(eval_tags):
                recons, grnths, labels = None, None, None

                data_loader = data_loader_tag[eval_tag]
                len_epoch = len(data_loader)
                n_data += len_epoch * data_loader.batch_size
                
                for idx, data in enumerate(data_loader):
                    signal, label = data.x, data.y
                    signal = signal.to(device)
                    label = label.to(device)
                    
                    if window is not None:
                        signal = signal[:, :, :window]
                    
                    if data_scaler is not None:
                        signal = data_scaler * signal

                    x_heart = signal[:, :-torso_len, omit:]
                    x_torso = signal[:, -torso_len:, omit:]

                    if signal_source == 'heart':
                        source = x_heart
                    elif signal_source == 'torso':
                        source = x_torso

                    physics_vars, statistic_vars = model(source, label, data_name)
                    x_ = physics_vars[0]

                    if idx == 0:
                        recons = tensor2np(x_)
                        grnths = tensor2np(source)
                        labels = tensor2np(label)
                    else:
                        recons = np.concatenate((recons, tensor2np(x_)), axis=0)
                        grnths = np.concatenate((grnths, tensor2np(source)), axis=0)
                        labels = np.concatenate((labels, tensor2np(label)), axis=0)

                    for met in metrics:
                        if met.__name__ == 'mse':
                            mse = met(x_, source)
                            mse = mse.mean([1, 2])
                            mse = tensor2np(mse)
                            if data_idx == 0:
                                mses = mse
                            else:
                                mses = np.concatenate((mses, mse), axis=0)
                        if met.__name__ == 'tcc':
                            if type(source) == torch.Tensor or type(x_) == torch.Tensor:
                                source = tensor2np(source)
                                x_ = tensor2np(x_)
                            tcc = met(x_, source)
                            if data_idx == 0:
                                tccs = tcc
                            else:
                                tccs = np.concatenate((tccs, tcc), axis=0)
                        if met.__name__ == 'scc':
                            if type(source) == torch.Tensor or type(x_) == torch.Tensor:
                                source = tensor2np(source)
                                x_ = tensor2np(x_)
                            scc = met(x_, source)
                            if data_idx == 0:
                                sccs = scc
                            else:
                                sccs = np.concatenate((sccs, scc), axis=0)
                        if met.__name__ == 'dcc':
                            if type(source) == torch.Tensor or type(x_) == torch.Tensor:
                                source = tensor2np(source)
                                x_ = tensor2np(x_)
                            dcc = met(x_, source)
                            if data_idx == 0:
                                dccs = dcc
                            else:
                                dccs = np.concatenate((dccs, dcc), axis=0)

                    data_idx += 1

                if eval_tag in data_tags:
                    save_result(exp_dir, recons, grnths, labels, data_name, eval_tag)
    
    for met in metrics:
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'tcc':
            print_results(exp_dir, 'tcc', tccs)
        if met.__name__ == 'scc':
            print_results(exp_dir, 'scc', sccs)
        if met.__name__ == 'dcc':
            print_results(exp_dir, 'dcc', dccs)


def prediction_driver(model, spt_data_loaders, qry_data_loaders, metrics, hparams, exp_dir, data_tags):
    eval_config = hparams.evaluating
    loss_func = hparams.loss

    prediction_epoch(model, spt_data_loaders, qry_data_loaders, metrics, exp_dir, hparams, data_tags, eval_config, loss_func=loss_func)


def prediction_epoch(model, spt_data_loaders, qry_data_loaders, metrics, exp_dir, hparams, data_tags, eval_config, loss_func=None):
    torso_len = eval_config['torso_len']
    signal_source = eval_config['signal_source']
    omit = eval_config['omit']
    window = eval_config.get('window')
    k_shot = eval_config.get('k_shot')
    changable = eval_config.get('changable')
    data_scaler = eval_config.get('data_scaler')
    model.eval()
    n_data = 0
    data_idx = 0
    total_time = 0
    mses = None
    tccs = None
    sccs = None
    dccs = None

    with torch.no_grad():
        data_names = list(qry_data_loaders.keys())
        for data_name in data_names:
            qry_loader_tag = qry_data_loaders[data_name]
            spt_loader_tag = spt_data_loaders[data_name]

            qry_tags = list(qry_loader_tag.keys())
            spt_tags = list(spt_loader_tag.keys())
            
            for tag_idx, qry_tag in enumerate(qry_tags):
                recons, grnths, labels = None, None, None

                qry_data_loader = qry_loader_tag[qry_tag]
                spt_data_loader = spt_loader_tag[spt_tags[tag_idx]]
                len_epoch = len(qry_data_loader)
                n_data += len_epoch * qry_data_loader.batch_size
                
                data_iterator = iter(spt_data_loader)
                for idx, qry_data in enumerate(qry_data_loader):
                    try:
                        spt_data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(spt_data_loader)
                        spt_data = next(data_iterator)

                    qry_signal, qry_label = qry_data.x, qry_data.y
                    qry_signal = qry_signal.to(device)
                    qry_label = qry_label.to(device)

                    if window is not None:
                        qry_signal = qry_signal[:, :, :window]
                    
                    if data_scaler is not None:
                        qry_signal = data_scaler * qry_signal

                    qry_x_heart = qry_signal[:, :-torso_len, omit:]
                    qry_x_torso = qry_signal[:, -torso_len:, omit:]

                    if signal_source == 'heart':
                        qry_source = qry_x_heart
                    elif signal_source == 'torso':
                        qry_source = qry_x_torso

                    spt_signal, spt_label = spt_data.x, spt_data.y
                    spt_signal = spt_signal.to(device)
                    spt_label = spt_label.to(device)
                    D_x = spt_data.D
                    D_y = spt_data.D_label
                    D_x = D_x.to(device)
                    D_y = D_y.to(device)

                    if window is not None:
                        spt_signal = spt_signal[:, :, :window]
                        D_x = D_x[:, :, :window]
                    
                    if data_scaler is not None:
                        spt_signal = data_scaler * spt_signal
                        D_x = data_scaler * D_x
                    
                    spt_x_heart = spt_signal[:, :-torso_len, omit:]
                    spt_x_torso = spt_signal[:, -torso_len:, omit:]

                    N, M, T = qry_signal.shape
                    D_x = D_x.view(N, -1, M ,T)
                    D_x_heart = D_x[:, :, :-torso_len, omit:]
                    D_x_torso = D_x[:, :, -torso_len:, omit:]

                    if signal_source == 'heart':
                        spt_source = spt_x_heart
                        D_source = D_x_heart
                    elif signal_source == 'torso':
                        spt_source = spt_x_torso
                        D_source = D_x_torso
                    
                    if changable:
                        K = D_source.shape[1]
                        sub_K = np.random.randint(low=1, high=K+1, size=1)[0]
                        D_source = D_source[:, :sub_K, :]
                        D_y = D_y[:, :sub_K, :]

                    physics_vars, statistic_vars = model.prediction(qry_source, spt_source, D_source, D_y, data_name)

                    x_ = physics_vars[0]

                    if idx == 0:
                        recons = tensor2np(x_)
                        grnths = tensor2np(qry_source)
                        labels = tensor2np(qry_label)
                    else:
                        recons = np.concatenate((recons, tensor2np(x_)), axis=0)
                        grnths = np.concatenate((grnths, tensor2np(qry_source)), axis=0)
                        labels = np.concatenate((labels, tensor2np(qry_label)), axis=0)

                    for met in metrics:
                        if met.__name__ == 'mse':
                            mse = met(x_, qry_source)
                            mse = mse.mean([1, 2])
                            mse = tensor2np(mse)
                            if data_idx == 0:
                                mses = mse
                            else:
                                mses = np.concatenate((mses, mse), axis=0)
                        if met.__name__ == 'tcc':
                            if type(qry_source) == torch.Tensor or type(x_) == torch.Tensor:
                                qry_source = tensor2np(qry_source)
                                x_ = tensor2np(x_)
                            tcc = met(x_, qry_source)
                            if data_idx == 0:
                                tccs = tcc
                            else:
                                tccs = np.concatenate((tccs, tcc), axis=0)
                        if met.__name__ == 'scc':
                            if type(qry_source) == torch.Tensor or type(x_) == torch.Tensor:
                                qry_source = tensor2np(qry_source)
                                x_ = tensor2np(x_)
                            scc = met(x_, qry_source)
                            if data_idx == 0:
                                sccs = scc
                            else:
                                sccs = np.concatenate((sccs, scc), axis=0)
                        if met.__name__ == 'dcc':
                            if type(qry_source) == torch.Tensor or type(x_) == torch.Tensor:
                                qry_source = tensor2np(qry_source)
                                x_ = tensor2np(x_)
                            dcc = met(x_, qry_source)
                            if data_idx == 0:
                                dccs = dcc
                            else:
                                dccs = np.concatenate((dccs, dcc), axis=0)
                    
                    data_idx += 1

                if qry_tag in data_tags:
                    save_result(exp_dir, recons, grnths, labels, data_name, qry_tag)

    for met in metrics:
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'tcc':
            print_results(exp_dir, 'tcc', tccs)
        if met.__name__ == 'scc':
            print_results(exp_dir, 'scc', sccs)
        if met.__name__ == 'dcc':
            print_results(exp_dir, 'dcc', dccs)


def print_results(exp_dir, met_name, mets):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')
    
    # data_names = list(mets.keys())
    # mets_total = None
    # for idx, data_name in enumerate(data_names):
    #     if idx == 0:
    #         mets_total = mets[data_name]
    #     else:
    #         mets_total = np.concatenate((mets_total, mets[data_name]), axis=0)
        
    #     print('{}: {} for full seq = {:05.5f}'.format(data_name, met_name, mets[data_name].mean()))
    #     with open(os.path.join(exp_dir, 'data/metric.txt'), 'a+') as f:
    #         f.write('{}: {} for full seq = {}\n'.format(data_name, met_name, mets[data_name].mean()))
    
    print('Total {} avg = {:05.5f}, std = {:05.5f}'.format(met_name, mets.mean(), mets.std()))
    with open(os.path.join(exp_dir, 'data/metric.txt'), 'a+') as f:
        f.write('Total {} avg = {}, std = {}\n'.format(met_name, mets.mean(), mets.std()))


def save_result(exp_dir, recons, grnths, labels, data_name, data_tag):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')
    
    sio.savemat(
        os.path.join(exp_dir, 'data/{}_{}.mat'.format(data_name, data_tag)), 
        {'recons': recons, 'inps': grnths, 'label': labels}
    )


def tensor2np(t):
    return t.cpu().detach().numpy()
