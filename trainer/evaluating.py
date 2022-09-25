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


def evaluate_driver(model, data_loaders, metrics, hparams, exp_dir, data_tags):
    eval_config = hparams.evaluating
    # reconstruction
    evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, eval_config, data_tags)


def evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, eval_config, data_tags):
    model.eval()
    total_len = eval_config.get('total_len')
    domain = eval_config.get('domain')
    n_steps = 0
    bces, mses, vpts = None, None, None
    dsts, vpds = None, None

    recons = None
    grdths = None
    labels = None

    with torch.no_grad():
        data_names = list(data_loaders.keys())
        for data_idx, data_name in enumerate(data_names):
            data_loader = data_loaders[data_name]
            bces_data, mses_data, vpts_data = None, None, None
            dsts_data, vpds_data = None, None
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
                
                if B == 1:
                    x_ = torch.reshape(x_, x.size())
                n_steps += 1
                
                if torch.isnan(x_).any():
                    x_ = torch.nan_to_num(x_)

                if idx == 0:
                    recons = tensor2np(x_)
                    grdths = tensor2np(x)
                    labels = tensor2np(label)                    
                else:
                    if len(x_.shape) < 4:
                        H, W = x.shape[2], x.shape[3]
                        x_ = x_.view(1, T, H, W)
                    recons = np.concatenate((recons, tensor2np(x_)), axis=0)
                    grdths = np.concatenate((grdths, tensor2np(x)), axis=0)
                    labels = np.concatenate((labels, tensor2np(label)), axis=0)
                
                for met in metrics:
                    if met.__name__ == 'bce':
                        # reconstruction
                        bce = met(x_, x)
                        bce = bce.mean([1, 2, 3])
                        bce = tensor2np(bce)
                        if idx == 0:
                            bces_data = bce
                        else:
                            bces_data = np.concatenate((bces_data, bce), axis=0)
                    if met.__name__ == 'mse':
                        # reconstruction
                        mse = met(x_, x)
                        mse = mse.mean([1, 2, 3])
                        mse = tensor2np(mse)
                        if idx == 0:
                            mses_data = mse
                        else:
                            mses_data = np.concatenate((mses_data, mse), axis=0)
                    if met.__name__ == 'vpt':
                        # reconstruction
                        vpt = met(x_, x)
                        vpt = tensor2np(vpt)
                        if idx == 0:
                            vpts_data = vpt
                        else:
                            vpts_data = np.concatenate((vpts_data, vpt), axis=0)
                    if met.__name__ == 'dst':
                        # reconstruction
                        if isinstance(x, torch.Tensor):
                            x = tensor2np(x)
                        if isinstance(x_, torch.Tensor):
                            x_ = tensor2np(x_)
                        dst = met(x_, x)
                        dst = dst.mean(1)
                        if idx == 0:
                            dsts_data = dst
                        else:
                            dsts_data = np.concatenate((dsts_data, dst), axis=0)
                    if met.__name__ == 'vpd':
                        # reconstruction
                        if isinstance(x, torch.Tensor):
                            x = tensor2np(x)
                        if isinstance(x_, torch.Tensor):
                            x_ = tensor2np(x_)
                        vpd = met(x_, x)
                        if idx == 0:
                            vpds_data = vpd
                        else:
                            vpds_data = np.concatenate((vpds_data, vpd), axis=0)
            
            if data_name in data_tags:
                save_result(exp_dir, recons, grdths, labels, data_name)
            for met in metrics:
                if met.__name__ == 'bce':
                    # reconstruction
                    if data_idx == 0:
                        bces = bces_data
                    else:
                        bces = np.concatenate((bces, bces_data), axis=0)
                if met.__name__ == 'mse':
                    # reconstruction
                    if data_idx == 0:
                        mses = mses_data
                    else:
                        mses = np.concatenate((mses, mses_data), axis=0)
                if met.__name__ == 'vpt':
                    # reconstruction
                    if data_idx == 0:
                        vpts = vpts_data
                    else:
                        vpts = np.concatenate((vpts, vpts_data), axis=0)
                if met.__name__ == 'dst':
                    # reconstruction
                    if data_idx == 0:
                        dsts = dsts_data
                    else:
                        dsts = np.concatenate((dsts, dsts_data), axis=0)
                if met.__name__ == 'vpd':
                    # reconstruction
                    if data_idx == 0:
                        vpds = vpds_data
                    else:
                        vpds = np.concatenate((vpds, vpds_data), axis=0)

    for met in metrics:
        if met.__name__ == 'bce':
            print_results(exp_dir, 'bce', bces)
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'vpt':
            print_results(exp_dir, 'vpt', vpts)
        if met.__name__ == 'dst':
            print_results(exp_dir, 'dst', dsts)
        if met.__name__ == 'vpd':
            print_results(exp_dir, 'vpd', vpds)


def prediction_driver(model, eval_data_loaders, pred_data_loaders, metrics, hparams, exp_dir, data_tags):
    eval_config = hparams.evaluating
    # reconstruction
    prediction_epoch(model, eval_data_loaders, pred_data_loaders, metrics, exp_dir, hparams, eval_config, data_tags)


def prediction_epoch(model, eval_data_loaders, pred_data_loaders, metrics, exp_dir, hparams, eval_config, data_tags):
    model.eval()
    total_len = eval_config.get('total_len')
    domain = eval_config.get('domain')
    changeable = eval_config.get('changeable')
    batch_size = eval_config.get('batch_size')
    n_steps = 0
    data_idx = 0
    bces, mses, vpts = None, None, None
    dsts, vpds = None, None

    recons = None
    grdths = None
    labels = None

    with torch.no_grad():
        eval_data_names = list(eval_data_loaders.keys())
        pred_data_names = list(pred_data_loaders.keys())
        for eval_data_name, pred_data_name in zip(eval_data_names, pred_data_names):
            pred_data_loader = pred_data_loaders[pred_data_name]
            eval_data_loader = eval_data_loaders[eval_data_name]
            bces_data, mses_data, vpts_data = None, None, None
            dsts_data, vpds_data = None, None
            data_iterator = iter(eval_data_loader)
            for idx, batch in enumerate(pred_data_loader):
                x, _, x_state, _, label = batch

                try:
                    eval_batch = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(eval_data_loader)
                    eval_batch = next(data_iterator)
                _, D, _, D_state, D_label = eval_batch

                if total_len is not None:
                    x = x[:, :total_len]

                B, T = x.shape[0], x.shape[1]
                x = x.to(device)
                
                if domain:
                    if total_len is not None:
                        D = D[:, :, :total_len]
                    K = D.shape[1]
                    if changeable:
                        sub_K = np.random.randint(low=1, high=K+1, size=1)[0]
                        D = D[:, :sub_K, :]
                    D = D.to(device)
                    if x.shape[0] < batch_size:
                        D = D[:x.shape[0], :]
                    x_ = model.prediction(x, D)
                else:
                    x_, mu_0, var_0, mu_c, var_c = model(D)
                
                if B == 1:
                    x_ = torch.reshape(x_, x.size())
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
                        bce = bce.mean([1, 2, 3])
                        bce = tensor2np(bce)
                        if idx == 0:
                            bces_data = bce
                        else:
                            bces_data = np.concatenate((bces_data, bce), axis=0)
                    if met.__name__ == 'mse':
                        # reconstruction
                        mse = met(x_, x)
                        mse = mse.mean([1, 2, 3])
                        mse = tensor2np(mse)
                        if idx == 0:
                            mses_data = mse
                        else:
                            mses_data = np.concatenate((mses_data, mse), axis=0)
                    if met.__name__ == 'vpt':
                        # reconstruction
                        vpt = met(x_, x)
                        vpt = tensor2np(vpt)
                        if idx == 0:
                            vpts_data = vpt
                        else:
                            vpts_data = np.concatenate((vpts_data, vpt), axis=0)
                    if met.__name__ == 'dst':
                        # reconstruction
                        if isinstance(x, torch.Tensor):
                            x = tensor2np(x)
                        if isinstance(x_, torch.Tensor):
                            x_ = tensor2np(x_)
                        dst = met(x_, x)
                        dst = dst.mean(1)
                        if idx == 0:
                            dsts_data = dst
                        else:
                            dsts_data = np.concatenate((dsts_data, dst), axis=0)
                    if met.__name__ == 'vpd':
                        # reconstruction
                        if isinstance(x, torch.Tensor):
                            x = tensor2np(x)
                        if isinstance(x_, torch.Tensor):
                            x_ = tensor2np(x_)
                        vpd = met(x_, x)
                        if idx == 0:
                            vpds_data = vpd
                        else:
                            vpds_data = np.concatenate((vpds_data, vpd), axis=0)

            if pred_data_name in data_tags:
                save_result(exp_dir, recons, grdths, labels, pred_data_name)
            for met in metrics:
                if met.__name__ == 'bce':
                    # reconstruction
                    if data_idx == 0:
                        bces = bces_data
                    else:
                        bces = np.concatenate((bces, bces_data), axis=0)
                if met.__name__ == 'mse':
                    # reconstruction
                    if data_idx == 0:
                        mses = mses_data
                    else:
                        mses = np.concatenate((mses, mses_data), axis=0)
                if met.__name__ == 'vpt':
                    # reconstruction
                    if data_idx == 0:
                        vpts = vpts_data
                    else:
                        vpts = np.concatenate((vpts, vpts_data), axis=0)
                if met.__name__ == 'dst':
                    # reconstruction
                    if data_idx == 0:
                        dsts = dsts_data
                    else:
                        dsts = np.concatenate((dsts, dsts_data), axis=0)
                if met.__name__ == 'vpd':
                    # reconstruction
                    if data_idx == 0:
                        vpds = vpds_data
                    else:
                        vpds = np.concatenate((vpds, vpds_data), axis=0)
            data_idx += 1

    for met in metrics:
        if met.__name__ == 'bce':
            print_results(exp_dir, 'bce', bces)
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'vpt':
            print_results(exp_dir, 'vpt', vpts)
        if met.__name__ == 'dst':
            print_results(exp_dir, 'dst', dsts)
        if met.__name__ == 'vpd':
            print_results(exp_dir, 'vpd', vpds)


def embedding_driver(model, eval_data_loaders, pred_data_loaders, metrics, hparams, exp_dir, data_tags):
    eval_config = hparams.evaluating
    model.eval()
    total_len = eval_config.get('total_len')
    domain = eval_config.get('domain')
    changeable = eval_config.get('changeable')
    batch_size = eval_config.get('batch_size')
    n_steps = 0
    data_idx = 0

    with torch.no_grad():
        eval_data_names = list(eval_data_loaders.keys())
        pred_data_names = list(pred_data_loaders.keys())
        for eval_data_name, pred_data_name in zip(eval_data_names, pred_data_names):
            pred_data_loader = pred_data_loaders[pred_data_name]
            eval_data_loader = eval_data_loaders[eval_data_name]
            data_iterator = iter(eval_data_loader)

            c_mus, c_vars, c_zs, labels = [], [], [], []
            recons, grdths = None, None
            for idx, batch in enumerate(pred_data_loader):
                x, _, x_state, _, label = batch

                try:
                    eval_batch = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(eval_data_loader)
                    eval_batch = next(data_iterator)
                _, D, _, D_state, D_label = eval_batch

                if total_len is not None:
                    x = x[:, :total_len]

                B, T = x.shape[0], x.shape[1]
                x = x.to(device)
                
                if total_len is not None:
                    D = D[:, :, :total_len]
                K = D.shape[1]
                if changeable:
                    sub_K = np.random.randint(low=1, high=K+1, size=1)[0]
                    D = D[:, :sub_K, :]
                D = D.to(device)
                if x.shape[0] < batch_size:
                    D = D[:x.shape[0], :]
                z_c, mu_c, logvar_c = model.prediction_embedding(x, D)
                
                if B == 1:
                    C = z_c.shape
                    z_c = torch.reshape(z_c, [1, C])
                    mu_c = torch.reshape(mu_c, [1, C])
                    logvar_c = torch.reshape(logvar_c, [1, C])
                
                n_steps += 1

                kmeans_path = '{}/embeddings/kmeans_clusters.npz'.format(exp_dir)
                if os.path.exists(kmeans_path):
                    centroids = np.load(kmeans_path)
                    centers = centroids['centers']
                    clabels = centroids['labels']

                    D_centers = torch.from_numpy(centers).to(device)
                    D_centers = D_centers.unsqueeze(1).repeat(1, x.shape[0], 1)
                    # predictions = x.unsqueeze(0)
                    x_ = None

                    # initial condition
                    z_0, _, _ = model.latent_initialization(x[:, :model.init_dim, :])

                    for cid, center in enumerate(D_centers):
                        if torch.unique(label).shape[0] == 1 and clabels[cid] == torch.unique(label)[0]:
                            x_ = model.prediction_sampling(x, z_0, center)
                            # predictions = torch.vstack((predictions, x_.unsqueeze(0)))
                        else:
                            continue

                    # np.save(f"{exp_dir}/embeddings/predictions_{pred_data_name}_initial_state_{idx}.npy",
                    #         predictions.detach().cpu().numpy())

                if idx == 0:
                    c_mus = tensor2np(mu_c)
                    c_vars = tensor2np(logvar_c)
                    c_zs = tensor2np(z_c)
                    labels = tensor2np(label)

                    if x_ is not None:
                        recons = tensor2np(x_)
                        grdths = tensor2np(x)
                else:
                    c_mus = np.concatenate((c_mus, tensor2np(mu_c)), axis=0)
                    c_vars = np.concatenate((c_vars, tensor2np(logvar_c)), axis=0)
                    c_zs = np.concatenate((c_zs, tensor2np(z_c)), axis=0)
                    labels = np.concatenate((labels, tensor2np(label)), axis=0)

                    if x_ is not None:
                        recons = np.concatenate((recons, tensor2np(x_)), axis=0)
                        grdths = np.concatenate((grdths, tensor2np(x)), axis=0)

            print(c_zs.shape)
            if not os.path.exists(kmeans_path):
                if not os.path.exists(exp_dir + '/embeddings'):
                    os.makedirs(exp_dir + '/embeddings')
                
                np.savez(f"{exp_dir}/embeddings/embeddings_{pred_data_name}.npz",
                        c_vars=c_vars, c_mus=c_mus, c_zs=c_zs, labels=labels)
            else:
                if recons is not None:
                    sio.savemat(os.path.join(exp_dir, 'embeddings/{}.mat'.format(pred_data_name)), 
                                {'recons': recons, 'inputs': grdths, 'label': labels})


def print_results(exp_dir, met_name, mets):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')
    
    print('{} for seq avg = {:05.5f}'.format(met_name, mets.mean()))
    print('{} for seq std = {:05.5f}'.format(met_name, mets.std()))
    with open(os.path.join(exp_dir, 'data/metric.txt'), 'a+') as f:
        f.write('{} for seq avg = {}\n'.format(met_name, mets.mean()))
        f.write('{} for seq std = {}\n'.format(met_name, mets.std()))


def save_result(exp_dir, recons, inputs, labels, data_tag):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')

    sio.savemat(os.path.join(exp_dir, 'data/{}.mat'.format(data_tag)), {'recons': recons, 'inputs': inputs, 'label': labels})


def tensor2np(t):
    return t.cpu().detach().numpy()
