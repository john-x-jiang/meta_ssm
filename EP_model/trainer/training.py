import os
import sys
import time
import random
import numpy as np

import scipy.io
import scipy.stats as stats
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_driver(model, checkpt, epoch_start, optimizer, lr_scheduler, \
    train_loaders, valid_loaders, loss, metrics, hparams, exp_dir):
    train_loss, val_loss = [], []

    kl_t, nll_p_t, nll_q_t = [], [], []
    kl_e, nll_p_e, nll_q_e = [], [], []

    train_config = dict(hparams.training)
    monitor_mode, monitor_metric = train_config['monitor'].split()

    metric_err = None
    not_improved_count = 0

    if checkpt is not None:
        train_loss, val_loss = checkpt['train_loss'], checkpt['val_loss']

        kl_t, kl_e = checkpt['kl_t'], checkpt['kl_e']
        nll_q_t, nll_q_e = checkpt['nll_q_t'], checkpt['nll_q_e']
        nll_p_t, nll_p_e = checkpt['nll_p_t'], checkpt['nll_p_e']

        metric_err = checkpt.get('opt_metric_err')
        if metric_err is None:
            metric_err = checkpt[monitor_metric][-1]
        not_improved_count = checkpt['not_improved_count']

    for epoch in range(epoch_start, train_config['epochs'] + 1):
        ts = time.time()
        # train epoch
        total_loss_t, kl_loss_t, nll_p_loss_t, nll_q_loss_t = \
            train_epoch(model, epoch, loss, optimizer, train_loaders, hparams)
        
        # valid epoch
        total_loss_e, kl_loss_e, nll_p_loss_e, nll_q_loss_e = \
            valid_epoch(model, epoch, loss, valid_loaders, hparams)
        te = time.time()

        # Append epoch losses to arrays
        train_loss.append(total_loss_t)
        val_loss.append(total_loss_e)

        kl_t.append(kl_loss_t)
        kl_e.append(kl_loss_e)
        nll_p_t.append(nll_p_loss_t)
        nll_p_e.append(nll_p_loss_e)
        nll_q_t.append(nll_q_loss_t)
        nll_q_e.append(nll_q_loss_e)

        # Step LR
        if lr_scheduler is not None:
            lr_scheduler.step()
            last_lr = lr_scheduler._last_lr
        else:
            last_lr = 1
        
        # Generate the checkpoint for this current epoch
        log = {
            # Base parameters to reload
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'cur_learning_rate': last_lr,
            'not_improved_count': not_improved_count,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'opt_metric_err': metric_err,
            
            'kl_t': kl_t,
            'kl_e': kl_e,
            'nll_p_t': nll_p_t,
            'nll_p_e': nll_p_e,
            'nll_q_t': nll_q_t,
            'nll_q_e': nll_q_e
        }
        
        # Save the latest model
        torch.save(log, exp_dir + '/m_latest')

        # Save the model for every saving period
        if epoch % train_config['save_period'] == 0:
            torch.save(log, exp_dir + '/m_' + str(epoch))
        
        # Print and write out epoch logs
        logs = '[Epoch: {:04d}, Time: {:.4f}], train_loss: {:05.5f}, valid_loss: {:05.5f}'.format(
            epoch, (te - ts) / 60, total_loss_t, total_loss_e)
        print(logs)
        with open(os.path.join(exp_dir, 'log.txt'), 'a+') as f:
            f.write(logs + '\n')
        
        # Check if current epoch is better than best so far
        if epoch == 1:
            metric_err = log[monitor_metric][-1]
        else:
            improved = (monitor_mode == 'min' and log[monitor_metric][-1] <= metric_err) or \
                       (monitor_mode == 'max' and log[monitor_metric][-1] >= metric_err)
            if improved:
                metric_err = log[monitor_metric][-1]
                log['opt_metric_err'] = metric_err
                torch.save(log, exp_dir + '/m_best')
                not_improved_count = 0
            else:
                not_improved_count += 1
            
            if not_improved_count > train_config['early_stop']:
                info = "Validation performance didn\'t improve for {} epochs. Training stops.".format(train_config['early_stop'])
                break
        
    # save & plot losses
    losses = {
        'loss_total': [
            train_loss,
            val_loss
        ],
        'kl': [
            kl_t,
            kl_e
        ],
        'nll_p': [
            nll_p_t,
            nll_p_e
        ],
        'nll_q': [
            nll_q_t,
            nll_q_e
        ]
    }
    save_losses(exp_dir, train_config['epochs'], losses)


def train_epoch(model, epoch, loss, optimizer, data_loaders, hparams):
    model.train()
    train_config = dict(hparams.training)
    kl_args = train_config['kl_args']
    torso_len = train_config['torso_len']
    signal_source = train_config['signal_source']
    omit = train_config['omit']
    window = train_config.get('window')
    k_shot = train_config.get('k_shot')
    changable = train_config.get('changable')
    meta_dataset = train_config.get('meta_dataset')
    loss_type = train_config.get('loss_type')
    loss_func = hparams.loss
    total_loss = 0
    kl_loss, nll_p_loss, nll_q_loss = 0, 0, 0
    n_steps = 0
    batch_size = hparams.batch_size

    data_names = list(data_loaders.keys())
    random.shuffle(data_names)
    len_epoch = len(data_loaders[data_names[0]]) * len(data_names)
    for data_name in data_names:
        data_loader = data_loaders[data_name]
        if epoch > 1 and meta_dataset:
            data_loader = data_loader.next()
        for idx, data in enumerate(data_loader):
            signal, label = data.x, data.y
            signal = signal.to(device)
            label = label.to(device)

            if window is not None:
                signal = signal[:, :, :window]

            x_heart = signal[:, :-torso_len, omit:]
            x_torso = signal[:, -torso_len:, omit:]

            if signal_source == 'heart':
                source = x_heart
            elif signal_source == 'torso':
                source = x_torso

            optimizer.zero_grad()

            kl_annealing_factor = determine_annealing_factor(kl_args['min_annealing_factor'],
                                                                kl_args['anneal_update'],
                                                                epoch - 1, len_epoch, n_steps)
            r_kl = kl_args['lambda']
            kl_factor = kl_annealing_factor * r_kl

            r1 = train_config.get('r1')
            r2 = train_config.get('r2')
            r3 = train_config.get('r3')
            l = train_config.get('l')
            if r1 is None:
                r1 = 1
            if r2 is None:
                r2 = 0
            if r3 is None:
                r3 = 1
            
            if k_shot is None:
                physics_vars, statistic_vars = model(source, label, data_name)
            else:
                D_x = data.D
                D_y = data.D_label
                D_x = D_x.to(device)
                D_y = D_y.to(device)

                if window is not None:
                    D_x = D_x[:, :, :window]

                N, M, T = signal.shape
                D_x = D_x.view(N, -1, M ,T)

                D_x_heart = D_x[:, :, :-torso_len, omit:]
                D_x_torso = D_x[:, :, -torso_len:, omit:]

                if signal_source == 'heart':
                    D_source = D_x_heart
                elif signal_source == 'torso':
                    D_source = D_x_torso
                
                if changable:
                    K = D_source.shape[1]
                    sub_K = np.random.randint(low=1, high=K+1, size=1)[0]
                    D_source = D_source[:, :sub_K, :]
                    D_y = D_y[:, :sub_K, :]

                physics_vars, statistic_vars = model(source, label, D_source, D_y, data_name)
            
            if loss_func == 'recon_loss':
                x_ = physics_vars[0]
                _, _, _, _, mu_0, logvar_0 = statistic_vars
                if loss_type is None:
                    loss_type = 'mse'

                nll, kl_0, total = loss(x_, source, mu_0, logvar_0, kl_factor, loss_type, r3, l)
            elif loss_func == 'meta_loss':
                x_ = physics_vars[0]
                mu_c, logvar_c, mu_t, logvar_t = statistic_vars

                if loss_type is None:
                    loss_type = 'mse'
                
                kl_c, nll, kl_0, total = \
                    loss(x_, source, mu_c, logvar_c, mu_t, logvar_t, kl_factor, loss_type, r1, r2, l)
            elif loss_func == 'meta_loss_new':
                x_ = physics_vars[0]
                mu_c, logvar_c, mu_t, logvar_t, mu_0, logvar_0 = statistic_vars

                if loss_type is None:
                    loss_type = 'mse'
                
                kl_c, nll, kl_0, total = \
                    loss(x_, source, mu_c, logvar_c, mu_t, logvar_t, mu_0, logvar_0, kl_factor, loss_type, r1, r2, r3, l)
            else:
                raise NotImplemented

            total.backward()

            total_loss += total.item()
            nll_p_loss += nll.item()
            nll_q_loss += kl_0.item()
            if loss_func == 'meta_loss' or loss_func == 'meta_loss_new' :
                kl_loss += kl_c.item()
            n_steps += 1

            optimizer.step()
            logs = 'Training epoch {}, step {}, Average loss for epoch: {:05.5f}'.format(epoch, n_steps, total_loss / n_steps)
            util.inline_print(logs)

    total_loss /= n_steps
    kl_loss /= n_steps
    nll_p_loss /= n_steps
    nll_q_loss /= n_steps

    return total_loss, kl_loss, nll_p_loss, nll_q_loss


def valid_epoch(model, epoch, loss, data_loaders, hparams):
    model.eval()
    train_config = dict(hparams.training)
    kl_args = train_config['kl_args']
    torso_len = train_config['torso_len']
    signal_source = train_config['signal_source']
    omit = train_config['omit']
    window = train_config.get('window')
    k_shot = train_config.get('k_shot')
    changable = train_config.get('changable')
    meta_dataset = train_config.get('meta_dataset')
    loss_type = train_config.get('loss_type')
    loss_func = hparams.loss
    total_loss = 0
    kl_loss, nll_p_loss, nll_q_loss = 0, 0, 0
    n_steps = 0
    batch_size = hparams.batch_size

    with torch.no_grad():
        data_names = list(data_loaders.keys())
        len_epoch = len(data_loaders[data_names[0]]) * len(data_names)
        for data_name in data_names:
            data_loader = data_loaders[data_name]
            if epoch > 1 and meta_dataset:
                data_loader = data_loader.next()
            for idx, data in enumerate(data_loader):
                signal, label = data.x, data.y
                signal = signal.to(device)
                label = label.to(device)

                if window is not None:
                    signal = signal[:, :, :window]
                
                x_heart = signal[:, :-torso_len, omit:]
                x_torso = signal[:, -torso_len:, omit:]

                if signal_source == 'heart':
                    source = x_heart
                elif signal_source == 'torso':
                    source = x_torso

                r_kl = kl_args['lambda']
                kl_factor = 1 * r_kl

                r1 = train_config.get('r1')
                r2 = train_config.get('r2')
                r3 = train_config.get('r3')
                l = train_config.get('l')
                if r1 is None:
                    r1 = 1
                if r2 is None:
                    r2 = 0
                if r3 is None:
                    r3 = 1
                
                if k_shot is None:
                    physics_vars, statistic_vars = model(source, label, data_name)
                else:
                    D_x = data.D
                    D_y = data.D_label
                    D_x = D_x.to(device)
                    D_y = D_y.to(device)

                    if window is not None:
                        D_x = D_x[:, :, :window]

                    N, M, T = signal.shape
                    D_x = D_x.view(N, -1, M ,T)
                    D_x_heart = D_x[:, :, :-torso_len, omit:]
                    D_x_torso = D_x[:, :, -torso_len:, omit:]

                    if signal_source == 'heart':
                        D_source = D_x_heart
                    elif signal_source == 'torso':
                        D_source = D_x_torso
                    
                    if changable:
                        K = D_source.shape[1]
                        sub_K = np.random.randint(low=1, high=K+1, size=1)[0]
                        D_source = D_source[:, :sub_K, :]
                        D_y = D_y[:, :sub_K, :]

                    physics_vars, statistic_vars = model(source, label, D_source, D_y, data_name)
                
                if loss_func == 'recon_loss':
                    x_ = physics_vars[0]
                    _, _, _, _, mu_0, logvar_0 = statistic_vars
                    if loss_type is None:
                        loss_type = 'mse'

                    nll, kl_0, total = loss(x_, source, mu_0, logvar_0, kl_factor, loss_type, r3, l)
                elif loss_func == 'meta_loss':
                    x_ = physics_vars[0]
                    mu_c, logvar_c, mu_t, logvar_t = statistic_vars

                    if loss_type is None:
                        loss_type = 'mse'
                    
                    kl_c, nll, kl_0, total = \
                        loss(x_, source, mu_c, logvar_c, mu_t, logvar_t, kl_factor, loss_type, r1, r2, l)
                elif loss_func == 'meta_loss_new':
                    x_ = physics_vars[0]
                    mu_c, logvar_c, mu_t, logvar_t, mu_0, logvar_0 = statistic_vars

                    if loss_type is None:
                        loss_type = 'mse'
                    
                    kl_c, nll, kl_0, total = \
                        loss(x_, source, mu_c, logvar_c, mu_t, logvar_t, mu_0, logvar_0, kl_factor, loss_type, r1, r2, r3, l)
                else:
                    raise NotImplemented

                total_loss += total.item()
                nll_p_loss += nll.item()
                nll_q_loss += kl_0.item()
                if loss_func == 'meta_loss' or loss_func == 'meta_loss_new' :
                    kl_loss += kl_c.item()
                n_steps += 1

    total_loss /= n_steps
    kl_loss /= n_steps
    nll_p_loss /= n_steps
    nll_q_loss /= n_steps

    return total_loss, kl_loss, nll_p_loss, nll_q_loss


def determine_annealing_factor(min_anneal_factor,
                               anneal_update,
                               epoch, n_batch, batch_idx):
    n_updates = epoch * n_batch + batch_idx

    if anneal_update > 0 and n_updates < anneal_update:
        anneal_factor = min_anneal_factor + \
            (1.0 - min_anneal_factor) * (
                (n_updates / anneal_update)
            )
    else:
        anneal_factor = 1.0
    return anneal_factor


def save_losses(exp_dir, num_epochs, losses):
    """Plot epoch against train loss and test loss 
    """
    # plot of the train/validation error against num_epochs
    for loss_type, loss_cmb in losses.items():
        train_a, test_a = loss_cmb
        plot_loss(exp_dir, num_epochs, train_a, test_a, loss_type)
        train_a = np.array(train_a)
        test_a = np.array(test_a)
        np.save(os.path.join(exp_dir, 'loss_{}_t.npy'.format(loss_type)), train_a)
        np.save(os.path.join(exp_dir, 'loss_{}_e.npy'.format(loss_type)), test_a)


def plot_loss(exp_dir, num_epochs, train_a, test_a, loss_type):
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.set_xticks(np.arange(0 + 1, num_epochs + 1, step=10))
    ax1.set_xlabel('epochs')
    ax1.plot(train_a, color='green', ls='-', label='train accuracy')
    ax1.plot(test_a, color='red', ls='-', label='test accuracy')
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, fontsize='14', frameon=False)
    ax1.grid(linestyle='--')
    plt.tight_layout()
    fig.savefig(exp_dir + '/loss_{}.png'.format(loss_type), dpi=300, bbox_inches='tight')
    plt.close()
