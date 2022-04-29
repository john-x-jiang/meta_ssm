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
from data_loader.seq_util import reverse_sequence, binary_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_driver(model, checkpt, epoch_start, optimizer, lr_scheduler, \
    train_loader, valid_loader, loss, metrics, hparams, exp_dir):
    kl_m_c_t, kl_m_ct_t, kl_initial_t, likelihood_t, train_loss = [], [], [], [], []
    kl_m_c_e, kl_m_ct_e, kl_initial_e, likelihood_e, val_loss = [], [], [], [], []

    train_config = dict(hparams.training)
    monitor_mode, monitor_metric = train_config['monitor'].split()

    metric_err = None
    not_improved_count = 0

    if checkpt is not None:
        train_loss, val_loss = checkpt['train_loss'], checkpt['val_loss']

        kl_m_c_t, kl_m_c_e = checkpt['kl_m_c_t'], checkpt['kl_m_c_e']
        kl_m_ct_t, kl_m_ct_e = checkpt['kl_m_ct_t'], checkpt['kl_m_ct_e']
        kl_initial_t, kl_initial_e = checkpt['kl_initial_t'], checkpt['kl_initial_e']
        likelihood_t, likelihood_e = checkpt['likelihood_t'], checkpt['likelihood_e']

        metric_err = checkpt[monitor_metric][-1]
        not_improved_count = checkpt['not_improved_count']

    for epoch in range(epoch_start, train_config['epochs'] + 1):
        ts = time.time()
        # train epoch
        kl_m_c_loss_t, kl_m_ct_loss_t, kl_initial_loss_t, likelihood_loss_t, total_loss_t = \
            train_epoch(model, epoch, loss, optimizer, train_loader, hparams)
        
        # valid epoch
        kl_m_c_loss_e, kl_m_ct_loss_e, kl_initial_loss_e, likelihood_loss_e, total_loss_e = \
            valid_epoch(model, epoch, loss, valid_loader, hparams)
        te = time.time()

        # Append epoch losses to arrays
        kl_m_c_t.append(kl_m_c_loss_t)
        kl_m_ct_t.append(kl_m_ct_loss_t)
        kl_initial_t.append(kl_initial_loss_t)
        likelihood_t.append(likelihood_loss_t)
        train_loss.append(total_loss_t)
        
        kl_m_c_e.append(kl_m_c_loss_e)
        kl_m_ct_e.append(kl_m_ct_loss_e)
        kl_initial_e.append(kl_initial_loss_e)
        likelihood_e.append(likelihood_loss_e)
        val_loss.append(total_loss_e)

        # Step LR
        if lr_scheduler is not None:
            lr_scheduler.step()
            last_lr = lr_scheduler._last_lr
        else:
            last_lr = optimizer.param_groups[0]['lr']
        
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

            # Holding the current arrays for training
            'kl_m_c_t': kl_m_c_t,
            'kl_m_ct_t': kl_m_ct_t,
            'kl_initial_t': kl_initial_t,
            'likelihood_t': likelihood_t,

            # Holding the current arrays for testing
            'kl_m_c_e': kl_m_c_e,
            'kl_m_ct_e': kl_m_ct_e,
            'kl_initial_e': kl_initial_e,
            'likelihood_e': likelihood_e,
        }
        
        # Save the latest model
        torch.save(log, exp_dir + '/m_latest')

        # Save the model for every saving period
        if epoch % train_config['save_period'] == 0:
            torch.save(log, exp_dir + '/m_' + str(epoch))
        
        # Print and write out epoch logs
        logs = '[Epoch: {:04d}, Time: {:.4f}]                                        '.format(epoch, (te - ts) / 60) \
            + '\ntrain_loss: {:05.5f}, likelihood: {:05.5f}, kl_c: {:05.5f}, kl_0: {:05.5f}, kl_ct: {:05.5f}'.format(
                total_loss_t, likelihood_loss_t, kl_m_c_loss_t, kl_initial_loss_t, kl_m_ct_loss_t) \
            + '\nvalid_loss: {:05.5f}, likelihood: {:05.5f}, kl_c: {:05.5f}, kl_0: {:05.5f}, kl_ct: {:05.5f}'.format(
                total_loss_e, likelihood_loss_e, kl_m_c_loss_e, kl_initial_loss_e, kl_m_ct_loss_e)
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
                torch.save(log, exp_dir + '/m_best')
                not_improved_count = 0
            else:
                not_improved_count += 1
            
            if not_improved_count > train_config['early_stop']:
                info = "Validation performance didn\'t improve for {} epochs. Training stops.".format(train_config['early_stop'])
                break
        
    # save & plot losses
    losses = {
        'total': [
            train_loss,
            val_loss
        ],
        'kl_m_c': [
            kl_m_c_t,
            kl_m_c_e
        ],
        'kl_m_ct': [
            kl_m_ct_t,
            kl_m_ct_e
        ],
        'kl_initial': [
            kl_initial_t,
            kl_initial_e
        ],
        'likelihood': [
            likelihood_t,
            likelihood_e
        ]
    }
    save_losses(exp_dir, train_config['epochs'], losses)


def train_epoch(model, epoch, loss, optimizer, data_loader, hparams):
    model.train()
    train_config = dict(hparams.training)
    total_len = train_config.get('total_len')
    reverse = train_config.get('reverse')
    loss_type = train_config.get('loss_type')
    domain = train_config.get('domain')
    kl_m_c_loss, kl_m_ct_loss, kl_initial_loss = 0, 0, 0
    likelihood_loss, total_loss = 0, 0
    len_epoch = len(data_loader)
    n_steps = 0
    batch_size = hparams.batch_size

    if epoch > 1:
        data_loader = data_loader.next()
    for idx, batch in enumerate(data_loader):
        x, D, x_state, D_state, label = batch
        if total_len is not None:
            x = x[:, :total_len]
            D = D[:, :, :total_len]

        B, T = x.shape[0], x.shape[1]
        K = D.shape[1]
        x = x.to(device)
        D = D.to(device)
        seq_length = total_len * torch.ones(B).int()
        if reverse:
            x_reversed = reverse_sequence(x, seq_length)
            x_reversed = x_reversed.to(device)
            inputs = x_reversed

            inputs_D = torch.zeros_like(D)
            for i in range(K):
                inputs_D[:, i, :] = reverse_sequence(D[:, i, :], seq_length)
        else:
            inputs = x
            inputs_D = D

        optimizer.zero_grad()

        kl_annealing_factor = determine_annealing_factor(train_config['min_annealing_factor'],
                                                         train_config['anneal_update'],
                                                         epoch - 1, len_epoch, idx)
        r_kl = train_config['lambda']
        kl_factor = kl_annealing_factor * r_kl

        r1 = train_config.get('r1')
        r2 = train_config.get('r2')
        r3 = train_config.get('r3')
        
        if domain:
            x_, D_, mu_c, var_c, mu_t, var_t, mu_0, var_0, D_mu0, D_var0 = model(inputs, inputs_D)

            kl_m_c, kl_m_ct, kl_initial, likelihood, total = loss(x, x_, D, D_, \
                mu_c, var_c, mu_t, var_t, mu_0, var_0, D_mu0, D_var0, kl_factor, loss_type, r1, r2, r3)
        else:
            x_, mu_0, var_0, mu_c, var_c = model(inputs)
            kl_m_c, kl_initial, likelihood, total = loss(x, x_, mu_0, var_0, mu_c, var_c, kl_factor, loss_type, r1, r2)
        total.backward()

        kl_initial_loss += kl_initial.item()
        likelihood_loss += likelihood.item()
        kl_m_c_loss += kl_m_c.item()
        if domain:
            kl_m_ct_loss += kl_m_ct.item()
        total_loss += total.item()
        n_steps += 1

        optimizer.step()
        logs = 'Training epoch {}, step {}, Average loss for epoch: {:05.5f}'.format(epoch, n_steps, total_loss / n_steps)
        util.inline_print(logs)

    kl_m_c_loss /= n_steps
    kl_m_ct_loss /= n_steps
    kl_initial_loss /= n_steps
    likelihood_loss /= n_steps
    total_loss /= n_steps

    return kl_m_c_loss, kl_m_ct_loss, kl_initial_loss, likelihood_loss, total_loss


def valid_epoch(model, epoch, loss, data_loader, hparams):
    model.eval()
    train_config = hparams.training
    total_len = train_config.get('total_len')
    reverse = train_config.get('reverse')
    loss_type = train_config.get('loss_type')
    domain = train_config.get('domain')
    kl_m_c_loss, kl_m_ct_loss, kl_initial_loss = 0, 0, 0
    likelihood_loss, total_loss = 0, 0
    n_steps = 0
    batch_size = hparams.batch_size

    with torch.no_grad():
        if epoch > 1:
            data_loader = data_loader.next()
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

            r_kl = train_config['lambda']
            kl_factor = 1 * r_kl
            
            r1 = train_config.get('r1')
            r2 = train_config.get('r2')
            r3 = train_config.get('r3')
            
            if domain:
                x_, D_, mu_c, var_c, mu_t, var_t, mu_0, var_0, D_mu0, D_var0 = model(inputs, inputs_D)

                kl_m_c, kl_m_ct, kl_initial, likelihood, total = loss(x, x_, D, D_, \
                    mu_c, var_c, mu_t, var_t, mu_0, var_0, D_mu0, D_var0, kl_factor, loss_type, r1, r2, r3)
            else:
                x_, mu_0, var_0, mu_c, var_c = model(inputs)
                kl_m_c, kl_initial, likelihood, total = loss(x, x_, mu_0, var_0, mu_c, var_c, kl_factor, loss_type, r1, r2)
            
            kl_initial_loss += kl_initial.item()
            likelihood_loss += likelihood.item()
            kl_m_c_loss += kl_m_c.item()
            if domain:
                kl_m_ct_loss += kl_m_ct.item()
            total_loss += total.item()
            n_steps += 1

    kl_m_c_loss /= n_steps
    kl_m_ct_loss /= n_steps
    kl_initial_loss /= n_steps
    likelihood_loss /= n_steps
    total_loss /= n_steps

    return kl_m_c_loss, kl_m_ct_loss, kl_initial_loss, likelihood_loss, total_loss


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
