import numpy as np
import scipy.io as sio
import os
from model.metric import *
import torch
import torch.nn.functional as F
import argparse


def parse_args():
    """
    Args:
        config: json file with hyperparams and exp settings
        seed: random seed value
        stage: 1 for traing VAE, 2 for optimization,  and 12 for both
        logging: 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='meta', help='model type')
    parser.add_argument('--id', type=str, default='05', help='exp index')

    args = parser.parse_args()
    return args

args = parse_args()
path_root = './experiments/{}/{}/'.format(args.model, args.id)

for i in [0, 4, 6, 9, 12, 14]:
# for i in [6, 9, 12, 14]:
    data = sio.loadmat('{}/data/qry_{}.mat'.format(path_root, i))
    recons = data['recons']
    inputs = data['inputs']
    # recons = data['recons'][:, 8:, :]
    # inputs = data['inputs'][:, 8:, :]

    recons_torch = torch.Tensor(recons)
    inputs_torch = torch.Tensor(inputs)
    mse_total = mse(recons_torch, inputs_torch)
    mse_total = mse_total.mean([1, 2, 3])
    mse_total = mse_total.cpu().detach().numpy()
    vpt_total = vpt(recons_torch, inputs_torch)
    vpt_total = vpt_total.cpu().detach().numpy()
    dst_total = dst(recons, inputs)
    dst_total = dst_total.mean(1)
    vpd_total = vpd(recons, inputs)

    print('set {} mse for seq avg = {}'.format(i, mse_total.mean()))
    print('set {} mse for seq std = {}'.format(i, mse_total.std()))
    print('set {} vpt for seq avg = {}'.format(i, vpt_total.mean()))
    print('set {} vpt for seq std = {}'.format(i, vpt_total.std()))
    print('set {} dst for seq avg = {}'.format(i, dst_total.mean()))
    print('set {} dst for seq std = {}'.format(i, dst_total.std()))
    print('set {} vpd for seq avg = {}'.format(i, vpd_total.mean()))
    print('set {} vpd for seq std = {}'.format(i, vpd_total.std()))

    with open('{}/data/metric.txt'.format(path_root), 'a+') as f:
        f.write('set {} mse for seq avg = {}\n'.format(i, mse_total.mean()))
        f.write('set {} mse for seq std = {}\n'.format(i, mse_total.std()))
        f.write('set {} vpt for seq avg = {}\n'.format(i, vpt_total.mean()))
        f.write('set {} vpt for seq std = {}\n'.format(i, vpt_total.std()))
        f.write('set {} dst for seq avg = {}\n'.format(i, dst_total.mean()))
        f.write('set {} dst for seq std = {}\n'.format(i, dst_total.std()))
        f.write('set {} vpd for seq avg = {}\n'.format(i, vpd_total.mean()))
        f.write('set {} vpd for seq std = {}\n'.format(i, vpd_total.std()))
