import numpy as np
import scipy.io as sio
import os
from model.metric import *
import torch
import argparse
from utils import Params

def parse_args():
    """
    Args:
        config: json file with hyperparams and exp settings
        seed: random seed value
        stage: 1 for traing VAE, 2 for optimization,  and 12 for both
        logging: 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='13', help='config filename')
    parser.add_argument('--tag', type=str, default='test', help='data tag')

    args = parser.parse_args()
    return args

args = parse_args()
path_root = './experiments/base/{}'.format(args.config)

all_recons = []
all_inputs = []
# for i in range(16):
# for i in [0, 1, 2, 3, 5, 6, 7, 8, 11, 13, 14, 15]:
for i in range(15):
    data = sio.loadmat('{}/{:02d}/data/test.mat'.format(path_root, i))
    recons = data['recons']
    inputs = data['inputs']
    # recons = data['recons'][:, 8:, :]
    # inputs = data['inputs'][:, 8:, :]
    all_recons.append(recons)
    all_inputs.append(inputs)

all_recons = np.concatenate(all_recons, axis=0)
all_inputs = np.concatenate(all_inputs, axis=0)

recons_torch = torch.Tensor(all_recons)
inputs_torch = torch.Tensor(all_inputs)
mse_total = mse(recons_torch, inputs_torch)
mse_total = mse_total.mean([1, 2, 3])
mse_total = mse_total.cpu().detach().numpy()
vpt_total = vpt(recons_torch, inputs_torch)
vpt_total = vpt_total.cpu().detach().numpy()
dst_total = dst(all_recons, all_inputs)
dst_total = dst_total.mean(1)
vpd_total = vpd(all_recons, all_inputs)

print('mse for seq avg = {}'.format(mse_total.mean()))
print('mse for seq std = {}'.format(mse_total.std()))
print('vpt for seq avg = {}'.format(vpt_total.mean()))
print('vpt for seq std = {}'.format(vpt_total.std()))
print('dst for seq avg = {}'.format(dst_total.mean()))
print('dst for seq std = {}'.format(dst_total.std()))
print('vpd for seq avg = {}'.format(vpd_total.mean()))
print('vpd for seq std = {}'.format(vpd_total.std()))

with open('{}/metric.txt'.format(path_root), 'a+') as f:
    f.write('mse for seq avg = {}\n'.format(mse_total.mean()))
    f.write('mse for seq std = {}\n'.format(mse_total.std()))
    f.write('vpt for seq avg = {}\n'.format(vpt_total.mean()))
    f.write('vpt for seq std = {}\n'.format(vpt_total.std()))
    f.write('dst for seq avg = {}\n'.format(dst_total.mean()))
    f.write('dst for seq std = {}\n'.format(dst_total.std()))
    f.write('vpd for seq avg = {}\n'.format(vpd_total.mean()))
    f.write('vpd for seq std = {}\n'.format(vpd_total.std()))
