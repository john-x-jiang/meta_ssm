import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import argparse
from utils import Params
from PIL import Image


def parse_args():
    """
    Args:
        config: json file with hyperparams and exp settings
        seed: random seed value
        stage: 1 for traing VAE, 2 for optimization,  and 12 for both
        logging: 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='b01', help='config filename')
    parser.add_argument('--tag', type=str, default='test', help='data tag')
    parser.add_argument('--sample', type=int, default=10, help='number of image samples')
    parser.add_argument('--obs', type=int, default=20, help='number of observed time steps')
    parser.add_argument('--timestep', type=int, default=20, help='number of time steps')
    parser.add_argument('--start', type=int, default=0, help='starting time steps')

    args = parser.parse_args()
    return args


def normalization(images):
    num_total = images.shape[0]
    time_total = images.shape[1]
    H, W = images.shape[2], images.shape[3]
    rtn = np.zeros((num_total, time_total, H, W))

    for i in range(num_total):
        for t in range(time_total):
            res = np.linalg.norm(images[i, t], axis=2)
            rtn[i, t] = res
    return rtn

np.random.seed(123)
args = parse_args()
# filename of the params
fname_config = args.config + '.json'
# read the params file
json_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), "config", fname_config)
hparams = Params(json_path)
exp_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'experiments', hparams.exp_name, hparams.exp_id)

mat = sio.loadmat(os.path.join(exp_dir, 'data/{}.mat'.format(args.tag)), squeeze_me=True, struct_as_record=False)
inputs = mat['inputs']
recons = mat['recons']

inputs = normalization(inputs)
recons = normalization(recons)

fig, axes = plt.subplots(2, 25, figsize=(25, 5))
for i, ax in enumerate(axes[0]):
    ax.imshow(inputs[0, i])
    
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

for i, ax in enumerate(axes[1]):
    ax.imshow(recons[0, i])
    
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

plt.savefig(os.path.join(exp_dir, 'data/{}_sample.png'.format(args.tag)), format='png', bbox_inches='tight')
