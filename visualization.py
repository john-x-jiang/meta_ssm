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
    parser.add_argument('--obs', type=int, default=10, help='number of observed time steps')
    parser.add_argument('--timestep', type=int, default=60, help='number of time steps')
    parser.add_argument('--start', type=int, default=0, help='starting time steps')

    args = parser.parse_args()
    return args

np.random.seed(123)
args = parse_args()
# filename of the params
fname_config = args.config + '.json'
# read the params file
json_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), "config", fname_config)
hparams = Params(json_path)

exp_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'experiments', hparams.exp_name, hparams.exp_id)
emission = hparams.evaluating['emission']

# mat = sio.loadmat(os.path.join(exp_dir, 'data/{}.mat'.format(args.tag)), squeeze_me=True, struct_as_record=False)
# all_recons = mat['recons']
# all_inps = mat['inps']
# all_labels = mat['label']

mat = sio.loadmat(os.path.join(exp_dir, 'data/{}_{}.mat'.format(args.tag, args.obs)), squeeze_me=True, struct_as_record=False)
all_inps = mat['inps']
p_recons = mat['p_recons']
q_recons = mat['q_recons']
gens = mat['gens']

blank = 5
num_sample = args.sample
time_steps = args.timestep
obs = args.obs
start = args.start

pannel = np.ones((32 * 3 * num_sample + blank * (num_sample + 2), 32 * time_steps + 3 * blank)) * 255
pannel = np.uint8(pannel)
pannel = Image.fromarray(pannel)

selected_idx = np.random.choice(all_inps.shape[0], num_sample, replace=False)
selected_idx = sorted(selected_idx)

for num, idx in enumerate(selected_idx):
    selected_inps = all_inps[idx]
    if emission == 'p':
        selected_rcns = p_recons[idx]
    elif emission == 'q':
        selected_rcns = q_recons[idx]
    elif emission == 'pq':
        selected_rcns = p_recons[idx]
        selected_cmps = q_recons[idx]
    
    selected_inps = np.uint8(selected_inps * 255)
    selected_rcns = np.uint8(selected_rcns * 255)
    if emission == 'pq':
        selected_cmps = np.uint8(selected_cmps * 255)

    img = np.zeros((32 * 3, obs * 32)).astype(np.uint8)
    for i in range(obs):
        img[:32, i * 32: (i + 1) * 32] = selected_inps[i]
        img[32:64, i * 32: (i + 1) * 32] = selected_rcns[i]
        if emission == 'pq':
            img[64:96, i * 32: (i + 1) * 32] = selected_cmps[i]
    
    img = Image.fromarray(img)
    pannel.paste(img, (blank, blank * (num + 1) + num * 32 * 3))
    
    if time_steps > obs:
        selected_gens = gens[idx]
        selected_gens = np.uint8(selected_gens * 255)
        img_gen = np.zeros((32 * 3, (time_steps - obs) * 32)).astype(np.uint8)
        for i in range(time_steps - obs):
            img_gen[:32, i * 32: (i + 1) * 32] = selected_inps[i + obs]
            img_gen[32:64, i * 32: (i + 1) * 32] = selected_gens[i]

        img_gen = Image.fromarray(img_gen)
        pannel.paste(img_gen, (blank * 2 + 32 * obs, blank * (num + 1) + num * 32 * 3))

pannel.save('{}/data/{}_sample.png'.format(exp_dir, args.tag))

# pannel = np.ones((32 * 2 * num_sample + blank * (num_sample + 2), 32 * time_steps + 3 * blank)) * 255
# pannel = np.uint8(pannel)
# pannel = Image.fromarray(pannel)

# # selected_idx = np.random.choice(all_inps.shape[0], num_sample, replace=False)
# # selected_idx = sorted(selected_idx)
# selected_idx = np.arange(num_sample)

# for num, idx in enumerate(selected_idx):
#     selected_inps = all_inps[idx]
#     selected_rcns = q_recons[idx]
#     selected_gens = gens[idx]

#     selected_inps = np.uint8(selected_inps * 255)
#     selected_rcns = np.uint8(selected_rcns * 255)
#     selected_gens = np.uint8(selected_gens * 255)

#     img = np.zeros((32 * 2, obs * 32)).astype(np.uint8)
#     for i in range(obs):
#         img[:32, i * 32: (i + 1) * 32] = selected_inps[i]
#         img[32:64, i * 32: (i + 1) * 32] = selected_rcns[i]
    
#     img = Image.fromarray(img)
#     pannel.paste(img, (blank, blank * (num + 1) + num * 32 * 2))
    
#     img_gen = np.zeros((32 * 2, (time_steps - obs) * 32)).astype(np.uint8)
#     for i in range(time_steps - obs):
#         img_gen[:32, i * 32: (i + 1) * 32] = selected_inps[i + obs]
#         img_gen[32:64, i * 32: (i + 1) * 32] = selected_gens[i]

#     img_gen = Image.fromarray(img_gen)
#     pannel.paste(img_gen, (blank * 2 + 32 * obs, blank * (num + 1) + num * 32 * 2))

# pannel.save('{}/data/{}_sample.png'.format(exp_dir, args.tag))
