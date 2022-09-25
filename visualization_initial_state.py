import os
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
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

    parser.add_argument('--config', type=str, default='b01', help='config filename')
    args = parser.parse_args()
    return args


def show_images(images, exp_dir, qry, state, num_out=None):
    """
    Constructs an image of multiple time-series reconstruction samples compared against its relevant ground truth
    Saves locally in the given out location
    :param images: ground truth images
    :param preds: predictions from a given model
    :out_loc: where to save the generated image
    :param num_out: how many images to stack. If None, stack all
    """
    assert len(images.shape) == 4       # Assert both matrices are [Batch, Timesteps, H, W]
    assert type(num_out) is int or type(num_out) is None

    # Splice to the given num_out
    if num_out is not None:
        images = images[:num_out]

    # Iterate through each sample, stacking into one image
    out_image = None
    for idx, gt in enumerate(images):
        # Pad between individual timesteps
        final = np.pad(gt, pad_width=(
            (0, 0), (5, 5), (0, 5)
        ), constant_values=1)

        final = np.hstack([i for i in final])

        # Stack into out_image
        if out_image is None:
            out_image = final
        else:
            out_image = np.vstack((out_image, final))

    # Save to out location
    # plt.imsave(out_loc, out_image)
    plt.imshow(out_image, cmap='gray')
    plt.axis('off')
    # plt.show()
    plt.savefig('{}/embeddings/qry_{}init_{}.png'.format(exp_dir, qry, state), format='png', bbox_inches='tight')


np.random.seed(123)
args = parse_args()

# filename of the params
fname_config = args.config + '.json'
# read the params file
json_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), "config", fname_config)
hparams = Params(json_path)
exp_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'experiments', hparams.exp_name, hparams.exp_id)

# Handles loading in the stack of initial_state reconstructions from the K-Means algorithm
# and plotting as a stacked thing against the ground truth
for qry in [0,]:
    for state in [1,]:
        data = np.load('{}/embeddings/predictions_qry_0_initial_state_1.npy'.format(exp_dir))[:, 1]
        print(data.shape)
        show_images(data, exp_dir, qry, state, num_out=10)
