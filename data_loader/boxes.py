"""
@file boxes.py
@author Ryan Missel

Handles generating the datasets for the box experiments under a number of situations
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class PymunkData(Dataset):
    """
    Load sequences of images
    """
    def __init__(self, file_path, config):
        self.k_shot = config['k_shot']
        self.shuffle = config['shuffle']
        self.is_train = config['is_train']
        # Load data
        npzfile = np.load(file_path)
        self.images = npzfile['images'].astype(np.float32)
        if config['out_distr'] == 'bernoulli':
            self.images = (self.images > 0).astype('float32')

        # self.images = torch.from_numpy(self.images).to(device=torch.Tensor().device)

        self.labels = npzfile['label'].astype(np.int16)
        self.label_idx = {}
        for label in np.unique(self.labels):
            idx = np.where(self.labels == label)[0]
            self.label_idx[label] = idx
        # self.labels = torch.from_numpy(self.labels).to(device=torch.Tensor().device)

        # Load ground truth position and velocity (if present). This is not used in the KVAE experiments in the paper.
        if 'state' in npzfile:
            # Only load the position, not velocity
            self.state = npzfile['state'].astype(np.float32)[:, :, :2]
            self.velocity = npzfile['state'].astype(np.float32)[:, :, 2:]

            # Normalize the mean
            # self.state = self.state - self.state.mean(axis=(0, 1))

            # Set state dimension
            self.state_dim = self.state.shape[-1]

        # Get data dimensions
        self.sequences, self.timesteps, self.d1, self.d2 = self.images.shape

        # We set controls to zero (we don't use them even if dim_u=1). If they are fixed to one instead (and dim_u=1)
        # the B matrix represents a bias.
        self.controls = np.zeros((self.sequences, self.timesteps, config['dim_u']), dtype=np.float32)
        # self.controls = torch.from_numpy(self.controls).to(device=torch.Tensor().device)

        np.random.seed(0)
        self.split()

    def __len__(self):
        return self.qry_idx.shape[0]

    def __getitem__(self, idx):
        """ Couple images and controls together for compatibility with other datasets """
        label_qry = self.labels[self.qry_idx[idx]]
        image_qry = self.images[self.qry_idx[idx], :]
        state_qry = self.state[self.qry_idx[idx], :]
        control_qry = self.controls[self.qry_idx[idx], :]
        image_spt = self.images[self.spt_idx[label_qry], :]
        state_spt = self.state[self.spt_idx[label_qry], :]
        control_spt = self.controls[self.spt_idx[label_qry], :]

        image_qry = torch.from_numpy(image_qry)
        state_qry = torch.from_numpy(state_qry)
        control_qry = torch.from_numpy(control_qry)
        label_qry = torch.Tensor([label_qry])
        image_spt = torch.from_numpy(image_spt)
        state_spt = torch.from_numpy(state_spt)
        control_spt = torch.from_numpy(control_spt)

        return torch.Tensor([idx]), image_qry, state_qry, control_qry, label_qry, \
            image_spt, state_spt, control_spt
    
    def split(self):
        self.spt_idx = {}
        self.qry_idx = []
        for label_id, samples in self.label_idx.items():
            sample_idx = np.arange(0, len(samples))
            if len(samples) < self.k_shot:
                self.spt_idx[label_id] = samples
            else:
                if self.shuffle:
                    np.random.shuffle(sample_idx)
                    spt_idx = np.sort(sample_idx[0:self.k_shot])
                else:
                    spt_idx = sample_idx[0:self.k_shot]
                self.spt_idx[label_id] = samples[spt_idx]

            if self.is_train == 'train' or self.is_train == 'valid':
                qry_idx = sample_idx[self.k_shot:]
                self.qry_idx.extend(samples[qry_idx].tolist())
            else:
                self.qry_idx = samples
        
        self.qry_idx = np.array(self.qry_idx)
        self.qry_idx = np.sort(self.qry_idx)


if __name__ == '__main__':
    config = {'dataset': 'polygon_train', 'out_distr': 'asd', 'dim_u': 1}

    dataset = PymunkData("box_data/{}.npz".format(config['dataset']), config)
    print(dataset.images.shape)
    print(dataset.images[0].shape)

    for image in dataset.images[np.random.randint(0, dataset.images.shape[0], 1)[0]]:
        plt.imshow(image)
        plt.show()
