import os.path as osp
import numpy as np

import scipy.io
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
# from torch_geometric.data import InMemoryDataset


class DataWithDomain(Data):
    def __init__(self, x, y, pos, mask=None, D=None, D_label=None):
        super().__init__()
        self.x = x
        self.y = y
        self.pos = pos
        self.mask = mask
        self.D = D
        self.D_label = D_label


class HeartGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. 
    """

    def __init__(self,
                 root,
                 data_name,
                 signal_type='egm',
                 num_mesh=None,
                 seq_len=None,
                 split='train',
                 subset=1):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'signal/{}/'.format(data_name))

        filename = '{}_{}_{}.mat'.format(split, signal_type, num_mesh)
        self.data_path = osp.join(self.raw_dir, filename)
        matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
        dataset = matFiles['params']
        label = matFiles['label']
        # mask = matFiles['inf_idx']

        dataset = dataset.transpose(2, 0, 1)

        N = dataset.shape[0]
        if subset == 1:
            index = np.arange(N)
        elif subset == 0:
            raise RuntimeError('No data')
        else:
            indices = list(range(N))
            np.random.shuffle(indices)
            split = int(np.floor(subset * N))
            sub_index = indices[:split]
            dataset = dataset[sub_index, :, :]
            index = np.arange(dataset.shape[1])
        
        label = label.astype(int)
        self.label = torch.from_numpy(label[index])
        self.data = torch.from_numpy(dataset[index, :, :]).float()
        # self.corMfree = corMfree
        # self.mask = mask
        self.heart_name = data_name
        print('final data size: {}'.format(self.data.shape[0]))

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        x = self.data[[idx], :, :]
        y = self.label[[idx]]
        sample = DataWithDomain(
            x=x,
            y=y,
            pos=self.heart_name,
            # mask=self.mask
        )
        return sample


class HeartEpisodicDataset_old(Dataset):
    def __init__(self,
                 root,
                 data_name,
                 signal_type='egm',
                 num_mesh=None,
                 seq_len=None,
                 split='train',
                 shuffle=True,
                 subset=1,
                 k_shot=2):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'signal/{}/'.format(data_name))
        self.k_shot = k_shot
        self.is_train = split
        self.shuffle = shuffle

        filename = '{}_{}_{}.mat'.format(split, signal_type, num_mesh)
        self.data_path = osp.join(self.raw_dir, filename)
        matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
        dataset = matFiles['params']
        label = matFiles['label']
        mask = matFiles['inf_idx']

        dataset = dataset.transpose(2, 0, 1)

        N = dataset.shape[0]
        if subset == 1:
            index = np.arange(N)
        elif subset == 0:
            raise RuntimeError('No data')
        else:
            indices = list(range(N))
            np.random.shuffle(indices)
            split = int(np.floor(subset * N))
            sub_index = indices[:split]
            dataset = dataset[sub_index, :, :]
            index = np.arange(dataset.shape[1])
        
        label = label.astype(int)
        self.label = torch.from_numpy(label[index])
        self.data = torch.from_numpy(dataset[index, :, :]).float()
        # self.corMfree = corMfree
        self.mask = mask
        self.heart_name = data_name

        scar = self.label[:, 1]
        scar = np.unique(scar)
        self.scar_idx = {}
        for s in scar:
            idx = np.where(self.label[:, 1] == s)[0]
            self.scar_idx[s] = idx

        print('final data size: {}'.format(self.data.shape[0]))
        np.random.seed(0)
        self.split()

    def __len__(self):
        # return (self.data.shape[0])
        return self.x_qry.shape[0]

    def __getitem__(self, idx):
        if self.shuffle:
            x = self.x_qry[[idx], :, :]
            y = self.y_qry[[idx]]
        else:
            x = self.data[[idx], :, :]
            y = self.label[[idx]]

        scar = y[:, 1].numpy()[0]
        D_x = self.x_spt[scar]
        D_y = self.y_spt[scar]
        num_sample = min(len(self.scar_idx[scar]), self.k_shot)
        D_y = D_y.view(1, num_sample, -1)

        sample = DataWithDomain(
            x=x,
            y=y,
            pos=self.heart_name,
            mask=self.mask,
            D=D_x,
            D_label=D_y
        )
        return sample
    
    def split(self):
        self.x_spt, self.y_spt = {}, {}
        self.x_qry, self.y_qry = [], []
        for scar_id, samples in self.scar_idx.items():
            sample_idx = np.arange(0, len(samples))
            if len(samples) < self.k_shot:
                self.x_spt[scar_id] = self.data[samples, :]
                self.y_spt[scar_id] = self.label[samples, :]
            else:
                np.random.shuffle(sample_idx)
                spt_idx = np.sort(sample_idx[0:self.k_shot])
                self.x_spt[scar_id] = self.data[samples[spt_idx], :]
                self.y_spt[scar_id] = self.label[samples[spt_idx], :]

            if self.shuffle:
                qry_idx = sample_idx[self.k_shot:]
                self.x_qry.append(self.data[samples[qry_idx], :])
                self.y_qry.append(self.label[samples[qry_idx], :])
            else:
                self.x_qry.append(self.data[samples, :])
                self.y_qry.append(self.label[samples, :])
        
        self.x_qry = torch.cat(self.x_qry, dim=0)
        self.y_qry = torch.cat(self.y_qry, dim=0)


class HeartEpisodicDataset(Dataset):
    def __init__(self,
                 root,
                 data_name,
                 signal_type='egm',
                 num_mesh=None,
                 seq_len=None,
                 split='train',
                 shuffle=True,
                 subset=1,
                 k_shot=2):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'signal/{}/'.format(data_name))
        self.k_shot = k_shot
        self.split_name = split
        self.shuffle = shuffle

        filename = '{}_{}_{}.mat'.format(split, signal_type, num_mesh)
        self.data_path = osp.join(self.raw_dir, filename)
        matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
        dataset = matFiles['params']
        label = matFiles['label']
        # mask = matFiles['inf_idx']

        dataset = dataset.transpose(2, 0, 1)

        N = dataset.shape[0]
        if subset == 1:
            index = np.arange(N)
        elif subset == 0:
            raise RuntimeError('No data')
        else:
            indices = list(range(N))
            np.random.shuffle(indices)
            split = int(np.floor(subset * N))
            sub_index = indices[:split]
            dataset = dataset[sub_index, :, :]
            index = np.arange(dataset.shape[1])
        
        label = label.astype(int)
        self.label = torch.from_numpy(label[index])
        self.data = torch.from_numpy(dataset[index, :, :]).float()
        # self.corMfree = corMfree
        # self.mask = mask
        self.heart_name = data_name

        scar = self.label[:, 1]
        scar = np.unique(scar)
        self.scar_idx = {}
        for s in scar:
            idx = np.where(self.label[:, 1] == s)[0]
            self.scar_idx[s] = idx

        print('final data size: {}'.format(self.data.shape[0]))
        np.random.seed(0)
        self.split()

    def __len__(self):
        return (self.qry_idx.shape[0])

    def __getitem__(self, idx):
        y = self.label[[self.qry_idx[idx]]]
        x = self.data[[self.qry_idx[idx]], :, :]

        scar = y[:, 1].numpy()[0]
        D_x = self.data[self.spt_idx[scar], :, :]
        D_y = self.label[self.spt_idx[scar]]
        num_sample = min(len(self.scar_idx[scar]), self.k_shot)
        D_y = D_y.view(1, num_sample, -1)

        sample = DataWithDomain(
            x=x,
            y=y,
            pos=self.heart_name,
            # mask=self.mask,
            D=D_x,
            D_label=D_y
        )
        return sample
    
    def split(self):
        self.spt_idx = {}
        self.qry_idx = []
        for scar_id, samples in self.scar_idx.items():
            sample_idx = np.arange(0, len(samples))
            if len(samples) < self.k_shot:
                self.spt_idx = samples
            else:
                if self.shuffle:
                    np.random.shuffle(sample_idx)
                    spt_idx = np.sort(sample_idx[0:self.k_shot])
                else:
                    spt_idx = sample_idx[0:self.k_shot]
                self.spt_idx[scar_id] = samples[spt_idx]
            
            self.qry_idx.extend(samples.tolist())
        
        self.qry_idx = np.array(self.qry_idx)
        self.qry_idx = np.sort(self.qry_idx)


class HeartEmptyGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. The features and target values are 
    set to zeros in given graph.
    Not suitable for training.
    """

    def __init__(self,
                 mesh_graph,
                 label_type=None):
        self.graph = mesh_graph
        dim = self.graph.pos.shape[0]
        self.datax = np.zeros((dim, 101))
        self.label = np.zeros((101))

    def __len__(self):
        return (self.datax.shape[1])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.datax[:, [idx]]).float()  # torch.tensor(dataset[:,[i]],dtype=torch.float)
        y = torch.from_numpy(self.label[[idx]]).float()  # torch.tensor(label_aha[[i]],dtype=torch.float)

        sample = Data(x=x,
                      y=y,
                      edge_index=self.graph.edge_index,
                      edge_attr=self.graph.edge_attr,
                      pos=self.graph.pos)
        return sample
