import os
import numpy as np

from data_loader.heart_data import HeartGraphDataset, HeartEpisodicDataset
from torch_geometric.loader import DataLoader


class HeartDataLoader(DataLoader):
    def __init__(self, batch_size, data_dir='data/', split='train', shuffle=True, collate_fn=None,
                 num_workers=1, data_name=None, signal_type=None, num_mesh=None, seq_len=None, k_shot=None):
        # assert split in ['train', 'valid', 'test', 'test00', 'test01','test10', 'test11']

        self.dataset = HeartGraphDataset(data_dir, data_name, signal_type, num_mesh, seq_len, split)

        super().__init__(self.dataset, batch_size, shuffle, drop_last=True, num_workers=num_workers)


class HeartEpisodicDataLoader(DataLoader):
    def __init__(self, batch_size, data_dir='data/', split='train', shuffle=True, collate_fn=None,
                 num_workers=1, data_name=None, signal_type=None, num_mesh=None, seq_len=None, k_shot=None):
        self.dataset = HeartEpisodicDataset(data_dir, data_name, signal_type, num_mesh, seq_len, split, shuffle, k_shot=k_shot)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'drop_last': True,
            'num_workers': num_workers
        }
        super().__init__(dataset=self.dataset, **self.init_kwargs)
    
    def next(self):
        self.dataset.split()
        return DataLoader(dataset=self.dataset, **self.init_kwargs)
