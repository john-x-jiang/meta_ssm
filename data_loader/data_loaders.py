import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from data_loader.seq_util import *
from data_loader.boxes import PymunkData, PymunkEpisoticData


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


def bouncingball_collate(batch):
    """
    Collate function for the bouncing ball experiments
    Args:
        batch: given batch at this generator call

    Returns: indices of batch, images, controls
    """
    images, states, labels = [], [], []

    for b in batch:
        _, image, state, _, label = b
        images.append(image)
        states.append(state)
        labels.append(label)

    images = torch.stack(images)
    states = torch.stack(states)
    labels = torch.stack(labels)

    # B, T, W, H = images.shape

    return images, None, states, None, labels


def bouncingball_episotic_collate(batch):
    """
    Collate function for the bouncing ball experiments
    Args:
        batch: given batch at this generator call

    Returns: indices of batch, images, controls
    """
    images, states, labels, \
        images_D, states_D \
         = [], [], [], [], []

    for b in batch:
        _, image, state, _, label, image_D, state_D, _ = b
        images.append(image)
        states.append(state)
        labels.append(label)
        images_D.append(image_D)
        states_D.append(state_D)

    images = torch.stack(images)
    states = torch.stack(states)
    labels = torch.stack(labels)
    images_D = torch.stack(images_D)
    states_D = torch.stack(states_D)

    # B, T, W, H = images.shape

    return images, images_D, states, states_D, labels


class BouncingBallDataLoader(BaseDataLoader):
    """
    Dataloader for the base implementation of bouncing ball w/ gravity experiments, available here:
    https://github.com/simonkamronn/kvae
    """
    def __init__(self, batch_size, data_dir='data/box_data', split='train', shuffle=True,
                 collate_fn=bouncingball_collate, num_workers=1, data_name=None, k_shot=3):
        # assert split in ['train', 'valid', 'test', 'pred']

        # Generate dataset and initialize loader
        # config = {'dataset': 'mixed_gravity', 'out_distr': None, 'dim_u': 1}
        config = {
            'dataset': data_name,
            'out_distr': 'bernoulli',
            'dim_u': 1,
            'k_shot': k_shot,
            'shuffle': shuffle,
            'is_train': split
        }

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'validation_split': 0.0,
            'num_workers': num_workers,
            'collate_fn': collate_fn
        }

        self.dataset = PymunkData("{}/{}_{}.npz".format(data_dir, config['dataset'], split), config)
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset, **self.init_kwargs)


class BouncingBallEpisoticDataLoader(BaseDataLoader):
    """
    Dataloader for the base implementation of bouncing ball w/ gravity experiments, available here:
    https://github.com/simonkamronn/kvae
    """
    def __init__(self, batch_size, data_dir='data/box_data', split='train', shuffle=True,
                 collate_fn=bouncingball_episotic_collate, num_workers=1, data_name=None, k_shot=3):
        # assert split in ['train', 'valid', 'test', 'pred']

        # Generate dataset and initialize loader
        # config = {'dataset': 'mixed_gravity', 'out_distr': None, 'dim_u': 1}
        config = {
            'dataset': data_name,
            'out_distr': 'bernoulli',
            'dim_u': 1,
            'k_shot': k_shot,
            'shuffle': shuffle,
        }

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'validation_split': 0.0,
            'num_workers': num_workers,
            'collate_fn': collate_fn
        }

        self.dataset = PymunkEpisoticData("{}/{}_{}.npz".format(data_dir, config['dataset'], split), config)
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset, **self.init_kwargs)

    def next(self):
        self.dataset.split()
        self.init_kwargs['validation_split'] = 0.0
        return BaseDataLoader(dataset=self.dataset,
                              batch_size=self.init_kwargs['batch_size'],
                              shuffle=self.init_kwargs['shuffle'],
                              validation_split=self.init_kwargs['validation_split'],
                              num_workers=self.init_kwargs['num_workers'],
                              collate_fn=self.init_kwargs['collate_fn'])


class DataLoader(BaseDataLoader):
    """
    Dataloader for the base implementation of bouncing ball w/ gravity experiments, available here:
    https://github.com/simonkamronn/kvae
    """
    def __init__(self, batch_size, data_dir='data/box_data', split='train', shuffle=True,
                 collate_fn=bouncingball_collate, num_workers=1, out_distr='bernoulli', data_name=None, k_shot=3):
        # assert split in ['train', 'valid', 'test', 'pred']

        # Generate dataset and initialize loader
        # config = {'dataset': 'mixed_gravity', 'out_distr': None, 'dim_u': 1}
        config = {
            'dataset': data_name,
            'out_distr': out_distr,
            'dim_u': 1,
            'k_shot': k_shot,
            'shuffle': shuffle,
            'is_train': split
        }

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'validation_split': 0.0,
            'num_workers': num_workers,
            'collate_fn': collate_fn
        }

        self.dataset = PymunkData("{}/{}_{}.npz".format(data_dir, config['dataset'], split), config)
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset, **self.init_kwargs)


class EpisoticDataLoader(BaseDataLoader):
    """
    Dataloader for the base implementation of bouncing ball w/ gravity experiments, available here:
    https://github.com/simonkamronn/kvae
    """
    def __init__(self, batch_size, data_dir='data/box_data', split='train', shuffle=True,
                 collate_fn=bouncingball_episotic_collate, num_workers=1, out_distr='bernoulli', data_name=None, k_shot=3):
        # assert split in ['train', 'valid', 'test', 'pred']

        # Generate dataset and initialize loader
        # config = {'dataset': 'mixed_gravity', 'out_distr': None, 'dim_u': 1}
        config = {
            'dataset': data_name,
            'out_distr': out_distr,
            'dim_u': 1,
            'k_shot': k_shot,
            'shuffle': shuffle,
        }

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'validation_split': 0.0,
            'num_workers': num_workers,
            'collate_fn': collate_fn
        }

        self.dataset = PymunkEpisoticData("{}/{}_{}.npz".format(data_dir, config['dataset'], split), config)
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset, **self.init_kwargs)

    def next(self):
        self.dataset.split()
        self.init_kwargs['validation_split'] = 0.0
        return BaseDataLoader(dataset=self.dataset,
                              batch_size=self.init_kwargs['batch_size'],
                              shuffle=self.init_kwargs['shuffle'],
                              validation_split=self.init_kwargs['validation_split'],
                              num_workers=self.init_kwargs['num_workers'],
                              collate_fn=self.init_kwargs['collate_fn'])


if __name__ == '__main__':
    """ 
    Test to check for each dataset to be loaded in and working at a batch level
    """
    import matplotlib.pyplot as plt

    def movie_to_frame(images):
        """ Compiles a list of images into one composite frame """
        n_steps, w, h = images.shape
        colors = np.linspace(0.4, 1, n_steps)
        image = np.zeros((w, h))
        for i, color in zip(images, colors):
            image = np.clip(image + i * color, 0, color)
        return image

    # Mocap test
    dataloader = MocapDataLoader(batch_size=4, data_dir='../data/ode_exp/', split='train')

    for idx, X, y in dataloader:
        print(y)
        print(idx, X.shape, y.shape)

        tt = X.shape[1]
        D = X.shape[2]
        nrows = np.ceil(D / 5)

        plt.figure(2, figsize=(20, 40))
        for i in range(D):
            plt.subplot(nrows, 5, i + 1)
            plt.plot(range(0, tt), X[0, :, i], 'r.')

        plt.show()
        plt.close()
        break

    # Mixed bouncing ball test
    dataloader = MixGravityDataLoader(batch_size=32, data_dir='../data/box_data/', split='train')

    # Check one batch for its size and composite image of one data point
    i = 0
    for idx, images, controls in dataloader:
        if i > 5:
            break

        print(idx.shape, images.shape, controls.shape)
        plt.imshow(movie_to_frame(images[0].cpu().numpy()), cmap='gray')
        plt.show()
        i += 1