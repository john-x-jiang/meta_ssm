import os
import torch
import torch.nn as nn


def reverse_sequence(x, seq_lengths):
    """
    Brought from
    https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py

    Parameters
    ----------
    x: tensor (b, T_max, input_dim)
    seq_lengths: tensor (b, )

    Returns
    -------
    x_reverse: tensor (b, T_max, input_dim)
        The input x in reversed order w.r.t. time-axis
    """
    x_reverse = torch.zeros_like(x)
    for b in range(x.size(0)):
        t = seq_lengths[b]
        time_slice = torch.arange(t - 1, -1, -1, device=x.device)
        reverse_seq = torch.index_select(x[b, :, :], 0, time_slice)
        x_reverse[b, 0:t, :] = reverse_seq

    return x_reverse


def binary_mask(x_state, W, H):
    mask = torch.zeros([x_state.shape[0], x_state.shape[1], W, H])
    for i in range(x_state.shape[0]):
        for j in range(x_state.shape[1]):
            x = x_state[i, j, 0].int()
            y = x_state[i, j, 1].int()
            mask[i, j, x, y] = 1
    return mask


def pad_and_reverse(rnn_output, seq_lengths):
    """
    Brought from
    https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py

    Parameters
    ----------
    rnn_output: tensor  # shape to be confirmed, should be packed rnn output
    seq_lengths: tensor (b, )

    Returns
    -------
    reversed_output: tensor (b, T_max, input_dim)
        The input sequence, unpacked and padded,
        in reversed order w.r.t. time-axis
    """
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output,
                                                     batch_first=True)
    reversed_output = reverse_sequence(rnn_output, seq_lengths)
    return reversed_output


def get_mini_batch_mask(x, seq_lengths):
    """
    Brought from
    https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py

    Parameters
    ----------
    x: tensor (b, T_max, input_dim)
    seq_lengths: tensor (b, )

    Returns
    -------
    mask: tensor (b, T_max)
        A binary mask generated according to `seq_lengths`
    """
    mask = torch.zeros(x.shape[0:2])
    for b in range(x.shape[0]):
        mask[b, 0:seq_lengths[b]] = torch.ones(seq_lengths[b])
    return mask


def get_mini_batch(mini_batch_indices, sequences, seq_lengths, cuda=True):
    """
    Prepare a mini-batch (size b) from the dataset (size D)
    for training or evaluation

    Brought from
    https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py

    Parameters
    ----------
    mini_batch_indices: tensor (b, )
        Indices of a mini-batch of data
    sequences: tensor (D, D_T_max, input_dim)
        Padded data
    seq_lengths: tensor (D, )
        Effective sequence lengths of each sequence in the dataset
    cuda: bool

    Returns
    -------
    mini_batch: tensor (b, T_max, input_dim)
        A mini-batch from the dataset
    mini_batch_reversed: pytorch packed object
        A mini-batch in the reversed order;
        used as the input to RnnEncoder in DeepMarkovModel
    """
    seq_lengths = seq_lengths[mini_batch_indices]
    _, sorted_seq_length_indices = torch.sort(seq_lengths)
    sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]

    T_max = torch.max(seq_lengths)
    mini_batch = sequences[sorted_mini_batch_indices, 0:T_max, :]
    mini_batch_reversed = reverse_sequence(mini_batch, sorted_seq_lengths)
    mini_batch_mask = get_mini_batch_mask(mini_batch, sorted_seq_lengths)

    if cuda:
        mini_batch = mini_batch.cuda()
        mini_batch_mask = mini_batch_mask.cuda()
        mini_batch_reversed = mini_batch_reversed.cuda()

    mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(
        mini_batch_reversed,
        sorted_seq_lengths,
        batch_first=True
    )

    return mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths


def get_data_directory(filepath=None):
    """
    Brought from
    https://github.com/pyro-ppl/pyro/blob/2b4a4013291e59f251564aeaf5815c4c3a18f4ff/pyro/contrib/examples/util.py#L66
    """
    if 'CI' in os.environ:
        return os.path.expanduser('~/.data')
    return os.path.abspath(os.path.join(os.path.dirname(filepath),
                                        '.data'))


def seq_collate_fn(batch):
    """
    A customized `collate_fn` intented for loading padded sequential data
    """
    idx, seq, seq_lengths = zip(*batch)
    idx = torch.tensor(idx)
    seq = torch.stack(seq)
    seq_lengths = torch.tensor(seq_lengths)
    _, sorted_seq_length_indices = torch.sort(seq_lengths)
    sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]

    T_max = torch.max(seq_lengths)
    mini_batch = seq[sorted_seq_length_indices, 0:T_max, :]
    mini_batch_reversed = reverse_sequence(mini_batch, sorted_seq_lengths)
    mini_batch_mask = get_mini_batch_mask(mini_batch, sorted_seq_lengths)

    return mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths


def pack_padded_seq(seq, seq_len, batch_first=True):
    # import ipdb; ipdb.set_trace()
    seq = torch.as_tensor(seq, device='cpu')
    seq_len = torch.as_tensor(seq_len, device='cpu')
    rtn = nn.utils.rnn.pack_padded_sequence(
        seq,
        seq_len,
        batch_first=batch_first
    )
    device = torch.device('cuda')
    rtn = rtn.to(device)
    return rtn
    # return nn.utils.rnn.pack_padded_sequence(
    #     seq,
    #     seq_len,
    #     batch_first=batch_first
    # )
