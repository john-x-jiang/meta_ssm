"""
Data preparation code bought from
https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py
"""

import os
from collections import namedtuple
from urllib.request import urlopen
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data_loader.seq_util import get_data_directory


dset = namedtuple("dset", ["name", "url", "filename"])

JSB_CHORALES = dset("jsb_chorales",
                    "https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/jsb_chorales.pickle",
                    "jsb_chorales.pkl")

PIANO_MIDI = dset("piano_midi",
                  "https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/piano_midi.pickle",
                  "piano_midi.pkl")

MUSE_DATA = dset("muse_data",
                 "https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/muse_data.pickle",
                 "muse_data.pkl")

NOTTINGHAM = dset("nottingham",
                  "https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/nottingham.pickle",
                  "nottingham.pkl")


# this function processes the raw data; in particular it unsparsifies it
def process_data(base_path, dataset, min_note=21, note_range=88):
    output = os.path.join(base_path, dataset.filename)
    if os.path.exists(output):
        try:
            with open(output, "rb") as f:
                return pickle.load(f)
        except (ValueError, UnicodeDecodeError):
            # Assume python env has changed.
            # Recreate pickle file in this env's format.
            os.remove(output)

    print("processing raw data - {} ...".format(dataset.name))
    data = pickle.load(urlopen(dataset.url))
    # added this line to see the difference between the raw and processed data
    pickle.dump(data, open(os.path.join(base_path,
                '-'.join(['raw', dataset.filename])), "wb"), pickle.HIGHEST_PROTOCOL)
    processed_dataset = {}
    for split, data_split in data.items():
        processed_dataset[split] = {}
        n_seqs = len(data_split)
        processed_dataset[split]['sequence_lengths'] = torch.zeros(n_seqs, dtype=torch.long)
        processed_dataset[split]['sequences'] = []
        for seq in range(n_seqs):
            seq_length = len(data_split[seq])
            processed_dataset[split]['sequence_lengths'][seq] = seq_length
            processed_sequence = torch.zeros((seq_length, note_range))
            for t in range(seq_length):
                note_slice = torch.tensor(list(data_split[seq][t])) - min_note
                slice_length = len(note_slice)
                if slice_length > 0:
                    processed_sequence[t, note_slice] = torch.ones(slice_length)
            processed_dataset[split]['sequences'].append(processed_sequence)
    pickle.dump(processed_dataset, open(output, "wb"), pickle.HIGHEST_PROTOCOL)
    print("dumped processed data to %s" % output)


# this logic will be initiated upon import
base_path = get_data_directory(__file__)
if not os.path.exists(base_path):
    os.mkdir(base_path)


# ingest training/validation/test data from disk
def load_data(dataset):
    # download and process dataset if it does not exist
    process_data(base_path, dataset)
    file_loc = os.path.join(base_path, dataset.filename)

    with open(file_loc, "rb") as f:
        dset = pickle.load(f)
        for k, v in dset.items():
            sequences = v["sequences"]
            dset[k]["sequences"] = pad_sequence(sequences, batch_first=True).type(torch.Tensor)
            dset[k]["sequence_lengths"] = v["sequence_lengths"].to(device=torch.Tensor().device)
    return dset


class PolyDataset(Dataset):
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split

        self.data = load_data(dataset)[split]
        self.seq_lengths = self.data['sequence_lengths']
        self.seq = self.data['sequences']
        self.n_seq = len(self.seq_lengths)
        self.n_time_slices = float(torch.sum(self.seq_lengths))

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        return idx, self.seq[idx], self.seq_lengths[idx]
