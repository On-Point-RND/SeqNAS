import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import DataLoader


def collate_sequences(batch):
    ids, sequences, target = zip(*batch)
    padded = {col: list() for col in sequences[0]}
    for feat_sequence in sequences:
        for col, seq in feat_sequence.items():
            padded[col].append(torch.as_tensor(seq))
    padded = {
        col: pad_packed_sequence(pack_sequence(data, False))
        for col, data in padded.items()
    }
    lengths = next(iter(padded.values()))[-1]
    padded = {col: data[0] for col, data in padded.items()}
    target = torch.as_tensor(target, dtype=torch.float32)
    return (padded, lengths), target
