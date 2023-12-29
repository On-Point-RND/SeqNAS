import torch.nn as nn
import torch.nn.functional as F
from apex.normalization import FusedLayerNorm


def length_padder_period(seq, seq_len, period):
    if seq_len % period != 0:
        n_pad = period - (seq_len % period)
        seq = F.pad(seq, (0, n_pad), "constant")
        seq_len = seq_len + n_pad
    return seq, seq_len


def length_padder(seq, seq_len, max_len):
    n_pad = max_len - seq_len
    if n_pad > 0:
        seq = F.pad(seq, (0, n_pad), "constant")
    return seq


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class LenPadder:
    def __init__(self, pad_size):
        self.pad_size = pad_size

    def __call__(self, seq):
        return F.pad(seq, (0, self.pad_size), "constant")


def find_min_padding(size, divisor, step=1):
    leftover, pad_size = min(
        ((size + pad) % divisor, pad) for pad in range(0, size, step)
    )
    return pad_size


def get_min_seq_len(l, periods, w):
    sizes = []
    for p in periods:
        pad_size = find_min_padding(l, p)
        lt = l + pad_size
        f_padding = find_min_padding(lt // p, w)
        sizes.append((lt / p + f_padding) // w * p)
    return int(min(sizes))


class FreqFilter(nn.Module):
    def __init__(self, period, window, features, seq_len, min_seq_len, dropout=0.2):
        """
        The idea here is quite simple but we need to do a lot tiny steps to get correct the same size
        outptuts due to integer division and paddings.
        """
        super().__init__()
        self.p = period
        self.w = window
        self.f = features
        self.m = int(min_seq_len)

        # it is hacky but easier with paddings
        if self.p % 2 == 0:
            self.p += 1

        pad_size = find_min_padding(seq_len, self.p)
        self.l = seq_len + pad_size
        self.padder = LenPadder(pad_size)

        self.f_padding = find_min_padding(self.l // self.p, self.w)
        self.filter = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(
                self.f,
                self.f,
                kernel_size=(1, self.p),
                padding=(0, self.p // 2),
                stride=(1, 1),
            ),
            nn.PReLU(),
            nn.Conv2d(
                self.f,
                self.f,
                kernel_size=(window, 1),
                padding=(self.f_padding, 0),
                stride=(window, 1),
            ),
            nn.PReLU(),
        )
        self.norm = FusedLayerNorm([self.f, self.m])

    def forward(self, seq):
        # B X Features X SEQ_LEN
        seq = seq.transpose(1, 2)
        seq = self.padder(seq)
        seq = seq.view(-1, self.f, self.l // self.p, self.p)
        seq = self.filter(seq)

        seq = seq.view(
            -1,
            self.f,
            (self.l // self.p + self.f_padding) // self.w * self.p,
        )
        return self.norm(seq[:, :, : self.m])
