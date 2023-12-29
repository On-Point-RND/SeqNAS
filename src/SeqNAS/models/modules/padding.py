import torch.nn.functional as F


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
    return min(sizes)
