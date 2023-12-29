import torch
import numpy as np
import random
from random import randint
from math import ceil


def ts_mul_noise(x, seq_len=None):
    """
    Adds multiplicative noise to x.

    :param x: input tensor
    :type x: torch.tensor with shape (Batch, Len, Embedding)
    :param seq_len: Len in x.shape, defaults to 0
    :type seq_len: int, optional

    :return: x with multiplicative noise
    :rtype: torch.tensor with shape (Batch, Len, Embedding)
    """
    return x * (torch.randn_like(x) * 0.1 + 1)


def add_noise(x, seq_len=None):
    """
    Adds additive noise to x.

    :param x: input tensor
    :type x: torch.tensor with shape (Batch, Len, Embedding)
    :param seq_len: Len in x.shape, defaults to 0
    :type seq_len: int, optional

    :return: x with additive noise
    :rtype: torch.tensor with shape (Batch, Len, Embedding)
    """
    return x + torch.randn_like(x) * 0.005


def total_permute(x, seq_len=None):
    """
    Permutes all the values in the batch - use for OOD detection
    """
    b, seq_len = x.shape
    indexes = torch.randperm(seq_len)
    data = x[torch.arange(b)]
    data = x[:, indexes]
    return data


def random_window_transform_big(x, seq_len=0, window=11):
    device = x.device
    coefs = [10 * (random.random() - 0.5) for _ in range(window)]
    weight = torch.tensor([[coefs]])
    weight = weight.to(device)
    x = x.unsqueeze(1)
    x = torch.nn.functional.conv1d(x, weight, bias=None, stride=1, padding=window // 2)
    return x.squeeze(1)


def random_window_transform_small(x, seq_len=0, window=3):
    device = x.device
    coefs = torch.tensor([random.random() for _ in range(window)])
    coefs = torch.nn.functional.softmax(coefs, dim=0).tolist()
    weight = torch.tensor([[coefs]])
    weight = weight.to(device)
    x = x.unsqueeze(1)
    x = torch.nn.functional.conv1d(x, weight, bias=None, stride=1, padding=window // 2)
    return x.squeeze(1)


def block_permute(x, seq_len, p=0.8):
    k = random.randint(2, 5)
    if random.random() > p:
        half_one = seq_len // k
        half_two = seq_len - half_one
        x_new = torch.zeros_like(x)
        x_new[:, half_two : half_one + half_two] = x[:, :half_one]
        x_new[:, :half_two] = x[:, half_one:]
        return x_new
    else:
        return x


def zero_noise(x, seq_len=0):
    """
    Adds zero noise to x.

    :param x: input tensor
    :type x: torch.tensor with shape (Batch, Len, Embedding)
    :param seq_len: Len in x.shape, defaults to 0
    :type seq_len: int, optional

    :return: x with zero noise
    :rtype: torch.tensor with shape (Batch, Len, Embedding)
    """
    return x * (torch.randn_like(x) < 0.7).type(torch.float)


def shift(x, seq_len):
    """
    Shifts x at Len dim in random number of steps.

    :param x: input tensor
    :type x: torch.tensor with shape (Batch, Len, Embedding)
    :param seq_len: Len in x.shape, defaults to 0
    :type seq_len: int, optional

    :return: x with shift
    :rtype: torch.tensor with shape (Batch, Len, Embedding)
    """
    n = seq_len
    s = randint(int(n * 0.05), int(n * 0.1))
    shift = torch.cat(
        [x.new_zeros((n, s)), x.new_zeros((n, n - s)).fill_diagonal_(1)], axis=1
    )
    return shift.matmul(x)


def permute(x, seq_len):
    """
    Randomly permutes from 5 to 10 percents of timesteps.

    :param x: input tensor
    :type x: torch.tensor with shape (Batch, Len, Embedding)
    :param seq_len: Len in x.shape, defaults to 0
    :type seq_len: int, optional

    :return: x with permutation
    :rtype: torch.tensor with shape (Batch, Len, Embedding)
    """
    mask = x.new_zeros(seq_len, seq_len).fill_diagonal_(1)
    mask = [mask[:, i : i + 1] for i in range(seq_len)]
    len_ = randint(int(seq_len * 0.05), int(seq_len * 0.1))
    idx_ = np.random.choice(list(range(seq_len)), size=len_)
    idx__ = np.random.permutation(len_)
    idx = np.arange(seq_len)
    idx[idx_] = idx[idx_[idx__]]
    mask = [mask[i] for i in idx]
    mask = torch.cat(mask, axis=1)
    return mask.matmul(x)


def repeat(x, seq_len):
    """
    Adds multiplicative noise at time axis.

    :param x: input tensor
    :type x: torch.tensor with shape (Batch, Len, Embedding)
    :param seq_len: Len in x.shape, defaults to 0
    :type seq_len: int, optional

    :return: x with multiplicative noise at time axis.
    :rtype: torch.tensor with shape (Batch, Len, Embedding)
    """
    len_ = randint(int(seq_len * 0.05), int(seq_len * 0.1))
    t = (
        torch.randn_like(
            x.new_zeros(
                len_,
            )
        )
        * 0.1
        + 1
    )
    t = t.repeat(ceil(seq_len / len_))[:seq_len]
    return (x.transpose(1, 2) * t).transpose(1, 2)
