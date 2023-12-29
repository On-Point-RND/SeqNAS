import torch
from torch import nn
from .augmentations import (
    ts_mul_noise,
    add_noise,
    zero_noise,
    shift,
    permute,
    repeat,
)


class SkipConnection(nn.Module):
    """
    Layer to skip connection.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class TimeConv(nn.Module):
    """
    Conv1d for tensors with shape (Timesteps, Batchsize, Hiddensize). All taking parameters
    pass into Conv1d.
    """

    def __init__(self, input_size, output_size, kernel, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(
            input_size, output_size, kernel_size=kernel, padding=padding, stride=stride
        )

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1)
        return x


class LockedDropout(nn.Module):
    """
    LockedDropout applies the same dropout mask to every time step.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class Augmentation:
    """
    Layer to random augmentation of batch.

    :param seq_len: Len dimension of input tensor, defaults to 0
    :type seq_len: int, optional
    :param p: probability of augmentation, defaults to 0.5
    :type p: float, optional
    :param augm_types: types of possible augmentations, defaults to { "ts_mul_noise": ts_mul_noise, "add_noise": add_noise, "azero_noise": zero_noise, "shift": shift, "permute": permute, "repeat": repeat, }
    :type augm_types: dict, optional
    """

    def __init__(
        self,
        seq_len=0,
        p=0.5,
        augm_types={
            "ts_mul_noise": ts_mul_noise,
            "add_noise": add_noise,
            "azero_noise": zero_noise,
            "shift": shift,
            "permute": permute,
            "repeat": repeat,
        },
    ):
        super().__init__()
        self.p = p
        self.augm_types = augm_types
        self.seq_len = seq_len

    def set_aug(self, aug):
        """
        Set augm_types

        Args:
            aug (dict): augmentation types in form {name: function}
        """
        self.augm_types = aug

    def __call__(self, x, training=True):
        if not training:
            return x

        mask = x.data.new(x.size(0)).bernoulli_(1 - self.p).unsqueeze(-1).unsqueeze(-1)
        aug_masks = []
        n = len(self.augm_types)
        for i in range(n - 1):
            temp_mask = mask.data.new(mask.shape).bernoulli_(1 / n)
            aug_masks.append(temp_mask * mask)
        if n > 1:
            aug_masks.append(
                mask
                - (torch.cat(aug_masks, axis=-1).sum(axis=-1, keepdim=True) > 0).type(
                    torch.float
                )
            )
            metamask = 1 / (
                torch.cat(aug_masks, axis=-1).sum(axis=-1, keepdim=True) + 1e-18
            )
        else:
            aug_masks = [mask]
            metamask = mask
        for i in range(len(aug_masks)):
            aug_masks[i] = aug_masks[i] * metamask
        augmented = [x.clone() for i in range(n)]
        res = 0
        for i, tp in enumerate(self.augm_types):
            res += self.augm_types[tp](augmented[i], self.seq_len) * aug_masks[i]
        return x * (1 - mask) + res
