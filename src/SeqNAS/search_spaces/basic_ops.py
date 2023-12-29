import torch.nn as nn

"""
Most of the forward fucntions will be rewritten after
module substituion and remain here just as an example.

"""


class SkipConnection(nn.Module):
    """
    Skip Connection layer.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class LayerChoice(nn.Module):
    """
    Will be replaced by DiffLayer... or SinglePathRandom

    :param ops: pool of layers from which will be randomly choose n_final_ops options
    :type ops: list of layers
    :param n_final_ops: number of options which will be choose
    :type n_final_ops: int
    """

    def __init__(self, ops, n_final_ops=1):
        super().__init__()
        self.ops = ops
        self.n_final_ops = n_final_ops

    # FX requires forward pass
    def forward(self, *x):
        return x


class LayerSkip(nn.Module):
    """
    Skips layer. Will be replaced by SkipSampler
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(*x):
        return x


# Can't nest LayerSkip
class LayerSkip2(nn.Module):
    """
    Experimental LayerSkip
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(*x):
        return x


class RepeatParallel(nn.Module):
    """
    Will be replaced by RepeatRandom
    """

    def __init__(self, ops, repeat):
        """
        :param ops: layer which will be repeated random number of times
        :type ops: torch.nn.Module
        :param repeat: possible numbers of repeat layer
        :type repeat: int or list of ints
        """
        super().__init__()
        self.ops = ops
        if isinstance(repeat, int):
            self.repeat = list(range(repeat))
        else:
            self.repeat = repeat

    # FX requires forward pass
    def forward(self, x):
        return [x]


class Repeat(nn.Module):
    """
    Will be replaced by RepeatRandom
    """

    def __init__(self, ops, repeat):
        """
        :param ops: layer which will be repeated random number of times
        :type ops: torch.nn.Module
        :param repeat: possible numbers of repeat layer
        :type repeat: int or list of ints
        """
        super().__init__()
        self.ops = ops
        if isinstance(repeat, int):
            self.repeat = list(range(repeat))
        else:
            self.repeat = repeat

    # FX requires forward pass
    def forward(self, x):
        return x


class Cell(nn.Module):
    """
    Will be replaced by RandomRNN
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        return x


class Residual(nn.Module):
    """
    Determine if a function should be residual or not.
    """

    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        return self.op(x) + x


class Identity(nn.Module):
    """
    Weighted connection betwen two parameters.
    Can be binary 0/1

    Example: can be used as a Skip connection:

    x + Identity(Conv2d(x))


    """

    def __init__(self):
        super().__init__()
        self.gate = 0

    def forward(self, x):
        return self.gate * x


class Searchabale(nn.Module):
    """
    A general block to determine searchable part of a model.

    """

    def __init__(self, *args):
        super().__init__()
        self.ops = nn.ModuleList(args)
        self._searchable = True

    def forward(self, x):
        for op in self.ops:
            x = op(x)
        return x
