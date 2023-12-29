import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.simplerick_rnn.searchable_rnn import SearchableRNN


class FunctionVector(nn.Module):
    """
    Container for layers to any search
    """

    def __init__(self, options):
        """
        Args:
            options (list of layers): layers to search
        """
        super().__init__()
        self.options = options
        self.alpha = [0] * len(self.options)
        self.size = len(self.alpha)
        self.samples = []

    def set_weights(self, weights):
        """
        Setting weights for every layer
        """
        self.weights = weights

    def get_functions(self):
        return self.options

    def remove_useless(self):
        """
        Call in set_final_and_clean to remove layers which wasn`t chosen
        """
        for i in range(len(self.weights)):
            if self.weights[i] == 0:
                self.options[i] = None


class BaseSampler(nn.Module):
    """
    Parent class for most searchable modules.
    """

    def __init__(self):
        super(BaseSampler, self).__init__()

    def sample_path(self):
        """
        Call to make a random sample
        """
        self._sample_random_path()
        self._set_ops()

    def _set_ops(self):
        self.ops = self.possible_options.get_functions()

    def _get_final_and_clean(self):
        """
        A general function to remove all unused functions
        """
        raise NotImplemented

    def _load_weights(self, weights):
        """
        Load weights for the layer.
        """
        self.possible_options.set_weights(weights)
        self._set_ops()

    def _get_weights(self):
        """
        Get layer weights
        """
        return self.possible_options.weights

    def forward(self, *input_x):
        first = True
        for op, a in zip(self.ops, self.possible_options.weights):
            if a == 0:
                continue
            out = op(*input_x)
            if first:
                if isinstance(out, tuple):
                    outs = tuple(a * it for it in out)
                else:
                    outs = a * out
                first = False
            else:
                if isinstance(out, tuple):
                    outs = tuple(acc + a * it for acc, it in zip(outs, out))
                else:
                    outs += a * out
        return outs


class SkipSampler(nn.Module):
    """
    Skip layer with 0.5 probability
    """

    def __init__(self, arg):
        super().__init__()
        self.layer = arg.layer
        self.layer_weight = 0

    def sample_path(self):
        """
        Call to make a random sample
        """
        self.layer_weight = 0 if torch.rand(()) < 0.5 else 1

    def _get_weights(self):
        """
        Get layer weights
        """
        return self.layer_weight

    def _load_weights(self, weights):
        """
        Load weights for the layer.
        """
        self.layer_weight = weights

    def _get_final_and_clean(self):
        """
        A general function to remove all unused functions
        """
        if self.layer_weight == 0:
            self.layer = None
        return self

    def forward(self, x):
        if self.layer is None or self.layer_weight == 0:
            if isinstance(x, tuple):
                # when used with FlexibleDecoder, we need to return the
                # encoder output instead of decoder input
                return x[-1]
            return x

        return self.layer(x)


class SkipSampler2(nn.Module):
    """
    Experimental Skip Sampler
    """

    def __init__(self, arg):
        super().__init__()
        self.layer = arg.layer
        self.layer_weight = 0

    def sample_path(self):
        """
        Call to make a random sample
        """
        self.layer_weight = 0 if torch.rand(()) < 0.5 else 1

    def _get_weights(self):
        """
        Get layer weights
        """
        return self.layer_weight

    def _load_weights(self, weights):
        """
        Load weights for the layer.
        """
        self.layer_weight = weights

    def _get_final_and_clean(self):
        """
        A general function to remove all unused functions
        """
        if self.layer_weight == 0:
            self.layer = None
        return self

    def forward(self, x):
        if self.layer is None or self.layer_weight == 0:
            return x

        return self.layer(x)


class CountReLU(BaseSampler):
    """
    Keep last ReLU mask after forward, need to compute proxy for train-free algorythm.

    :param argument: instance of fake class
    :type argument: equal to type of fake class
    """

    def __init__(self, argument):
        super(CountReLU, self).__init__()
        self.stats = None
        self.possible_options = FunctionVector([1])
        self.possible_options.set_weights([1])
        self.relu = nn.ReLU()

    def forward(self, x):
        self.stats = (x > 0).float()
        return self.relu(x)

    def sample_path(self):
        self.stats = None

    def _get_final_and_clean(self):
        return self


class CountPReLU(CountReLU):
    """
    Modification of CountReLU with PReLU instead ReLU.

    :param argument: instance of fake class
    :type argument: equal to type of fake class
    """

    def __init__(self, argument):
        super(CountPReLU, self).__init__(argument)
        self.relu = nn.PReLU()


class SumOutputs(nn.Module):
    """
    Service class to sum several outputs. Use in diff searcher after set_final_and_clean.

    :param functions: layers to sum together
    :type functions: list of nn.Modules
    """

    def __init__(self, functions):
        super().__init__()
        self.ops = nn.ModuleList(functions)

    def forward(self, *input_x):
        first = True
        for op in self.ops:
            out = op(*input_x)
            if first:
                outs = out
                first = False
            else:
                outs += out
        return outs


class DiffLayerSoftMax(BaseSampler):
    """
    Use in diff searcher to choose best layers, filter weights with Softmax.

    :param argument: instance of fake class
    :type argument: equal to type of fake class
    """

    def __init__(self, argument):
        super().__init__()
        self.possible_options = FunctionVector(argument.ops)
        self._set_ops()
        self._init_weights()
        self.n_final_ops = argument.n_final_ops
        self.temp = 1e-2
        self.activation = F.softmax  # F.gumbel_softmax #F.softmax
        self.kwargs = {"dim": -1}
        self.cached_alphas = None

    def _init_weights(self):
        self.possible_options.weights = (
            torch.ones(1, self.possible_options.size) / self.possible_options.size
        )

    def _get_weights(self):
        """
        Get layer weights
        """
        if self.cached_alphas is not None:
            return self.cached_alphas
        else:
            return self.possible_options.weights

    def _load_weights(self, weights):
        """
        Load weights for the layer.
        """
        self.possible_options.set_weights(weights)

    def _set_ops(self):
        self.ops = nn.ModuleList(self.possible_options.get_functions())

    def _get_final_and_clean(self):
        """
        A general function to remove all unused functions
        """
        ordered = torch.argsort(
            self.possible_options.weights.flatten(), dim=-1, descending=True
        )[: self.n_final_ops]
        print(
            "Indices for the final arch:",
            ordered,
            self.possible_options.weights,
        )
        functions = [self.ops[int(i)] for i in ordered]
        # return nn module here
        return SumOutputs(functions)

    def forward(self, *input_x):
        first = True
        alphas = self.activation(
            self.possible_options.weights / self.temp, **self.kwargs
        )
        self.cached_alphas = alphas
        for op, a in zip(self.ops, alphas.flatten()):
            out = op(*input_x)
            # out = F.layer_norm(out, out.shape[1:])
            if first:
                outs = a * out
                first = False
            else:
                outs += a * out
        return outs


class DiffLayerGumbel(DiffLayerSoftMax):
    """
    Use in diff searcher to choose best layers, filter weights with GumbelSoftmax.

    :param argument: instance of fake class
    :type argument: equal to type of fake class
    """

    def __init__(self, argument):
        super().__init__(argument)
        self.possible_options = FunctionVector(argument.ops)
        self._set_ops()
        self._init_weights()
        self.n_final_ops = argument.n_final_ops
        self.temp = 1e-3
        self.activation = F.gumbel_softmax
        self.kwargs = {"dim": -1}
        self.cached_alphas = None


class IdentityLayer(DiffLayerSoftMax):
    def __init__(self, argument):
        """
        Args:
            argument (Fake class): fake class wich will be replaced by this class
        """
        super().__init__(argument)
        self.possible_options = FunctionVector(argument.ops)
        self._set_ops()
        self._init_weights()
        self.n_final_ops = argument.n_final_ops
        self.activation = F.softmax  # F.gumbel_softmax #F.softmax
        self.kwargs = {"dim": -1}
        self.cached_alphas = None

    def _init_weights(self):
        w = torch.ones(1, self.possible_options.size) / self.possible_options.size
        self.possible_options.weights = w


class SinglePathRandom(BaseSampler):
    """
    Use in searchers to sample random architecture from supernet.

    :param argument: instance of fake class
    :type argument: equal to type of fake class
    """

    def __init__(self, argument):
        super().__init__()
        self.possible_options = FunctionVector(argument.ops)
        if hasattr(argument, "n_final_ops"):
            self.n_final_ops = argument.n_final_ops
        else:
            self.n_final_ops = 1
        self.sample_path()

    def _set_ops(self):
        self.ops = nn.ModuleList(self.possible_options.get_functions())

    def _sample_random_path(self):
        """
        Randomly choose one edge in supernet
        """
        if isinstance(self.n_final_ops, list):
            n = random.choice(self.n_final_ops)
        else:
            n = self.n_final_ops

        indecies = random.sample(range(self.possible_options.size), n)
        weights = self.possible_options.alpha[:]
        for idx in indecies:
            weights[idx] = 1
        self.possible_options.set_weights(weights)

    def _get_final_and_clean(self):
        """
        A general function to remove all unused functions
        """
        self.possible_options.remove_useless()
        # It is called single random so it returns only one function
        return self


class SinglePathRandomSimpleFirst(BaseSampler):
    """
    Use in searchers to sample random architecture from supernet.
    Assumes that ops are aranged from simple to complex and tries sampling
    simple architectures first

    :param argument: instance of fake class
    :type argument: equal to type of fake class
    """

    def __init__(self, argument):
        super().__init__()
        self.possible_options = FunctionVector(argument.ops)
        self.level = 0
        self.sample_path()

    def _set_ops(self):
        self.ops = nn.ModuleList(self.possible_options.get_functions())

    def _sample_random_path(self):
        """
        Randomly choose one edge in supernet
        """
        max_idx = min(self.possible_options.size - 1, int(self.level))
        idx = random.randint(0, max_idx)
        weights = self.possible_options.alpha[:]
        weights[idx] = 1
        self.possible_options.set_weights(weights)
        self.level += 0.5

    def _get_final_and_clean(self):
        """
        A general function to remove all unused functions
        """
        self.possible_options.remove_useless()
        # It is called single random so it returns only one function
        return self


class ParallelRepsRandom(BaseSampler):
    """
    Use in searchers to random number repeat one layer.
    Outputs are returned concatenated along the last dimension.

    :param argument: instance of fake class
    :type argument: equal to type of fake class
    """

    def __init__(self, argument):
        super().__init__()
        self.repeat = argument.repeat
        max_reps = max(self.repeat)
        ops = []
        while len(ops) < max_reps:
            ops.append(copy.deepcopy(argument.ops))

        self.possible_options = FunctionVector(ops)
        self.sample_path()

    def _set_ops(self):
        self.ops = nn.ModuleList(self.possible_options.get_functions())

    def _sample_random_path(self):
        """
        Choose random number of repeats
        """
        idx = random.randint(0, len(self.repeat) - 1)
        num_reps = self.repeat[idx]
        weights = self.possible_options.alpha[:]
        weights[:num_reps] = [1 for _ in range(num_reps)]
        self.possible_options.set_weights(weights)
        self.num_reps = num_reps

    def _load_weights(self, weights):
        super()._load_weights(weights)
        self.num_reps = sum(weights)

    def _get_final_and_clean(self):
        """
        A general function to remove all unused functions
        """
        self.possible_options.remove_useless()
        # del self.possible_options
        return self

    def forward(self, *input_x):
        outputs = []
        for op, a in zip(self.ops, self.possible_options.weights):
            if a == 0:
                continue
            out = op(*input_x)
            if isinstance(out, tuple):
                out = out[0]
            outputs.append(out)

        return torch.cat(outputs, dim=-1)


class RepeatRandom(BaseSampler):
    """
    Use in searchers to random number repeat one layer.

    :param argument: instance of fake class
    :type argument: equal to type of fake class
    """

    def __init__(self, argument):
        super().__init__()
        self.repeat = argument.repeat
        max_reps = max(self.repeat)
        ops = []
        while len(ops) < max_reps:
            ops.append(copy.deepcopy(argument.ops))

        self.possible_options = FunctionVector(ops)
        self.sample_path()

    def _set_ops(self):
        self.ops = nn.ModuleList(self.possible_options.get_functions())

    def _sample_random_path(self):
        """
        Choose random number of repeats
        """
        idx = random.randint(0, len(self.repeat) - 1)
        num_reps = self.repeat[idx]
        weights = self.possible_options.alpha[:]
        weights[:num_reps] = [1 for _ in range(num_reps)]
        self.possible_options.set_weights(weights)

    def forward(self, x):
        for op, a in zip(self.ops, self.possible_options.weights):
            if a == 0:
                continue
            x = op(x) * a

        return x

    def _get_final_and_clean(self):
        """
        A general function to remove all unused functions
        """
        self.possible_options.remove_useless()
        # del self.possible_options
        return self


class RandomRNN(SearchableRNN):
    def __init__(self, argument):
        """
        Args:
            argument (Fake class): fake class wich will be replaced by this class
        """
        super().__init__(**argument.kwargs)
