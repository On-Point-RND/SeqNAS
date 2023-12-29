import torch
from time import time
import numpy as np
from ...search_spaces.multi_trail_model import CountReLU
import torch.nn.functional as F
import dict_hash


def random_arch(model):
    """
    Set random sampled architecture to model
    """
    # self.model.reinit()
    model.sample_random()
    current_arch = model.get_arch()
    arch_hash = dict_hash.sha256(current_arch)
    return arch_hash


def reset_weights(model):
    @torch.no_grad()
    def apply(m):
        for name, child in m.named_children():
            if hasattr(child, "_parameters"):
                for param_name in child._parameters:
                    # print(name, param_name)
                    if len(child._parameters[param_name].shape) < 2:
                        torch.nn.init.normal_(child._parameters[param_name].data)
                    else:
                        torch.nn.init.xavier_uniform_(
                            child._parameters[param_name].data
                        )
            reset_parameters = getattr(child, "reset_parameters", None)
            if callable(reset_parameters):
                child.reset_parameters()
            else:
                apply(child)

    apply(model)


def _data_to_device(data, device, req_grad=True):
    # recursivly set to device
    def to_device(obj):
        for key in obj:
            if isinstance(obj[key], dict):
                to_device(obj[key])
            else:
                obj[key] = obj[key].to(device)
                if obj[key].dtype == torch.float32:
                    obj[key].requires_grad_(req_grad)

    to_device(data)

    return data


# TODO for any inputs
def binary_input_end_only(model):
    """
    Evaluate one batch at model and compute binary matrix with shape (Batchsize, Number of ReLU in last layer of model),
    which contain 1 at position i, j if j ReLU at i element of batch was activated 0 else

    Return this matrix (only for last layer of model)
    """
    inp = []

    def relu_collector(m, inp):
        for name, child in m.named_children():
            if isinstance(child, CountReLU):
                if child.stats is not None:
                    if child.stats is not None:
                        inp.append(child.stats)
            else:
                relu_collector(child, inp)
        return inp

    inp = relu_collector(model, inp)
    ind_batch = set(inp[0].shape)
    for i in range(len(inp)):
        ind_batch = ind_batch & set(inp[i].shape)
    ind_batch = list(ind_batch)[0]
    for i in range(len(inp)):
        target_ind = inp[i].shape.index(ind_batch)
        inp[i] = inp[i].transpose(0, target_ind).reshape(inp[i].shape[target_ind], -1)
    t = inp[-1]
    return t


def binary_input(model):
    """
    Evaluate one batch at model and compute binary matrix with shape (Batchsize, Number of ReLU),
    which contain 1 at position i, j if j ReLU at i element of batch was activated 0 else

    Return this matrix
    """
    inp = []

    def relu_collector(m, inp):
        for name, child in m.named_children():
            if isinstance(child, CountReLU):
                if child.stats is not None:
                    inp.append(child.stats)
            else:
                relu_collector(child, inp)
        return inp

    inp = relu_collector(model, inp)
    ind_batch = set(inp[0].shape)
    for i in range(len(inp)):
        ind_batch = ind_batch & set(inp[i].shape)
    ind_batch = list(ind_batch)[0]
    for i in range(len(inp)):
        target_ind = inp[i].shape.index(ind_batch)
        inp[i] = inp[i].transpose(0, target_ind).reshape(inp[i].shape[target_ind], -1)
    t = torch.cat(inp, axis=1)
    return t
