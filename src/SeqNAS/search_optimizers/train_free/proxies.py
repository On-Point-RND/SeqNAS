import torch
from time import time
import numpy as np
from ...search_spaces.multi_trail_model import CountReLU
import torch.nn.functional as F
from tqdm import tqdm
from .utils import (
    random_arch,
    reset_weights,
    _data_to_device,
    binary_input_end_only,
    binary_input,
)


def calculate_proxies(
    model,
    loader,
    criterion,
    device="cpu",
    n=5,
    proxies=[
        "NReg",
        "NReg_eo",
        "RegCor",
        "RegCor_eo",
        "NTK",
        "Snip",
        "Synflow",
        "Fisher",
    ],
):
    """
    Calculate all proxies from proxies and return dict with form {name: value}

    Args:
        ...
        n (int): number of batches to calculate proxies
        proxies (list): names of proxies to calculate (each of them must be in available_proxies)
    """

    available_proxies = {
        "NReg": Nregions,
        "NReg_eo": Nregions_end_only,
        "RegCor": RegionsCorrelation,
        "RegCor_eo": RegionsCorrelation_end_only,
        "NTK": NTK,
        "Snip": Snip,
        "Synflow": Synflow,
        "Fisher": Fisher,
    }
    if len(loader) < n:
        print(
            "Warning! Length of loader is smaller than n, so n is set to", len(loader)
        )
        n = len(loader)
    scores = {i: [] for i in proxies if i != "NTK"}
    for i, batch in enumerate(loader):
        if i == n:
            break
        model.train()
        model.to(device)
        reset_weights(model)
        batch = _data_to_device(batch, device)
        model.zero_grad()
        out = model(batch["model_input"])
        loss = criterion(out, batch)
        for proxy in proxies:
            ans = available_proxies[proxy](model, out, loss)
            if proxy == "NTK":
                for p in ans:
                    if p in scores:
                        scores[p].append(ans[p])
                    else:
                        scores[p] = [ans[p]]
            else:
                scores[proxy].append(ans)
    for k in scores:
        scores[k] = np.mean(scores[k])
    return scores


"""
Functions to calculate proxies
Args:
    ...
    out (tensor or dict): output of model
    loss (function): loss function to model
"""


def Nregions(model, out, loss):
    Binput = binary_input(model)
    return Binput.unique(dim=0).shape[0]


def Nregions_end_only(model, out, loss):
    Binput = binary_input_end_only(model)
    return Binput.unique(dim=0).shape[0]


def RegionsCorrelation(model, out, loss):
    Binput = binary_input(model).unsqueeze(0)
    dist = torch.cdist(Binput, 1 - Binput, p=0)
    return torch.log(torch.norm(dist) / dist.shape[1] ** 1.5).item()


def RegionsCorrelation_end_only(model, out, loss):
    Binput = binary_input_end_only(model).unsqueeze(0)
    dist = torch.cdist(Binput, 1 - Binput, p=0)
    return torch.log(torch.norm(dist) / dist.shape[1] ** 1.5).item()


def NTK(model, out, loss):
    if isinstance(out, dict):
        out = out["preds"]
    i_grad = []
    for i in range(out.shape[0]):
        j_grad = []
        for j in range(out.shape[1]):
            model.zero_grad()
            out[i][j].backward(retain_graph=True)
            for p in model.parameters():
                gr = p.grad
                if gr is not None:
                    j_grad.append(gr.reshape(-1))
            out[i][j] = 0.0
        i_grad.append(torch.cat(j_grad, axis=0))
    i_grad = torch.tensor([[i.matmul(j).item() for i in i_grad] for j in i_grad])
    eigv = torch.linalg.eigvals(i_grad).abs() + 1e-12
    return {
        "NTK_ratio": (eigv.max().item() / eigv.min().item()),
        # "NTK_min":eigv.min().item(),
        "NTK_max": eigv.max().item(),
        # "NTK_mean":eigv.mean().item(),
    }


def Snip(model, out, loss):
    model.zero_grad()
    loss.backward(retain_graph=True)
    c = []
    for p in model.parameters():
        grad = p.grad
        if grad is not None:
            c.append((p * grad).abs().mean().item())
    return np.mean(c)


def Synflow(model, out, loss):
    model.zero_grad()
    loss.backward(retain_graph=True)
    c = []
    for p in model.parameters():
        grad = p.grad
        if grad is not None:
            c.append((p * grad).mean().item())
    return np.mean(c)


def Fisher(model, out, loss):
    model.zero_grad()
    loss.backward(retain_graph=True)
    c = []
    for p in model.parameters():
        grad = p.grad
        if grad is not None:
            c.append((grad**2).mean().item())
    return np.mean(c)
