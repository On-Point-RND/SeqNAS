from ..train_free.utils import _data_to_device
import numpy as np
import torch


def Jacobians(model, criterion, loader, device="cpu"):
    model.to(device)
    model.train()
    grads = {i: [] for i in next(iter(loader))["model_input"]}
    sum_ = 0
    for batch in loader:
        model.zero_grad()
        batch = _data_to_device(batch, device, req_grad=True)
        out = model(batch["model_input"])
        loss = criterion(out, batch)
        loss.backward()
        for name in batch["model_input"]:
            grad = batch["model_input"][name].grad
            if grad is not None:
                grads[name].append(grad.abs().mean().item())
    for name in grads:
        grads[name] = np.mean(grads[name])
        if grads[name] is not None:
            sum_ += grads[name]

    for name in grads:
        grads[name] /= sum_
    return grads


def FeaturePermutation(model, metric, loader, device="cpu"):
    model.to(device)
    model.eval()
    batch_ = next(iter(loader))["model_input"]
    perm = {i: [] for i in batch_}
    for feature in batch_:
        for batch in loader:
            batch = _data_to_device(batch, device, req_grad=True)
            batch["model_input"][feature] = batch["model_input"][feature][
                torch.randperm(len(batch["model_input"][feature]))
            ]
            out = model(batch["model_input"])
            score = metric(out, batch)
            perm[feature].append(score)
        perm[feature] = np.mean(perm[feature])
    return perm
