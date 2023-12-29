import os
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def data_to_device(data, device):
    # recursivly set to device
    def to_device(obj):
        for key in obj:
            if isinstance(obj[key], dict):
                to_device(obj[key])
            else:
                obj[key] = obj[key].to(device)

    to_device(data)
    return data


def iterate_one_epoch(model, val_loader, device):
    preds = []
    with torch.no_grad():
        for batch in val_loader:
            batch = data_to_device(batch, device)
            outputs = model(batch["model_input"])
            entropy = outputs["entropy"].detach().cpu().tolist()
            preds += entropy
    return preds


def eval_ensemble(model, val_loader, device, augmentations=(), exp_path=""):
    entropy = []
    targets = []
    model.set_uq_mode(True)
    model.to(device)
    for aug, label in augmentations:
        model.augmentations = aug
        preds = iterate_one_epoch(model, val_loader, device)
        entropy += preds
        targets += [label] * len(preds)

    mean_entropy = sum(entropy) / len(entropy)
    fpr, tpr, _ = roc_curve(targets, entropy)
    plt.plot(
        fpr,
        tpr,
        color="purple",
        lw=2,
        label=f"ROC curve (area = {auc(fpr, tpr):.4f})\nMEAN Entropy: {mean_entropy:.4f}",
    )
    plt.grid()
    plt.legend()
    path = os.path.join(exp_path, "auc.png")
    plt.savefig(path)
