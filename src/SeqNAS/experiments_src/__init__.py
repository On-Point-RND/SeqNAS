import torchmetrics
import importlib
import torch


METRIC_REGISTRY = {
    "f1_macro": {"metric_class": torchmetrics.F1Score, "metric_type": "classification"},
    "f1_weighted": {
        "metric_class": torchmetrics.F1Score,
        "metric_type": "classification",
    },
    "accuracy": {
        "metric_class": torchmetrics.Accuracy,
        "metric_type": "classification",
    },
    "auc": {"metric_class": torchmetrics.AUROC, "metric_type": "classification"},
    "r2": {"metric_class": torchmetrics.R2Score, "metric_type": "regression"},
}
LOSS_REGISTRY = {
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "WeightCrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "MSELoss": torch.nn.MSELoss,
    "MAELoss": torch.nn.L1Loss,
}


def register_metric(name, metric_type):
    """Decorator to register a new metric (e.g., Accuracy), metric_type - [classification, regression, custom]."""

    def register_metric_cls(cls):
        if name in METRIC_REGISTRY:
            raise ValueError("Cannot register duplicate metric ({})".format(name))
        METRIC_REGISTRY[name] = {"metric_class": cls, "metric_type": metric_type}
        return cls

    return register_metric_cls


def register_loss(name):
    """Decorator to register a new loss (e.g., CrossEntropyLoss)."""

    def register_loss_cls(cls):
        if name in LOSS_REGISTRY:
            raise ValueError("Cannot register duplicate loss ({})".format(name))
        LOSS_REGISTRY[name] = cls
        return cls

    return register_loss_cls


importlib.import_module("SeqNAS.experiments_src.metrics")
