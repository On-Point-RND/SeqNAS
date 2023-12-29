import torch
import torchmetrics
from typing import Optional
from torchmetrics.utilities import rank_zero_warn

from . import register_metric, register_loss
from . import METRIC_REGISTRY, LOSS_REGISTRY


def init_loss(criterion, loss_params=None):
    """
    Function for loss initialization

    :param criterion: loss function name
    :param loss_params: parameters for this loss
    """
    assert criterion in LOSS_REGISTRY.keys(), f"Loss isn't supported: {criterion}"
    if loss_params is None:
        loss_params = {}
    return LOSS_REGISTRY[criterion](**loss_params)


def init_metric(metric_name, device="cpu", metric_params=None):
    """
    Function for metric initialization

    :param metric_name: name of metric
    :param device: device of metric
    :param metric_params: metric parameters
    """
    assert (
        metric_name in METRIC_REGISTRY.keys()
    ), f"Metric isn't supported: {metric_name}"
    if metric_params is None:
        metric_params = {}
    return METRIC_REGISTRY[metric_name]["metric_class"](**metric_params).to(device)


def is_clf_metric(metric_name):
    """
    Function for checking whether metric is for classification
    """
    if METRIC_REGISTRY[metric_name]["metric_type"] == "classification":
        return True
    return False


@register_metric("LossMetric", "custom")
class LossMetric(torchmetrics.Metric):
    """
    Metric for accumulating losses
    """

    higher_is_better = False
    full_state_update = True
    is_differentiable = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "loss_accum",
            default=torch.tensor(0, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum"
        )

    def update(self, loss: torch.Tensor):
        self.loss_accum += loss
        self.total += 1

    def compute(self):
        return self.loss_accum / self.total


@register_metric("AmexMetric", "classification")
class AmexMetric(torchmetrics.Metric):
    """
    Metric for AMEX competition https://www.kaggle.com/code/what5up/amex-lstm-pytorch-lightning-training
    """

    is_differentiable: Optional[bool] = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self, num_classes):
        super().__init__()

        self.add_state("all_true", default=[], dist_reduce_fx="cat")
        self.add_state("all_pred", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `Amex` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_true = y_true.double()
        y_pred = y_pred[:, 1].double()

        self.all_true.append(y_true)
        self.all_pred.append(y_pred)

    def compute(self):
        if type(self.all_true) is list:
            y_true = torch.cat(self.all_true)
        else:
            y_true = self.all_true

        if type(self.all_pred) is list:
            y_pred = torch.cat(self.all_pred)
        else:
            y_pred = self.all_pred

        # count of positives and negatives
        n_pos = y_true.sum()
        n_neg = y_pred.shape[0] - n_pos

        # sorting by descring prediction values
        indices = torch.argsort(y_pred, dim=0, descending=True)
        preds, target = y_pred[indices], y_true[indices]

        # filter the top 4% by cumulative row weights
        weight = 20.0 - target * 19.0
        cum_norm_weight = (weight / weight.sum()).cumsum(dim=0)
        four_pct_filter = cum_norm_weight <= 0.04

        # default rate captured at 4%
        d = target[four_pct_filter].sum() / n_pos

        # weighted gini coefficient
        lorentz = (target / n_pos).cumsum(dim=0)
        gini = ((lorentz - cum_norm_weight) * weight).sum()

        # max weighted gini coefficient
        gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

        # normalized weighted gini coefficient
        g = gini / gini_max

        return 0.5 * (g + d)
