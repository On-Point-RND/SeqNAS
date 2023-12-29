from torch import optim as opt
from apex.optimizers import FusedAdam, FusedSGD, FusedNovoGrad, FusedLAMB

OPT = {
    "SGD": lambda params_dict, lr=1e-3, momentum=0.9, weight_decay=0, betas=(
        0.9,
        0.999,
    ), eps=1e-08: opt.SGD(
        params_dict, lr=lr, momentum=momentum, weight_decay=weight_decay
    ),
    "ADAM": lambda params_dict, lr=1e-3, momentum=0.9, weight_decay=0, betas=(
        0.9,
        0.999,
    ), eps=1e-08: opt.Adam(
        params_dict, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    ),
    "FUSEDADAM": lambda params_dict, lr=1e-3, momentum=0.9, weight_decay=0, betas=(
        0.9,
        0.999,
    ), eps=1e-08: FusedAdam(
        params_dict, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    ),
    "FUSEDNOVOGRAD": lambda params_dict, lr=1e-3, momentum=0.9, weight_decay=0, betas=(
        0.9,
        0.999,
    ), eps=1e-08: FusedNovoGrad(
        params_dict, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    ),
    "FUSEDSGD": lambda params_dict, lr=1e-3, momentum=0.9, nesterov=False, weight_decay=0: FusedSGD(
        params_dict,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
    ),
    "FUSEDLAMB": lambda params_dict, lr=1e-3, momentum=0.9, weight_decay=0, betas=(
        0.9,
        0.999,
    ), eps=1e-08: FusedLAMB(
        params_dict, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    ),
}
