import numpy as np


def _data_to_device(data, device):
    # recursivly set to device
    def to_device(obj):
        for key in obj:
            if isinstance(obj[key], dict):
                to_device(obj[key])
            else:
                obj[key] = obj[key].to(device)

    to_device(data)
    return data


def get_alphas(left=0, right=1, steps=100):
    step_size = (right - left) / steps
    alphas = np.arange(left, right, step_size)
    return alphas
