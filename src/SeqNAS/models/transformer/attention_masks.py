import torch
from math import sin, cos


"""
Different types of fixed attentions masks
"""


def diag(len, device):
    h = len / 20
    return torch.tensor(
        [[1.0 if abs(i - j) < h else 0.0 for i in range(len)] for j in range(len)]
    ).to(device)


def cross(len, device):
    h = len / 20
    return torch.tensor(
        [
            [
                1.0 if abs((len - i) - j) < h or abs(i - j) < h else 0.0
                for i in range(len)
            ]
            for j in range(len)
        ]
    ).to(device)


def diag_1(len, device):
    h = len // 10
    return torch.tensor(
        [
            [1.0 if i - j >= 1 and i - j <= h else 0.0 for i in range(len)]
            for j in range(len)
        ]
    ).to(device)


def diag1(len, device):
    h = len // 10
    return torch.tensor(
        [
            [1.0 if i - j <= -1 and i - j >= -h else 0.0 for i in range(len)]
            for j in range(len)
        ]
    ).to(device)


def bottom(len, device):
    k = len // 2
    a = torch.tensor(
        [
            [
                1.0 if i < len / 2 + 0.3 * k and j < len // 10 else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(device) + torch.tensor(
        [
            [
                k * 0.5 / (i - j) if (i - j) >= k and (i - j) <= 1.3 * k else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(
        device
    )
    b = torch.tensor(
        [
            [
                1.0 if i < len / 2 + 0.3 * k and j < len // 10 else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(device) * torch.tensor(
        [
            [
                k * 0.5 / (i - j) if (i - j) >= k and (i - j) <= 1.3 * k else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(
        device
    )
    return a - b


def top(len, device):
    k = len // 2
    a = torch.tensor(
        [
            [
                1.0 if i >= len / 2 - 0.3 * k and j > len - len // 10 else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(device) + torch.tensor(
        [
            [
                k * 0.5 / (j - i) if (j - i) >= k and (j - i) <= 1.3 * k else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(
        device
    )
    b = torch.tensor(
        [
            [
                1.0 if i >= len / 2 - 0.3 * k and j > len - len // 10 else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(device) * torch.tensor(
        [
            [
                k * 0.5 / (j - i) if (j - i) >= k and (j - i) <= 1.3 * k else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(
        device
    )
    return a - b


def bottom_inv(len, device):
    k = len // 2
    a = torch.tensor(
        [
            [
                1.0 if i > len / 2 - 0.3 * k and j < len // 10 else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(device) + torch.tensor(
        [
            [
                k * 0.5 / (len - i - j)
                if (len - i - j) >= k and (len - i - j) <= 1.3 * k
                else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(
        device
    )
    b = torch.tensor(
        [
            [
                1.0 if i > len / 2 - 0.3 * k and j < len // 10 else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(device) * torch.tensor(
        [
            [
                k * 0.5 / (len - i - j)
                if (len - i - j) >= k and (len - i - j) <= 1.3 * k
                else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(
        device
    )
    return a - b


def top_inv(len, device):
    k = len // 2
    a = torch.tensor(
        [
            [
                1.0 if i < len / 2 + 0.3 * k and j > len - len // 10 else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(device) + torch.tensor(
        [
            [
                k * 0.5 / (j - len + i)
                if (j - len + i) >= k and (j - len + i) <= 1.3 * k
                else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(
        device
    )
    b = torch.tensor(
        [
            [
                1.0 if i < len / 2 + 0.3 * k and j > len - len // 10 else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(device) * torch.tensor(
        [
            [
                k * 0.5 / (j - len + i)
                if (j - len + i) >= k and (j - len + i) <= 1.3 * k
                else 0.0
                for j in range(len)
            ]
            for i in range(len)
        ]
    ).to(
        device
    )
    return a - b


def left(len, device):
    h = len / 10
    return torch.tensor([[1 / (j / h + h) for j in range(len)] for i in range(len)]).to(
        device
    )


def right(len, device):
    h = len / 10
    return torch.tensor(
        [[1 / (len - j / h) for j in range(len)] for i in range(len)]
    ).to(device)


def sin_mask(len, device):
    h = len / 7
    return torch.tensor(
        [[sin((len - j) / h) for j in range(len)] for i in range(len)]
    ).to(device)


def cos_mask(len, device):
    h = len / 7
    return torch.tensor(
        [[cos((len - j) / h) for j in range(len)] for i in range(len)]
    ).to(device)


def noise_one(len, device):
    return torch.randn((len, len)).bernoulli_(1 / (len * 20)).to(device)


def noise_two(len, device):
    return torch.randn((len, len)).bernoulli_(1 / (len * 30)).to(device)


def zero(len, device):
    return torch.zeros((len, len)).to(device)
