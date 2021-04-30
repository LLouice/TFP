import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    if name == "swish":
        return Swish()
    if name == "mish":
        return Mish()
    return nn.ReLU()


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * torch.tanh(F.softplus(x))
        return x
