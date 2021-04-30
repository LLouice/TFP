import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    if name == "swish":
        return Swish()
    return nn.ReLU()


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x
