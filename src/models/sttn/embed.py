import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# value / position / timestamp


class SpatialEmbedding(nn.Module):
    def __init__(self, N, d_model, dropout=0.1):
        super(SpatialEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb = nn.Linear(N, d_model)

    def forward(self, adj):
        return self.dropout(self.emb(adj.float()))


def test_spatial_embedding():
    B, N, T, C = 2, 10, 12, 3
    A = torch.randint(0, 2, (N, N))
    print(N)
    model = SpatialEmbedding(N, 8)
    out = model(A)
    print(out.shape)


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=288, dropout=0.1):
        super(TemporalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor):
        # x = torch.arange(0, T).to(self.device)
        return self.dropout(self.emb(x))


def test_time_embedding():
    B, N, T, C = 2, 10, 12, 3
    model = TemporalEmbedding(8)
    out = model(T)
    print(out.shape)


if __name__ == '__main__':
    # test_spatial_embedding()
    test_time_embedding()
