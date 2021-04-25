import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embed import SpatialEmbedding, TemporalEmbedding
from .gcn import GraphConvNet


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        '''
        2D: [B, L, d]
        3D: [B, L', L_att, d]
        Spatial:  [B, H, T, N, d]
        Temporal: [B, H, N, T, d]
        '''
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q/K/V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        output: [B, H, L, L_att]
        '''
        d_k = Q.shape[-1]

        #[B, H, L, L_att, L_att]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        #[B, H, L, L_att]
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context


class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size
                ), "Embedding size needs to be divisible by heads"

        self.W_V = nn.Linear(self.embed_size,
                             self.head_dim * self.heads,
                             bias=False)
        self.W_K = nn.Linear(self.embed_size,
                             self.head_dim * self.heads,
                             bias=False)
        self.W_Q = nn.Linear(self.embed_size,
                             self.head_dim * self.heads,
                             bias=False)
        self.scaled_dot_prodouct_att = ScaledDotProductAttention()
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        [input]_Q/K/V: [batch_size, N, T, C] = [batch_size, N, T, self.embed_size]
        output:  [batch_size, N, T, C]
        '''
        B, N, T, C = input_Q.shape

        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        # Q/k/V: [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads,
                                   self.head_dim).transpose(1, 3)
        K = self.W_K(input_K).view(B, N, T, self.heads,
                                   self.head_dim).transpose(1, 3)
        V = self.W_V(input_V).view(B, N, T, self.heads,
                                   self.head_dim).transpose(1, 3)

        # context: [batch_size, n_heads, len_q, d_v]
        context = self.scaled_dot_prodouct_att(Q, K, V)  # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4)  #[B, N, T, h, d_k]
        context = context.reshape(B, N, T,
                                  self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)
        return output


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size
                ), "Embedding size needs to be divisible by heads"

        self.W_V = nn.Linear(self.embed_size,
                             self.head_dim * self.heads,
                             bias=False)
        self.W_K = nn.Linear(self.embed_size,
                             self.head_dim * self.heads,
                             bias=False)
        self.W_Q = nn.Linear(self.embed_size,
                             self.head_dim * self.heads,
                             bias=False)
        self.scaled_dot_prodouct_att = ScaledDotProductAttention()
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        [input]_Q/K/V: [batch_size, N, T, C] = [batch_size, N, T, self.embed_size]
        output:  [batch_size, N, T, C]
        '''
        B, N, T, C = input_Q.shape

        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        # Q/k/V: [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads,
                                   self.head_dim).permute(0, 3, 1, 2, 4)
        K = self.W_K(input_K).view(B, N, T, self.heads,
                                   self.head_dim).permute(0, 3, 1, 2, 4)
        V = self.W_V(input_V).view(B, N, T, self.heads,
                                   self.head_dim).permute(0, 3, 1, 2, 4)

        # context: [batch_size, n_heads, len_q, d_v]
        context = self.scaled_dot_prodouct_att(Q, K, V)  # [B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4)  #[B, N, T, h, d_k]
        context = context.reshape(B, N, T,
                                  self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)
        return output


class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, cheb_K, dropout,
                 forward_expansion):
        super(STransformer, self).__init__()

        self.adj = adj
        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        # 调用GCN (fixed GCN)
        # TODO tune ~order~ hyper parameter
        self.gcn = GraphConvNet(embed_size, embed_size * 2, embed_size,
                                dropout)
        # self.norm_adj = nn.InstanceNorm2d(1)  # 对邻接矩阵归一化

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, x, D_S, D_T):
        query = x
        B, N, T, C = query.shape
        query = query + D_S + D_T

        # two branch
        # GCN 部分
        # X_G = torch.Tensor(B, N, 0, C).to(self.device)  # empty tensor
        # self.adj = self.adj.unsqueeze(0).unsqueeze(0)  #[1, 1, N, N]
        # self.adj = self.norm_adj(self.adj.float())
        # self.adj = self.adj.squeeze(0).squeeze(0)

        # TODO: parallel
        # for t in range(query.shape[2]):
        #     o = self.gcn(query[:, :, t, :], [self.adj])  # [B, N, C]
        #     o = o.unsqueeze(2)  # shape [N, 1, C] [B, N, 1, C]
        #     #             print(o.shape)
        #     X_G = torch.cat((X_G, o), dim=2)  # cat on T dim
        # why not norm adj? it has norm?
        X_G = self.gcn(query, [self.adj.to(x.device)])  #[B C N T]
        X_G = X_G.permute(0, 2, 3, 1)

        # 最后X_G [B, N, T, C]

        # Spatial Transformer 部分
        attention = self.attention(query, query, query)  #(B, N, T, C)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        # TODO skip connection before norm
        U_S = self.dropout(self.norm2(forward + x))

        # 融合 STransformer and GCN
        g = torch.sigmoid(self.fs(U_S) + self.fg(X_G))  # (7)
        out = g * U_S + (1 - g) * X_G  # (8)

        return out  #(B, N, T, C)


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TTransformer, self).__init__()

        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, D_T):
        query = inp
        B, N, T, C = query.shape

        # temporal embedding加到query。 原论文采用concatenated
        # query is the sum of (X_S + Y_S), spatial skip connection
        query = query + D_T

        attention = self.attention(query, query, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Crosstransformer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(Crosstransformer, self).__init__()

        # left attention
        self.attention_left = SMultiHeadAttention(embed_size, heads)
        self.norm1_left = nn.LayerNorm(embed_size)
        self.norm2_left = nn.LayerNorm(embed_size)

        self.feed_forward_left = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout_left = nn.Dropout(dropout)

        # right attention
        self.attention_right = SMultiHeadAttention(embed_size, heads)
        self.norm1_right = nn.LayerNorm(embed_size)
        self.norm2_right = nn.LayerNorm(embed_size)

        self.feed_forward_right = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout_right = nn.Dropout(dropout)

        self.fc_left = nn.Linear(embed_size, embed_size)
        self.fc_right = nn.Linear(embed_size, embed_size)

    def forward(self, inp, D_S, D_T):
        query = inp
        B, N, T, C = query.shape

        # temporal embedding加到query。 原论文采用concatenated
        # query is the sum of (X_S + Y_S), spatial skip connection
        query = query + D_S + D_T

        # offset
        query_offset_left = query[:, :, 1:, :]
        query_offset_right = query[:, :, :-1, :]

        # pad 0
        query_offset_left = F.pad(query_offset_left, (0, 0, 1, 0))
        query_offset_right = F.pad(query_offset_right, (0, 0, 0, 1))

        # left attention
        attention_left = self.attention_left(query, query_offset_left,
                                             query_offset_left)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout_left(self.norm1_left(attention_left + query))
        forward = self.feed_forward_left(x)
        out_left = self.dropout_left(self.norm2_left(forward + x))

        # right attention
        attention_right = self.attention_right(query, query_offset_right,
                                               query_offset_right)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout_right(self.norm1_right(attention_right + query))
        forward = self.feed_forward_right(x)
        out_right = self.dropout_right(self.norm2_right(forward + x))

        # gate fusion
        # 融合 STransformer and GCN
        g = torch.sigmoid(self.fc_left(out_left) + self.fc_right(out_right))
        out = g * out_left + (1 - g) * out_right
        return out


class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, cheb_K, dropout,
                 forward_expansion):
        super(STTransformerBlock, self).__init__()

        self.adj = adj

        self.shared_spatial_embedding = SpatialEmbedding(
            self.adj.shape[0], embed_size)

        self.shared_temporal_embedding = TemporalEmbedding(
            embed_size, time_num)

        self.STransformer = STransformer(
            embed_size,
            heads,
            self.adj,
            cheb_K,
            dropout,
            forward_expansion,
        )
        self.TTransformer = TTransformer(
            embed_size,
            heads,
            dropout,
            forward_expansion,
        )

        self.Crosstransformer = Crosstransformer(
            embed_size,
            heads,
            dropout,
            forward_expansion,
        )

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # value,  key, query: [N, T, C] [B, N, T, C]
        # Add skip connection,run through normalization and finally dropout

        # spatial and temporal embedding
        B, N, T, C = x.shape

        D_S = self.shared_spatial_embedding(self.adj.to(x.device))  # [N, C]
        D_S = D_S.expand(B, T, N, C)  #[B, T, N, C]相当于在第2维复制了T份, 第一维复制B份
        D_S = D_S.permute(0, 2, 1, 3)  #[B, N, T, C]

        # D_T
        D_T = self.shared_temporal_embedding(torch.arange(0, T).to(x.device))

        x1 = self.norm1(self.STransformer(x, D_S, D_T) + x)  #(B, N, T, C)
        x2 = self.norm2(self.TTransformer(x1, D_T) + x1)
        x3 = self.dropout(self.norm3(self.Crosstransformer(x2, D_S, D_T) + x2))
        return x3


class Encoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        adj,
        time_num,
        forward_expansion,
        cheb_K,
        dropout,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList([
            STTransformerBlock(
                embed_size,
                heads,
                adj,
                time_num,
                cheb_K,
                dropout=dropout,
                forward_expansion=forward_expansion,
            ) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [N, T, C]  [B, N, T, C]
        out = self.dropout(x)
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        adj,
        embed_size,
        num_layers,
        heads,
        time_num,
        forward_expansion,  ##？
        cheb_K,
        dropout,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(embed_size, num_layers, heads, adj, time_num,
                               forward_expansion, cheb_K, dropout)

    def forward(self, src):
        ## scr: [N, T, C]   [B, N, T, C]
        enc_src = self.encoder(src)
        return enc_src  # [B, N, T, C]


class STTransformer(nn.Module):
    def __init__(
        self,
        adj,
        in_channels,
        embed_size,
        time_num,
        num_layers,
        T_dim,
        output_T_dim,
        heads,
        cheb_K,
        forward_expansion,
        dropout=0.1,
    ):
        super(STTransformer, self).__init__()

        self.forward_expansion = forward_expansion
        # 第一次卷积扩充通道数 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.transformer = Transformer(
            adj,
            embed_size,
            num_layers,
            heads,
            time_num,
            forward_expansion,
            cheb_K,
            dropout=dropout,
        )

        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input x shape[B, C, N, T]
        # C:通道数量。  N:传感器数量。  T:时间数量
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        #x shape [B, N, T, C]
        output_Transformer = self.transformer(x)  # [B, N, T, C]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)
        #output_Transformer shape[B, T, N, C]

        #         output_Transformer = output_Transformer.unsqueeze(0)
        out = self.relu(self.conv2(
            output_Transformer))  # 等号左边 out shape: [1, output_T_dim, N, C]
        out = out.permute(0, 3, 2,
                          1)  # 等号左边 out shape: [B, C, N, output_T_dim]
        out = self.conv3(out)  # 等号左边 out shape: [B, 1, N, output_T_dim]

        return out  #[B, 1, N, output_T_dim]


if __name__ == '__main__':
    #TODO the temporal embedding (0, T) has bug!
    N = 25
    A = torch.randint(0, 2, (N, N))  # 邻接矩阵
    in_channels = 1  # 输入通道数。只有速度信息，所以通道为1
    embed_size = 64  # Transformer通道数
    time_num = 288  # 1天时间间隔数量
    num_layers = 1  # Spatial-temporal block 堆叠层数
    T_dim = 12  # 输入时间维度。 输入前1小时数据，所以 60min/5min = 12
    output_T_dim = 3  # 输出时间维度。预测未来15,30,45min速度
    cheb_K = 2  # Order for Chebyshev Polynomials (Eq 2)
    forward_expansion = 4  # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
    heads = 1  # transformer head 数量。 时、空transformer头数量相同

    model = STTransformer(
        A.float(),
        in_channels,
        embed_size,
        time_num,
        num_layers,
        T_dim,
        output_T_dim,
        heads,
        cheb_K,
        forward_expansion,
    )
    x = torch.randn(3, in_channels, N, T_dim)
    out = model(x)
    print(out.shape)
