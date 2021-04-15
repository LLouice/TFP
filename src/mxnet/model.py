import mxnet as mx
from mxnet import np, npx, autograd, gluon
from mxnet import use_np
from mxnet.gluon import nn, Trainer
from mxnet.gluon.data import Dataset, DataLoader

# npx.set_np()


@use_np
def nconv(x, A):
    return np.einsum('ncvl,vw->ncwl', x, A)


def test_nconv():
    (b, c, n, t) = (12, 8, 5, 7)
    x = np.random.randn(b, c, n, t)
    A = np.random.randn(n, n)

    x.attach_grad()
    A.attach_grad()

    with autograd.record():
        output = nconv(x, A).sum()
        output.backward()
    print(x.grad.shape)
    print(A.grad.shape)
    print(A)


@use_np
class GraphConvNet(nn.HybridBlock):
    def __init__(self, c_out, dropout, support_len=3, order=2):
        super(GraphConvNet, self).__init__()
        self.order = order
        with self.name_scope():
            self.final_conv = nn.Conv2D(c_out, (1, 1),
                                        padding=(0, 0),
                                        strides=(1, 1),
                                        use_bias=True)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = np.concatenate(out, axis=1)
        h = self.final_conv(h)
        h = self.dropout(h)
        return h


def test_gcn():
    (b, c, n, t) = (12, 8, 5, 7)
    x = np.random.randn(b, c, n, t)
    A = np.random.randn(n, n)
    model = GraphConvNet(3, 0.3, 1)
    print(model)
    print(model.collect_params())
    model.initialize()
    output = model(x, [A])
    print(output.shape)


@use_np
class GWNet(nn.HybridBlock):
    def __init__(self,
                 num_nodes,
                 dropout=0.3,
                 supports=None,
                 do_graph_conv=True,
                 addaptadj=True,
                 aptinit=None,
                 out_dim=12,
                 residual_channels=32,
                 dilation_channels=32,
                 cat_feat_gc=False,
                 skip_channels=256,
                 end_channels=512,
                 kernel_size=2,
                 blocks=4,
                 layers=2,
                 apt_size=10):
        super(GWNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.cat_feat_gc = cat_feat_gc
        self.addaptadj = addaptadj

        with self.name_scope():
            if self.cat_feat_gc:
                self.start_conv = nn.Conv2D(  # hard code to avoid errors
                    residual_channels,
                    kernel_size=(1, 1))

                self.cat_feature_conv = nn.Conv2D(residual_channels,
                                                  kernel_size=(1, 1))
            else:
                self.start_conv = nn.Conv2D(residual_channels,
                                            kernel_size=(1, 1))

            self.fixed_supports = supports or []
            receptive_field = 1

            self.supports_len = len(self.fixed_supports)
            if do_graph_conv and addaptadj:
                assert aptinit is None
                self.nodevec1 = self.params.get("nodevec1",
                                                shape=(num_nodes, apt_size))
                self.nodevec2 = self.params.get("nodevec2",
                                                shape=(num_nodes, apt_size))
                self.supports_len += 1

            depth = list(range(blocks * layers))

            # 1x1 convolution for residual and skip connections (slightly different see docstring)
            self.residual_convs = nn.HybridSequential()
            self.skip_convs = nn.HybridSequential()
            self.bn = nn.HybridSequential()
            self.graph_convs = nn.HybridSequential()
            for _ in depth:
                self.residual_convs.add(nn.Conv2D(residual_channels, (1, 1)))
                self.skip_convs.add(nn.Conv2D(skip_channels, (1, 1)))
                self.bn.add(nn.BatchNorm())
                self.graph_convs.add(
                    GraphConvNet(residual_channels,
                                 dropout,
                                 support_len=self.supports_len))

            self.filter_convs = nn.HybridSequential()
            self.gate_convs = nn.HybridSequential()
            for b in range(blocks):
                additional_scope = kernel_size - 1
                D = 1  # dilation
                for i in range(layers):
                    # dilated convolutions
                    self.filter_convs.add(
                        nn.Conv2D(dilation_channels, (1, kernel_size),
                                  dilation=D))
                    self.gate_convs.add(
                        nn.Conv2D(dilation_channels, (1, kernel_size),
                                  dilation=D))
                    D *= 2
                    receptive_field += additional_scope
                    additional_scope *= 2
            self.receptive_field = receptive_field

            self.end_conv_1 = nn.Conv2D(end_channels, (1, 1))
            self.end_conv_2 = nn.Conv2D(out_dim, (1, 1))

    @staticmethod
    def add_model_specific_args(parent_parser):
        _ADJ_CHOICES = [
            'scalap', 'normlap', 'symnadj', 'transition', 'identity'
        ]
        parser = parent_parser.add_argument_group("GWNet")
        parser.add_argument('--data',
                            type=str,
                            default='data/METR-LA',
                            help='data path')
        parser.add_argument('--adjdata',
                            type=str,
                            default='data/sensor_graph/adj_mx.pkl',
                            help='adj data path')
        parser.add_argument('--adjtype',
                            type=str,
                            default='doubletransition',
                            help='adj type',
                            choices=_ADJ_CHOICES)
        parser.add_argument('--do_graph_conv',
                            action='store_true',
                            help='whether to add graph convolution layer')
        parser.add_argument('--aptonly',
                            action='store_true',
                            help='whether only adaptive adj')
        parser.add_argument('--addaptadj',
                            action='store_true',
                            help='whether add adaptive adj')
        parser.add_argument('--randomadj',
                            action='store_true',
                            help='whether random initialize adaptive adj')
        parser.add_argument('--seq_length', type=int, default=12, help='')
        parser.add_argument('--nhid',
                            type=int,
                            default=40,
                            help='Number of channels for internal conv')
        parser.add_argument('--in_dim',
                            type=int,
                            default=2,
                            help='inputs dimension')
        parser.add_argument('--num_nodes',
                            type=int,
                            default=207,
                            help='number of nodes')
        parser.add_argument('--batch_size',
                            type=int,
                            default=64,
                            help='batch size')
        parser.add_argument('--dropout',
                            type=float,
                            default=0.3,
                            help='dropout rate')
        parser.add_argument(
            '--n_obs',
            default=None,
            help='Only use this many observations. For unit testing.')
        parser.add_argument('--apt_size', default=10, type=int)
        parser.add_argument('--cat_feat_gc', action='store_true')
        parser.add_argument('--fill_zeroes', action='store_true')
        parser.add_argument('--checkpoint', type=str, help='')
        return parent_parser

    def forward(self, x):
        # Input shape is (bs, features, n_nodes, n_timesteps)
        in_len = x.shape[3]
        if in_len < self.receptive_field:
            x = np.pad(x,
                       mode="constant",
                       pad_width=(self.receptive_field - in_len, 0, 0, 0))
        if self.cat_feat_gc:
            f1, f2 = x[:, [0]], x[:, 1:]
            x1 = self.start_conv(f1)
            x2 = self.leaky_relu(self.cat_feature_conv(f2))
            x = x1 + x2
        else:
            x = self.start_conv(x)
        skip = 0
        adjacency_matrices = self.fixed_supports
        # calculate the current adaptive adj matrix once per iteration
        if self.addaptadj:
            adp = npx.softmax(npx.relu(
                np.dot(self.nodevec1.data(),
                       self.nodevec2.data().T)),
                              axis=1)
            adjacency_matrices = self.fixed_supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # EACH BLOCK

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |   |-dil_conv -- tanh --|                |
            #         ---|                  * ----|-- 1x1 -- + -->	*x_in*
            #                |-dil_conv -- sigm --|    |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            # dilated convolution
            filter = np.tanh(self.filter_convs[i](residual))
            gate = npx.sigmoid(self.gate_convs[i](residual))
            x = filter * gate
            # parametrized skip connection
            s = self.skip_convs[i](x)  # what are we skipping??
            try:  # if i > 0 this works
                skip = skip[:, :, :, -s.size(3):]  # TODO(SS): Mean/Max Pool?
            except:
                skip = 0
            skip = s + skip
            if i == (self.blocks * self.layers -
                     1):  # last X getting ignored anyway
                break

            if self.do_graph_conv:
                graph_out = self.graph_convs[i](x, adjacency_matrices)
                x = x + graph_out if self.cat_feat_gc else graph_out
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.shape[3]:]  # TODO(SS): Mean/Max Pool?
            x = self.bn[i](x)

        x = npx.relu(skip)  # ignore last X?
        x = npx.relu(self.end_conv_1(x))
        x = self.end_conv_2(
            x)  # downsample to (bs, seq_length, 207, nfeatures)
        return x


def test_gwnet():
    (b, c, n, t) = (12, 8, 5, 13)
    x = np.random.randn(b, c, n, t)
    A = np.random.randn(n, n)

    model = GWNet(n, supports=[A])
    model.initialize()
    print(model)
    output = model(x)  # B, T, N, C
    print(output.shape)
    model.hybridize()
    output = model(x)  # B, T, N, C
    print(output.shape)


if __name__ == '__main__':
    # test_nconv()
    # test_gcn()
    test_gwnet()
