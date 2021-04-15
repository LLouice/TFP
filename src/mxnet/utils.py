import os
import argparse
import pickle
import numpy as npo

from mxnet import np

import pandas as pd
import scipy.sparse as sp
from scipy.sparse import linalg


class StandardScaler():
    def __init__(self, mean, std, fill_zeroes=True):
        self.mean = mean
        self.std = std
        self.fill_zeroes = fill_zeroes

    def transform(self, data):
        if self.fill_zeroes:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = npo.array(adj.sum(1))
    d_inv_sqrt = npo.power(rowsum, -0.5).flatten()
    d_inv_sqrt[npo.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(
        npo.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = npo.array(adj.sum(1)).flatten()
    d_inv = npo.power(rowsum, -1).flatten()
    d_inv[npo.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(npo.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = npo.array(adj.sum(1))
    d_inv_sqrt = npo.power(d, -0.5).flatten()
    d_inv_sqrt[npo.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(
        d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = npo.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(npo.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']


def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [
            calculate_normalized_laplacian(adj_mx).astype(
                npo.float32).todense()
        ]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(npo.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [npo.diag(npo.ones(adj_mx.shape[0])).astype(npo.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir,
                 batch_size,
                 valid_batch_size=None,
                 test_batch_size=None,
                 n_obs=None,
                 fill_zeroes=True):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = npo.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x'].astype("float32")
        data['y_' + category] = cat_data['y'].astype("float32")
        if n_obs is not None:
            data['x_' + category] = data['x_' + category][:n_obs]
            data['y_' + category] = data['y_' + category][:n_obs]
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(),
                            std=data['x_train'][..., 0].std(),
                            fill_zeroes=fill_zeroes)
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][...,
                              0] = scaler.transform(data['x_' + category][...,
                                                                          0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'],
                                      batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'],
                                    valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'],
                                     test_batch_size)
    data['scaler'] = scaler
    return data


# TODO: port this
def calc_tstep_metrics(model, device, test_loader, scaler, realy,
                       seq_length) -> pd.DataFrame:
    model.eval()
    outputs = []
    for _, (x, __) in enumerate(test_loader.get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
    test_met = []

    for i in range(seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        pred = torch.clamp(pred, min=0., max=70.)
        real = realy[:, :, i]
        test_met.append([x.item() for x in calc_metrics(pred, real)])
    test_met_df = pd.DataFrame(test_met, columns=['mae', 'mape',
                                                  'rmse']).rename_axis('t')
    return test_met_df, yhat


def _to_ser(arr):
    return pd.DataFrame(arr.cpu().detach().numpy()).stack().rename_axis(
        ['obs', 'sensor_id'])


def make_pred_df(realy, yhat, scaler, seq_length):
    df = pd.DataFrame(
        dict(y_last=_to_ser(realy[:, :, seq_length - 1]),
             yhat_last=_to_ser(
                 scaler.inverse_transform(yhat[:, :, seq_length - 1])),
             y_3=_to_ser(realy[:, :, 2]),
             yhat_3=_to_ser(scaler.inverse_transform(yhat[:, :, 2]))))
    return df


def make_graph_inputs(args, ctx):
    sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adjdata, args.adjtype)
    supports = [np.array(i, ctx=ctx) for i in adj_mx]
    aptinit = None if args.randomadj else supports[
        0]  # ignored without do_graph_conv and add_apt_adj
    if args.aptonly:
        if not args.addaptadj and args.do_graph_conv:
            raise ValueError('WARNING: not using adjacency matrix')
        supports = None
    return aptinit, supports


def get_shared_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='')
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
                        choices=ADJ_CHOICES)
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
    return parser
