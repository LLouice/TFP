import os
from multiprocessing import cpu_count

import numpy as npo

import mxnet as mx
from mxnet import nd
from mxnet.gluon.data import Dataset, DataLoader

from config import Config
import utils
from model import GWNet


class MyDataset(Dataset):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        '''
        xs/ys: B T N C=2
        output: B N T C=1(speed, index=0)
        '''
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = npo.repeat(xs[-1:], num_padding, axis=0)
            y_padding = npo.repeat(ys[-1:], num_padding, axis=0)
            xs = npo.concatenate([xs, x_padding], axis=0)
            ys = npo.concatenate([ys, y_padding], axis=0)

        self.xs = nd.transpose(nd.array(xs), axes=(0, 3, 2, 1))
        self.ys = nd.transpose(nd.array(ys),axes=(0, 3, 2, 1))[:, 0, :, :]

        self.ys = nd.expand_dims(self.ys, axis=1)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]

    def __len__(self):
        return self.xs.shape[0]


def load_dataset(dataset_dir,
                 batch_size,
                 valid_batch_size=None,
                 test_batch_size=None,
                 n_obs=None,
                 fill_zeroes=True) -> dict:
    '''
    key: x_train/val/test
    '''
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = npo.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x'].astype("float32")
        data['y_' + category] = cat_data['y'].astype("float32")
        if n_obs is not None:
            data['x_' + category] = data['x_' + category][:n_obs]
            data['y_' + category] = data['y_' + category][:n_obs]

    scaler = utils.StandardScaler(mean=data['x_train'][..., 0].mean(),
                                  std=data['x_train'][..., 0].std(),
                                  fill_zeroes=fill_zeroes)

    # normalize
    for category in ['train', 'val', 'test']:
        data['x_' + category][...,
                              0] = scaler.transform(data['x_' + category][...,
                                                                          0])
    data['scaler'] = scaler
    return data


class DataModule():
    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger

    def prepare_data(self):
        if self.logger is not None:
            self.logger.info(".... prepare_data ....")

        self.data = load_dataset(self.config.data,
                                 self.config.bs,
                                 self.config.bs,
                                 self.config.bs,
                                 n_obs=self.config.n_obs,
                                 fill_zeroes=self.config.fill_zeroes)
        self.scaler = self.data["scaler"]
        self.x_trn = self.data["x_train"]
        self.y_trn = self.data["y_train"]
        self.x_val = self.data["x_val"]
        self.y_val = self.data["y_val"]
        self.x_test = self.data["x_test"]
        self.y_test = self.data["y_test"]

        # graph data
        if self.config.device == "cpu":
            self.ctx = mx.cpu()
        else:
            self.ctx = mx.gpu()

        self.aptinit, self.supports = utils.make_graph_inputs(
            self.config, self.ctx)

        if self.logger is not None:
            self.logger.info(".... prepare_data done! ....")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.ds_trn = MyDataset(self.x_trn, self.y_trn, self.config.bs)
            self.ds_val = MyDataset(self.x_val, self.y_val, self.config.bs)
        if stage == "test" or stage is None:
            self.ds_test = MyDataset(self.x_test, self.y_test, self.config.bs)

    def train_dataloader(self):
        self.dl_trn = DataLoader(self.ds_trn,
                                 batch_size=self.config.bs,
                                 shuffle=True,
                                 num_workers=cpu_count(),
                                 pin_memory=True,
                                 thread_pool=True)
        return self.dl_trn

    def val_dataloader(self):
        self.dl_val = DataLoader(self.ds_val,
                                 batch_size=self.config.bs_dev,
                                 shuffle=False,
                                 num_workers=cpu_count(),
                                 pin_memory=True,
                                 thread_pool=True)
        return self.dl_val

    def test_dataloader(self):
        self.dl_test = DataLoader(self.ds_test,
                                  batch_size=self.config.bs_dev,
                                  shuffle=False,
                                  num_workers=cpu_count(),
                                  pin_memory=True,
                                  thread_pool=True)
        return self.dl_test


def test_dm():
    config = Config()
    parser = config.get_parser()
    parser = GWNet.add_model_specific_args(parser)
    conf = config.get_config(parser)
    print(conf)
    dm = DataModule(conf)
    dm.prepare_data()
    dm.setup()
    dl_trn = dm.train_dataloader()
    dl_val = dm.val_dataloader()
    dl_test = dm.test_dataloader()

    print(len(dl_trn))
    print(len(dl_val))
    print(len(dl_test))

    for x, y in dl_trn:
        print(x.shape, y.shape)
        break

    for x, y in dl_val:
        print(x.shape, y.shape)
        break

    for x, y in dl_test:
        print(x.shape, y.shape)
        break


if __name__ == '__main__':
    test_dm()
