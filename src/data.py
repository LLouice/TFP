import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from config import Config
from dataset import load_dataset, MyDataset
import utils


class DataModule(LightningDataModule):
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
        self.aptinit, self.supports = utils.make_graph_inputs(self.config)

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
                                 num_workers=self.config.nw0,
                                 pin_memory=True)
        return self.dl_trn

    def val_dataloader(self):
        self.dl_val = DataLoader(self.ds_val,
                                 batch_size=self.config.bs,
                                 shuffle=False,
                                 num_workers=self.config.nw1,
                                 pin_memory=True)
        return self.dl_val

    def test_dataloader(self):
        self.dl_test = DataLoader(self.ds_test,
                                  batch_size=self.config.bs,
                                  shuffle=False,
                                  num_workers=self.config.nw1,
                                  pin_memory=True)
        return self.dl_test


def test_dm():
    config = Config()
    parser = config.get_parser()
    parser.add_argument('--data',
                        type=str,
                        default='data/METR-LA',
                        help='data path')
    parser.add_argument(
        '--n_obs',
        default=None,
        help='Only use this many observations. For unit testing.')
    parser.add_argument('--fill_zeroes', action='store_true')
    conf = config.get_config()
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
