import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils import StandardScaler


class MyDataset(Dataset):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        '''
        xs/ys: B T N C=2
        output: B N T C=1(speed, index=0)
        '''
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.xs = torch.from_numpy(xs).transpose(1,3).float()
        self.ys = torch.from_numpy(ys).transpose(1,3)[:, 0, :, :].float()
        self.ys = self.ys.unsqueeze(1)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]

    def __len__(self):
        return self.xs.size(0)


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
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x'].astype("float32")
        data['y_' + category] = cat_data['y'].astype("float32")
        if n_obs is not None:
            data['x_' + category] = data['x_' + category][:n_obs]
            data['y_' + category] = data['y_' + category][:n_obs]

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(),
                            std=data['x_train'][..., 0].std(),
                            fill_zeroes=fill_zeroes)

    # normalize
    for category in ['train', 'val', 'test']:
        data['x_' + category][...,
                              0] = scaler.transform(data['x_' + category][...,
                                                                          0])
    data['scaler'] = scaler
    return data
