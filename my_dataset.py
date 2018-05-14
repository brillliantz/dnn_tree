# encoding: utf-8

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        x, y = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return torch.from_numpy(x), torch.from_numpy(y)


class FutureTickDataset(Dataset):
    """
    China Future Tick Data.

    """
    def __init__(self, backward_window=240, forward_window=60, transform=None,
                 cut_len=0):
        self.dir = 'Data/future'
        self.data_fp = os.path.join(self.dir,
                                    'rb1701_201608-201611_AddFeature_lite.hd5')
        self.mask_fp = os.path.join(self.dir,
                                    'mask_all_lite.hd5')

        self.cut_len = cut_len
        self.transform = transform

        # 预测时间跨度
        self.forward_window = forward_window
        self.backward_window = backward_window

        store = pd.HDFStore(self.data_fp, mode='r')
        self._df_raw = store['dataset']
        store.close()

        store = pd.HDFStore(self.mask_fp, mode='r')
        self._mask = store['mask_all']
        store.close()

        self._validate()

        self._cut()

        self.df = None

        self._preprocess()

    def _validate(self):
        assert self._df_raw.shape[0] == 200000

    def _cut(self):
        if not self.cut_len:
            return

        if isinstance(self.cut_len, int):
            self._df_raw = self._df_raw.iloc[: self.cut_len]
        elif isinstance(self.cut_len, float):
            self._df_raw = self._df_raw.iloc[: int(len(self._df_raw) * self.cut_len)]

    def _preprocess(self):
        # 要使用的自变量
        X_COLS = [#'book_pressure_norm',
            #'datetime'
            'mid',
            'last', 'bidprice1', 'askprice1', 'bidqty1', 'askqty1', 'volume_diff', 'oi_diff',
        ]
        XY_COLS = X_COLS.copy()
        if 'mid' not in XY_COLS:
            XY_COLS.append('mid')

        self._df_raw = self._df_raw.reindex(columns=XY_COLS)
        # 目标预测变量
        self._df_raw.loc[:, 'y'] = self._df_raw['mid'].pct_change(self.forward_window).shift(-self.forward_window)

        self.df = self._df_raw.loc[self._mask].dropna()

        # X, Y都是np.ndarray
        self.x = self.df[X_COLS].values
        self.y = self.df['y'].values.reshape([-1, 1])

        # TODO
        self.x = (self.x - self.x.mean(axis=0)) / self.x.std(axis=0)

        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)

        # [time_window, n_feature] to [n_feature, time_window]
        # because torch.nn.Conv inputs are of shape [batch, n_channels, sample_shape]
        self.x = np.swapaxes(self.x, 0, 1)
        self.y = np.swapaxes(self.y, 0, 1)

    def __len__(self):
        return len(self.df) - self.backward_window

    def __getitem__(self, idx):
        sample_x = self.x[:, idx: idx + self.backward_window]
        sample_y = self.y[:, idx + self.backward_window]

        if self.transform is not None:
            sample_x, sample_y = self.transform((sample_x, sample_y))

        return sample_x, sample_y


def test_dataset_for_loop():
    composed = torchvision.transforms.Compose([ToTensor()])
    ds = FutureTickDataset(240, 60,
                           transform=composed
                           )

    for i in range(len(ds)):
        x, y = ds[i]
        # print(i, type(x), x.shape, type(y), y.shape)


def test_dataset_loader():
    ds = FutureTickDataset(240, 60)

    loader = DataLoader(ds, batch_size=4, shuffle=False,)

    for i_batch, (x_batched, y_batched) in enumerate(loader):
        pass
        # print(i_batch)



def time_it(func, *args, **kwargs):
    import time
    t0 = time.time()
    func(*args, **kwargs)
    t1 = time.time()

    t = t1 - t0
    print("Time elapsed: {:.2f} seconds.".format(t))


if __name__ == "__main__":
    # test_dataset_for_loop()
    time_it(
        #test_dataset_for_loop
        test_dataset_loader
    )
