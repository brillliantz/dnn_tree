# encoding: utf-8

import os

import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import preprocessing


class FutureTickDataset(Dataset):
    """
    China Future Tick Data.

    """
    def __init__(self, backward_window=240, forward_window=60, transform=None,
                 cut_len=0, lite_version=True, train=True):
        self.train_mode = train
        self.dir = 'Data/future'
        if lite_version:
            self.data_fp = os.path.join(self.dir,
                                        'rb1701_201608-201611_AddFeature_lite.hd5')
            self.mask_fp = os.path.join(self.dir,
                                        'mask_all_lite.hd5')
        else:
            self.data_fp = os.path.join(self.dir,
                                        'rb1701_201608-201611_AddFeature.hd5')
            self.mask_fp = os.path.join(self.dir,
                                        'mask_all.hd5')

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
        #assert self._df_raw.shape[0] == 200000
        #assert dataset.shape[0] == 3245785
        pass

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

        # TODO: std
        roll = self._df_raw.rolling(window=self.backward_window, axis=0)
        self._df_raw = (self._df_raw - roll.mean()) / roll.std()

        self.df = self._df_raw.loc[self._mask].dropna()

        # X, Y都是np.ndarray
        self.x = self.df[X_COLS].values
        self.y = self.df['y'].values.reshape([-1, 1])
        
        # DEBUG
        # self.y = np.random.lognormal(size=(len(self.y), 1))
        # self.y = np.random.randn(len(self.y), 1)

        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)

        train_ratio = 0.7
        train_len = int(len(self.x) * train_ratio)
        if self.train_mode:
            self.x = self.x[: train_len]
            self.y = self.y[: train_len]
        else:
            self.x = self.x[train_len:]
            self.y = self.y[train_len:]

        # [time_window, n_feature] to [n_feature, time_window]
        # because torch.nn.Conv inputs are of shape [batch, n_channels, sample_shape]
        self.x = np.swapaxes(self.x, 0, 1)
        self.y = np.swapaxes(self.y, 0, 1)
        
        import gc
        del self._df_raw
        del self.df
        gc.collect()

    def __len__(self):
        return self.x.shape[1] - self.backward_window

    def __getitem__(self, idx):
        sample_x = self.x[:, idx: idx + self.backward_window]
        sample_y = self.y[:, idx + self.backward_window]

        if self.transform is not None:
            sample_x, sample_y = self.transform((sample_x, sample_y))

        return sample_x, sample_y


class FutureTickDatasetNew(Dataset):
    """
    China Future Tick Data.

    """
    def __init__(self, data_paths, key='valid_data',
                 backward_window=240, forward_window=60,
                 train_mode=True, train_ratio=0.7,
                 ):
        if not isinstance(data_paths, list):
            data_paths = [data_paths]
            
        self.forward_window = forward_window
        self.backward_window = backward_window
        self.train_mode = train_mode
        self.train_ratio = train_ratio
        
        self._df_raw = None
        self.df = None
        
        self.index = None
        
        self.x = None
        self.y = None
        
        self._df_raw = self._load_data(data_paths, key)
        self._split_train_test()
        self._preprocess()
        self._validate()
    
    def _load_data(self, paths, key):
        dfs = []
        for p in paths:
            df = pd.read_hdf(p, key)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        return df
    
    def _split_train_test(self):
        n = len(self._df_raw)
        train_len = int(n * self.train_ratio)
        if self.train_mode:
            self._df_raw = self._df_raw[: train_len]
        else:
            self._df_raw = self._df_raw[train_len:]
        
    def _preprocess(self):
        # Roll dirty index to get complete dirty_index
        self._df_raw.loc[:, 'dirty_index'].iloc[: self.backward_window] = True
        self._df_raw.loc[:, 'dirty_index'] = \
            preprocessing.roll_dirty_index(self._df_raw['dirty_index'],
                                           backward_len=0,
                                           forward_predict_len=max(self.forward_window, self.backward_window))
        # 要使用的自变量
        X_COLS = ['mid', 'last',
                  'bidprice1', 'askprice1',
                  'bidvolume1', 'askvolume1',
                  'volume_diff', 'oi_diff',
                  ]
        XY_COLS = X_COLS.copy() + ['y']
        if 'mid' not in XY_COLS:
            XY_COLS.append('mid')
        
        self._df_raw.loc[:, 'y'] = self._df_raw['mid'].pct_change(self.forward_window).shift(-self.forward_window)
        self.df = self._df_raw.reindex(columns=XY_COLS)
        
        # TODO: std
        roll = self.df.rolling(window=self.backward_window, axis=0)
        std_eq_zero_mask = (roll.std() < 1e-8).any(axis=1)
        std_eq_zero_mask = preprocessing.roll_dirty_index(std_eq_zero_mask,
                                                          backward_len=0,
                                                          forward_predict_len=max(self.forward_window, self.backward_window))
        didx = self._df_raw['dirty_index']
        didx = np.logical_or(std_eq_zero_mask, didx)
        self._df_raw.loc[:, 'dirty_index'] = didx
        self.dirty_index = didx.copy()
        self.index = self.dirty_index.index.values[np.logical_not(self.dirty_index.values)]
        
        self.df = (self.df - roll.mean()) / roll.std()
        
        self.x = self.df[X_COLS].values
        self.y = self.df['y'].values.reshape([-1, 1])
        
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)
        
        # Because torch.nn.Conv inputs are of shape [batch, n_channels, sample_shape]
        # [time_window, n_feature] -> [n_feature, time_window]
        self.x = np.swapaxes(self.x, 0, 1)
        self.y = np.swapaxes(self.y, 0, 1)

        # self.x = torch.Tensor(self.x)
        # self.y = torch.Tensor(self.y)
        
    def _validate(self):
        assert self.df.shape[0] == self._df_raw.shape[0]
        for df in [self._df_raw, self.df]:
            clean_data = df.loc[self.index]
            nan_count = clean_data.isnull().sum().sum()
            assert nan_count == 0
        
        print("Validate pass.")
        
    def show_statistics(self):
        df = self.df.loc[self.index]
        print("\n" + "="*5 + "Dataset statistics: ")
        print(df.describe())
        print("\n" + "="*5 + "Median of abs(y): ")
        print(np.median(np.abs(df['y']), axis=0))
        
    def __len__(self):
        return len(self.index) - self.backward_window
    
    def __getitem__(self, idx):
        start = self.index[idx]
        sample_x = self.x[:, start: start + self.backward_window]
        sample_y = self.y[:, self.backward_window]
        
        return sample_x, sample_y


class FutureBarDataset(Dataset):
    """
    China Future Tick Data.

    """
    def __init__(self, train=True, cut_len=0):
        # self.dir = 'Data/future'
        if train:
            x = np.load('Data/futures-historical-data/rb_X_train_1501-1701.npy')
            y = np.load('Data/futures-historical-data/rb_Y_train_1501-1701.npy')
        else:
            x = np.load('Data/futures-historical-data/rb_X_test_1501-1701.npy')
            y = np.load('Data/futures-historical-data/rb_Y_test_1501-1701.npy')

        # test with minimal data size
        if cut_len:
            x = x[:cut_len]
            y = y[:cut_len]

        # from keras.utils.np_utils import to_categorical
        # y = to_categorical(y, num_classes=3)
        y = y.reshape([-1])

        self.x = np.asanyarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.x = np.swapaxes(self.x, 1, 2)
        # self.y = np.swapaxes(self.y, 1, 2)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample_x = self.x[idx]
        sample_y = self.y[idx]

        return sample_x, sample_y


def get_future_bar_classification_data(batch_size, cut_len):
    ds = FutureBarDataset(train=True, cut_len=cut_len)
    ds_test = FutureBarDataset(train=False, cut_len=0)
    print("Train dataset len: {:d}\n"
          "Test dataset len: {:d}".format(len(ds), len(ds_test)))

    y_abs = np.abs(ds.y)
    print("Y mean = {:.3e}, Y median = {:.3e}".format(np.mean(y_abs), np.median(y_abs)))

    ds_len = len(ds)
    itr_per_epoch = ds_len // batch_size
    print("Iterations needed per epoch: {:d}".format(itr_per_epoch))

    trainloader = DataLoader(ds, batch_size=batch_size, shuffle=False,)
    testloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,)

    return trainloader, testloader


def get_future_loader(batch_size, cut_len, lite_version=True):
    ds_train = FutureTickDataset(224, 60, cut_len=cut_len, train=True, lite_version=lite_version)
    ds_test = FutureTickDataset(224, 60, cut_len=cut_len, train=False, lite_version=lite_version)
    print("Train dataset len: {:d}\n"
          "Test dataset len: {:d}".format(len(ds_train), len(ds_test)))

    y_abs = np.abs(ds_train.y)
    print("Train dataset Y mean = {:.3e}, Y median = {:.3e}".format(np.mean(y_abs), np.median(y_abs)))
    y_abs = np.abs(ds_test.y)
    print("Test dataset Y mean = {:.3e}, Y median = {:.3e}".format(np.mean(y_abs), np.median(y_abs)))

    ds_len = len(ds_train)
    itr_per_epoch = ds_len // batch_size
    print("Iterations needed per epoch: {:d}".format(itr_per_epoch))

    trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=False,)
    testloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,)

    return trainloader, testloader


def get_future_loader_from_dataset(dataset, batch_size):
    print("Train dataset len: {:d}\n"
          "Test dataset len: {:d}".format(len(dataset), len(dataset)))
    
    # dataset.show_statistics()
    
    ds_len = len(dataset)
    itr_per_epoch = ds_len // batch_size
    print("Iterations needed per epoch: {:d}".format(itr_per_epoch))
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,)
    
    return loader


def test_dataset_for_loop():
    ds = FutureTickDataset(224, 60, cut_len=40000, train=True, lite_version=True)

    for i in range(len(ds)):
        x, y = ds[i]
        pass
        # print(i, type(x), x.shape, type(y), y.shape)


def test_dataset_loader():
    ds = FutureTickDataset(240, 60, lite_version=False)

    loader = DataLoader(ds, batch_size=4, shuffle=False,)

    from tqdm import tqdm
    for i_batch, (x_batched, y_batched) in tqdm(enumerate(loader)):
        pass
        # print(i_batch)


def get_cifar_10(batch_size, shuffle, num_workers=1):
    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='Data/cifar10', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='Data/cifar10', train=False,
                                           download=True, transform=transform)
    #if show:
    #show_imgs(trainloader, CLASSES)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def load_imagenet(data_dir, batch_size, n_workers):
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=n_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=True)

    return train_loader, val_loader


if __name__ == "__main__":
    # a, b = get_future_bar_classification_data(16)

    # test_dataset_for_loop()
    from utils import time_it
    time_it(
        test_dataset_for_loop
        # test_dataset_loader
        # get_cifar_10, 10, shuffle=True
    )
