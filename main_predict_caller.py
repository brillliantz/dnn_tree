# encoding: utf-8

import utils
import torch
import torch.nn as nn
import numpy as np

# 通过main_predict这个接口即可调用模型进行infer
from main import main_predict


def test():
    from my_dataset import FutureTickDatasetNew, get_future_loader_from_dataset
    
    batch_size = 128
    # from my_dataset import get_future_loader
    # train_loader, val_loader = get_future_loader(batch_size=batch_size, cut_len=40000, lite_version=True)
    month = [#(20160801, 20160831),
              #(20160901, 20160930),
              #(20161001, 20161031),
              (20161101, 20161130),
              #(20161201, 20161231),
              #(20170101, 20170131),
              ]
    folder = 'Data/future_new/'
    start, end = month[0]
    fn = "rb_{}_{}.hd5".format(start, end)
    hdf_path = folder + fn

    cut_len = 0
    ds = FutureTickDatasetNew(hdf_path,
                              'valid_data', backward_window=224, forward_window=60,
                              train_mode=True, train_ratio=1.0, cut_len=cut_len)
    val_loader = get_future_loader_from_dataset(ds, batch_size=batch_size)
    
    SAVE_MODEL_FP = 'saved_torch_models/r0.1_new/best_checkpoint.pytorch'
    y, yhat = main_predict(val_loader, SAVE_MODEL_FP)

    criterion = nn.MSELoss()
    score = utils.calc_rsq(y, yhat)
    loss = criterion(y, yhat)
    print("Val_loss = {:+4.6f}".format(loss.item()))
    print("Val_score = {:+4.6f}".format(score.item()))
    
    yhat = yhat.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    assert len(yhat) == len(ds.index)
    
    to_save = add_time_index_and_save(y, yhat, ds)
    print(to_save.describe())
    to_save.to_hdf('Data/r0.1_new_yhat/' + fn + '_yhat.hd5', key='yhat')


def add_time_index_and_save(y, yhat, ds):
    df = ds._df_raw.reindex(columns=['date', 'time', 'mid', 'y'])
    y_idx = ds.index + ds.backward_window - 1
    df.loc[:, 'yhat'] = np.nan
    df.loc[:, 'y_norm'] = np.nan
    df.loc[y_idx, 'yhat'] = yhat
    df.loc[y_idx, 'y_norm'] = y

    print(df.loc[y_idx, 'y'].describe())

    col = 'y'
    df.loc[:, col+'_restore' ] = df['y_norm'] * ds.rstd[col] + ds.rmean[col]
    df.loc[:, 'yhat'+'_restore' ] = df['yhat'] * ds.rstd[col] + ds.rmean[col]
    roll = df['y'].rolling(ds.backward_window)
    df.loc[:, 'y_restore2'] = df['y_norm'] * roll.std() + roll.mean()
    
    two = df[['y', 'yhat_restore']]
    two = two.loc[ds.index].dropna()
    print("Nan count: ", two.isnull().sum().sum())
    print("Correlation: ", two.corr().iloc[0, 1])
    print("rsq between y & yhat_restore: ",
          utils.calc_rsq(torch.Tensor(two['y'].values),
                         torch.Tensor(two['yhat_restore'].values)))
    
    return df.reindex(columns=['date', 'time', 'mid', 'y', 'y_norm', 'yhat', 'yhat_restore'])


def compare_with_old_res(yhat):
    """
    
    Parameters
    ----------
    yhat : np.ndarray

    Returns
    -------

    """
    compare_y_pred = np.load('y_pred_val_40k_r0.1.npy')
    to_be_compare = yhat
    
    abs_diff = np.abs(to_be_compare - compare_y_pred)
    res = abs_diff.sum()
    print("Sum of abs. diff: {:.4e}. Mean diff {:.4e}".format(res, abs_diff.mean()))
    is_all_close = np.allclose(to_be_compare, compare_y_pred, atol=1e-5, rtol=1e-3)
    print("allclose: {}".format(is_all_close))


def test_dataset():
    
    from my_dataset import get_future_loader_from_dataset
    from my_dataset import FutureTickDataset
    ds = FutureTickDataset(224, 60, cut_len=40000, train=True, lite_version=True)
    '''
    from my_dataset import FutureTickDatasetNew, get_future_loader_from_dataset
        ds = FutureTickDatasetNew(['Data/future_new/' + 'rb_20160801_20160831.hd5',
                               #'Data/future_new/' + 'rb_20160901_20160930.hd5',
                               #'Data/future_new/' + 'rb_20161001_20161031.hd5',
                               #'Data/future_new/' + 'rb_20161101_20161130.hd5',
                               ],
                              'valid_data', backward_window=224, forward_window=60,
                               train_mode=True, train_ratio=1)
    '''
    for i in range(len(ds)):
        x, y = ds[i]
        assert len(x.shape) == 2
        try:
            assert not np.any(np.isnan(x))
            assert not np.any(np.isnan(y))
        except AssertionError:
            print("error in for loop")
            real_i = ds.index[i]
            df = ds._df_raw.iloc[real_i: real_i + ds.backward_window]
            print(i, real_i)
            print(np.isnan(x).sum(axis=0))
    print("loop test done")
    
    loader = get_future_loader_from_dataset(ds, batch_size=128)
    for i, (x, y) in enumerate(loader):
        try:
            assert len(x.shape) == 3
            assert not np.any(np.isnan(x))
            assert not np.any(np.isnan(y))
        except AssertionError:
            print("error in loader")
            real_i = ds.index[i]
            x1 = x.numpy()
            print(i, real_i)
            print(np.isnan(x1).sum(axis=0))
    print("loader test done")
    

if __name__ == "__main__":
    # test_dataset()
    test()
