# encoding: utf-8

import utils
import torch.nn as nn
import numpy as np

# 通过main_predict这个接口即可调用模型进行infer
from main import main_predict


def test():
    from my_dataset import get_future_loader
    from my_dataset import FutureTickDatasetNew, get_future_loader_from_dataset
    
    batch_size = 16
    # train_loader, val_loader = get_future_loader(batch_size=batch_size, cut_len=40000, lite_version=True)
    '''
    '''
    ds = FutureTickDatasetNew(['Data/future_new/' + 'rb_20160801_20160831.hd5'],
                              'valid_data', backward_window=224, forward_window=60,
                              train_mode=True, train_ratio=1.0, cut_len=10000)
    val_loader = get_future_loader_from_dataset(ds, batch_size=batch_size)
    
    SAVE_MODEL_FP = 'saved_torch_models/r0.1_new/best_checkpoint.pytorch'
    y, yhat = main_predict(val_loader, SAVE_MODEL_FP)

    criterion = nn.MSELoss()
    score = utils.calc_rsq(y, yhat)
    loss = criterion(y, yhat)
    print("Val_loss = {:+4.6f}".format(loss.item()))
    print("Val_score = {:+4.6f}".format(score.item()))
    
    yhat = yhat.cpu().numpy().squeeze()
    y = y.numpy().squeeze()
    assert len(yhat) == len(ds.index)

    df = ds._df_raw.reindex(#index=ds.index,
            columns=['date', 'time', 'mid', 'y'])
    y_idx = ds.index + ds.backward_window - 1
    df.loc[:, 'yhat'] = np.nan
    df.loc[:, 'y_train'] = np.nan
    df.loc[y_idx, 'yhat'] = yhat
    df.loc[y_idx, 'y_train'] = y

    col = 'y'
    df.loc[:, col+'_restore' ] = df['y_train'] * ds.rstd[col] + ds.rmean[col]
    df.loc[:, 'yhat'+'_restore' ] = df['yhat'] * ds.rstd[col] + ds.rmean[col]
    roll = df['y'].rolling(ds.backward_window)
    df.loc[:, 'y_restore2'] = df['y_train'] * roll.std() + roll.mean()
    
    import torch
    two = df[['y', 'yhat_restore']]
    two = two.loc[ds.index].dropna()
    print("Nan: ", two.isnull().sum().sum())
    print(two.corr().iloc[0, 1])
    print(utils.calc_rsq(torch.Tensor(two['y'].values),
                         torch.Tensor(two['yhat_restore'].values)))

    '''
    compare_y_pred = np.load('y_pred_val_40k_r0.1.npy')
    to_be_compare = yhat.cpu().numpy()
    
    abs_diff = np.abs(to_be_compare - compare_y_pred)
    res = abs_diff.sum()
    print("Sum of abs. diff: {:.4e}. Mean diff {:.4e}".format(res, abs_diff.mean()))
    is_all_close = np.allclose(to_be_compare, compare_y_pred, atol=1e-5, rtol=1e-3)
    print("allclose: {}".format(is_all_close))
    '''
    pass
    # import numpy as np
    # print(yhat.shape)
    # np.save('y_true_{:d}'.format(batch_size), y.numpy())
    # np.save('y_pred_{:d}'.format(batch_size), yhat.numpy())


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
