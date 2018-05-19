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
    ds = FutureTickDatasetNew('rb1701.SHF_20160801.hd5', 'valid_data', backward_window=224, forward_window=60)
    val_loader = get_future_loader_from_dataset(ds, batch_size=16)
    
    SAVE_MODEL_FP = 'saved_torch_models/resnet_preact2/model.pytorch'
    y, yhat = main_predict(val_loader, SAVE_MODEL_FP)

    criterion = nn.MSELoss()
    score = utils.calc_rsq(y, yhat)
    loss = criterion(y, yhat)
    print("Val_loss = {:+4.6f}".format(loss.item()))
    print("Val_score = {:+4.6f}".format(score.item()))

    # import numpy as np
    # print(yhat.shape)
    # np.save('y_true_{:d}'.format(batch_size), y.numpy())
    # np.save('y_pred_{:d}'.format(batch_size), yhat.numpy())


def test_dataset():
    from my_dataset import FutureTickDatasetNew, get_future_loader_from_dataset
    
    ds = FutureTickDatasetNew('Data/future_new/' + 'rb_20160801_20160831.hd5',
                              'valid_data', backward_window=224, forward_window=60)
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
    
    loader = get_future_loader_from_dataset(ds, batch_size=16)
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
    
    print("")


if __name__ == "__main__":
    test_dataset()
    # test()
