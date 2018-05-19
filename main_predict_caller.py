# encoding: utf-8

import utils
import torch.nn as nn

# 通过main_predict这个接口即可调用模型进行infer
from main import main_predict


def test():
    from my_dataset import get_future_loader
    
    batch_size = 32
    train_loader, val_loader = get_future_loader(batch_size=batch_size, cut_len=4000, lite_version=False)
    
    SAVE_MODEL_FP = 'saved_torch_models/resnet_preact2/model.pytorch'
    y, yhat = main_predict(val_loader, SAVE_MODEL_FP)

    criterion = nn.MSELoss()
    score = utils.calc_rsq(y, yhat)
    loss = criterion(y, yhat)
    print("Val_loss = {:+4.6f}".format(loss.item()))
    print("Val_score = {:+4.6f}".format(score.item()))

    import numpy as np
    print(yhat.shape)
    np.save('y_true_{:d}'.format(batch_size), y.numpy())
    np.save('y_pred_{:d}'.format(batch_size), yhat.numpy())


if __name__ == "__main__":
    test()
