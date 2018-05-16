# encoding: utf-8

import numpy as np
import torch
from fileio import create_dir

__all__ = ['calc_rsq', 'calc_accu', 'calc_topk_accu',
           'argmax', 'time_it', 'create_dir']


def calc_rsq(y, yhat):
    residue = torch.add(y, torch.neg(yhat))
    ss_residue = torch.mean(torch.pow(residue, 2), dim=0)
    y_mean = torch.mean(y, dim=0)
    ss_total = torch.mean(torch.pow(torch.add(y, torch.neg(y_mean)), 2), dim=0)
    res = torch.add(torch.ones_like(ss_total),
                    torch.neg(torch.div(ss_residue, ss_total)))
    return res


def argmax(tensor, dim):
    _, res = torch.max(tensor, dim)
    return res


def calc_accu(y, yhat, argmax=True):
    # yhat = yhat.data
    if argmax:
        yhat = argmax(yhat, dim=1)
    eq = torch.eq(y, yhat)
    accuracy = torch.mean(torch.tensor(eq, dtype=torch.float32))
    return accuracy


def calc_topk_accu(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




def time_it(func, *args, **kwargs):
    import time
    t0 = time.time()
    func(*args, **kwargs)
    t1 = time.time()

    t = t1 - t0
    print("Time elapsed: {:.2f} seconds.".format(t))


def imshow(img):
    """
    functions to show an image

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision

    img = torchvision.utils.make_grid(img)

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # print labels
    # plt.title(' '.join(['%5s' % CLASSES[labels[j]] for j in range(4)]))

    plt.show()


###########################################################
# Tests

def test_calc_accu():
    a = range(10)
    b = [0, 1, 2, 9, 9, 5, 6, 1, 2, 1]

    t1 = torch.tensor(a)
    t2 = torch.tensor(b)

    eq = torch.eq(t1, t2)
    eq = torch.tensor(eq, dtype=torch.float32)
    ac = torch.mean(eq)

    assert ac.item() == 0.5
    assert calc_accu(t1, t2, argmax=False).item() == 0.5


def test_argmax():
    r = torch.randn(10000, 10)
    res = argmax(r, 1)

    assert res.shape[0] == 10000
    assert res.dtype is torch.int64


def test_calc_rsq():
    np.random.seed(369)

    n = 2333
    y_true = np.arange(n) / n - 0.5
    y_pred = y_true + np.random.randn(n) / 10

    t1 = torch.tensor(y_true, dtype=torch.float32)
    t2 = torch.tensor(y_pred, dtype=torch.float32)
    rsq = calc_rsq(t1, t2)
    assert abs(rsq - 0.8790188) < 1E-7


if __name__ == "__main__":
    test_calc_accu()
    test_argmax()
    test_calc_rsq()
