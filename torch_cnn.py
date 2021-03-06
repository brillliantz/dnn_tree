# encoding: utf-8

import os

from gpu_config_torch import device
import torch

from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn

import torch.optim as optim

from my_dataset import FutureTickDataset

SAVE_MODEL_FP = 'saved_torch_models_3mil/cnn.model'


CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train_model(model,
                optimizer, loss_func, score_func,
                n_epoch, train_loader, test_loader,
                save_and_eval_interval=2000):
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        # for i, (features, labels) in tqdm(enumerate(train_loader, 0)):
        for i, (features, labels) in enumerate(train_loader, 0):
            features, labels = features.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(features)

            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            loss_value = loss.item()
            # loss_value = ((labels.numpy() - outputs.detach().numpy())**2).mean()
            running_loss += loss_value
            if i % save_and_eval_interval == (save_and_eval_interval - 1):  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3e' %
                      (epoch + 1, i + 1, running_loss / save_and_eval_interval))
                running_loss = 0.0

        # after per epoch
        score, loss = model_score(test_loader, model, score_func, loss_func)
        print("Loss = {:+4.6f}".format(loss.item()))
        print("Score = {:+4.6f}".format(score.item()))
        torch.save(model.state_dict(), SAVE_MODEL_FP)
        print("Model saved.")


            # DEBUG
            # model.show(str(i))
            # DEBUG

    print('Finished Training')


def model_predict(test_loader, model, out_numpy=False):
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)

            # outputs = outputs.numpy()
            # labels = labels.numpy()
            y_true_list.append(labels)
            y_pred_list.append(outputs)

    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)

    if out_numpy:
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

    return y_true, y_pred


def model_score(test_loader, model, score_func, loss_func):
    y_true, y_pred = model_predict(test_loader, model)
    score = score_func(y_true, y_pred)
    loss = loss_func(y_pred, y_true)

    return score, loss


def main():
    from my_dataset import get_cifar_10
    trainset, testset = get_cifar_10(show=False)

    batch_size = 16

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)


    net = Net(64*4*4, 10, in_channels=3, dim=2)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_model(model=net,
                optimizer=optimizer, loss_func=criterion, score_func=calc_accu_torch,
                n_epoch=100, train_loader=trainloader, test_loader=testloader,
                save_and_eval_interval=200)
    #model_score(trainloader, net, calc_accu_torch, criterion)
    #model_score(testloader, net, calc_accu_torch, criterion)


def main_future():
    ds = FutureTickDataset(240, 60, cut_len=None, lite_version=False)
    ds_test = FutureTickDataset(240, 60, cut_len=None)
    print("Train dataset len: {:d}\n"
          "Test dataset len: {:d}".format(len(ds), len(ds_test)))

    y_abs = np.abs(ds.y)
    print("Y mean = {:.3e}, Y median = {:.3e}".format(np.mean(y_abs), np.median(y_abs)))

    ds_len = len(ds)
    batch_size = 20
    itr_per_epoch = ds_len // batch_size
    print("Iterations needed per epoch: {:d}".format(itr_per_epoch))

    trainloader = DataLoader(ds, batch_size=batch_size, shuffle=False,)
    testloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,)

    net = Net(64 * 56, 1, in_channels=8, dim=1)

    if os.path.exists(SAVE_MODEL_FP):
        net.load_state_dict(torch.load(SAVE_MODEL_FP,
                                       #map_location=repr(device)
                                       ))
        print("Load model from {:s}".format(SAVE_MODEL_FP))

    net.to(device)

    criterion = nn.MSELoss()

    train_mode = 1
    if train_mode:
        optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

        train_model(model=net,
                    optimizer=optimizer, loss_func=criterion, score_func=calc_rsq_torch,
                    n_epoch=100, train_loader=trainloader, test_loader=testloader,
                    save_and_eval_interval=6000)

        torch.save(net.state_dict(), SAVE_MODEL_FP)
        print("Model saved.")

        #rsq, loss = model_score(trainloader, net, calc_rsq_torch, criterion)
        #print("Loss = {:.3e}%".format(loss))
        #print("Rsquared = {:.2f}%".format(rsq * 100))
    else:
        y_true, y_pred = model_predict(trainloader, net)
        score = calc_rsq_torch(y_true, y_pred)
        loss = criterion(y_true, y_pred)
        print("Loss = {:.3e}".format(loss.item()))
        print("Score = {:.3e}".format(score.item()))

        y = y_true.numpy().squeeze()
        yhat = y_pred.numpy().squeeze()
        np.save('ytrue.npz', y)
        np.save('ypred.npz', yhat)

        y_true, y_pred = model_predict(trainloader, net)


if __name__ == "__main__":
    main_future()
    # main()
