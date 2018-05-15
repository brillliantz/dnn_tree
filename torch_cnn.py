# encoding: utf-8

import os

import gpu_config_torch
import torch
# Assume that we are on a CUDA machine, then this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("torch device: ", device)

from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from my_dataset import FutureTickDataset
from demo_fully_diff_ndf import calc_rsq

SAVE_MODEL_FP = 'saved_torch_models_3mil/cnn.model'


CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def calc_rsq_torch(y, yhat):
    residue = torch.add(y, torch.neg(yhat))
    ss_residue = torch.mean(torch.pow(residue, 2), dim=0)
    y_mean = torch.mean(y, dim=0)
    ss_total = torch.mean(torch.pow(torch.add(y, torch.neg(y_mean)), 2), dim=0)
    res = torch.add(torch.ones_like(ss_total),
                    torch.neg(torch.div(ss_residue, ss_total)))
    return res


def torch_argmax(tensor, dim):
    _, res = torch.max(tensor, dim)
    return res


def calc_accu_torch(y, yhat):
    yhat = yhat.data
    yhat = torch_argmax(yhat, dim=1)
    accu = (y == yhat).to(device, dtype=torch.float32).mean()
    return accu


class Net(nn.Module):
    def __init__(self, fc_in_size, output_size, in_channels, dim=2):
        super(Net, self).__init__()

        if dim == 1:
            conv_func = nn.Conv1d
            pool_func = nn.MaxPool1d
        else:
            conv_func = nn.Conv2d
            pool_func = nn.MaxPool2d

        self.output_size = output_size
        self.fc_in_size = fc_in_size
        self.in_channels = in_channels

        self.conv1 = conv_func(in_channels=self.in_channels, out_channels=8,kernel_size=5)
        self.pool1 = pool_func(kernel_size=2, stride=2)
        # self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = conv_func(in_channels=8, out_channels=16, kernel_size=3)
        self.pool2 = pool_func(kernel_size=2, stride=2)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.conv3 = conv_func(in_channels=16, out_channels=64, kernel_size=3)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=self.fc_in_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(256, self.output_size)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        # shapes = x.shape[1:]
        # shape = np.prod(shapes) # 16 * 6 * 6
        shape = self.fc_in_size
        x = x.view(-1, shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def show(self, desc):
        print("\n\n ", desc)
        ps = list(self.parameters())
        print(ps[0][0])


def get_cifar_10(show=False):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='Data/cifar10', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='Data/cifar10', train=False,
                                           download=True, transform=transform)
    #if show:
        #show_imgs(trainloader, CLASSES)

    return trainset, testset


# functions to show an image
def imshow(img):
    import matplotlib.pyplot as plt
    import numpy as np

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # print labels
    # plt.title(' '.join(['%5s' % CLASSES[labels[j]] for j in range(4)]))

    plt.show()


def show_imgs(data_loader, classes):
    # get some random training images
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))


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


def test_model_toy(test_loader, model):
    dataiter = iter(test_loader)
    features, labels = dataiter.next()

    # print Groundtruth
    print('GroundTruth: ', ' '.join('%5s' % CLASSES[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(features))

    # predict
    outputs = model(features)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % CLASSES[predicted[j]]
                                  for j in range(4)))


def test_model_classification(test_loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            features, labels = data
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def model_predict(test_loader, model, out_tensor=True):
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

    if not out_tensor:
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

    return y_true, y_pred


def model_score(test_loader, model, score_func, loss_func):
    y_true, y_pred = model_predict(test_loader, model)
    score = score_func(y_true, y_pred)
    loss = loss_func(y_pred, y_true)

    return score, loss


def main():
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
