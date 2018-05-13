# encoding: utf-8

import gpu_config_torch
import torch
# Assume that we are on a CUDA machine, then this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("torch device: ", device)

from tqdm import tqdm
import numpy as np

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

BATCH_SIZE = 8
CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.output_size = 10
        self.fc_in_size = 16 * 6 * 6

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=self.fc_in_size, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(84, self.output_size)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # shapes = x.shape[1:]
        # shape = np.prod(shapes) # 16 * 6 * 6
        shape = self.fc_in_size
        x = x.view(-1, shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_cifar_10(show=False):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='Data/cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='Data/cifar10', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    if show:
        show_imgs(trainloader, CLASSES)

    return trainloader, testloader


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
                optimizer, loss_func,
                n_epoch, train_loader,
                save_and_eval_interval=2000):
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (features, labels) in tqdm(enumerate(train_loader, 0)):
            features, labels = features.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(features)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % save_and_eval_interval == (save_and_eval_interval - 1):  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

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


def test_model(test_loader, model):
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


def main():
    trainloader, testloader = get_cifar_10(show=False)

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_model(model=net,
                optimizer=optimizer, loss_func=criterion,
                n_epoch=2, train_loader=trainloader,
                save_and_eval_interval=2000)

    test_model_toy(testloader, net)
    test_model(trainloader, net)
    test_model(testloader, net)


if __name__ == "__main__":
    main()
