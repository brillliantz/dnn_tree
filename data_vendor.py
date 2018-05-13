# encoding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataVendor:
    def __init__(self):
        self.x = None
        self.y = None
        self.input_shape_without_batch = (0, )
        # for regression, n_classes = 1
        self.n_classes = 0

        self.prepare_data()

    def prepare_data(self):
        pass

    def get_data(self, test_size=0.3, shuffle=True, random_state=369):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
                                                            test_size=test_size,
                                                            shuffle=shuffle, random_state=random_state)

        return x_train, x_test, y_train, y_test, self.input_shape_without_batch, self.n_classes


class DataWineQuality(DataVendor):
    def prepare_data(self):
        data = pd.read_csv('Data/wine_quality/winequality-white.csv', delimiter=';')

        X = data.iloc[:, :-1]
        Y = data.iloc[:, [-1]]
        X = X.values
        Y = Y.values

        import sklearn.preprocessing
        Y = sklearn.preprocessing.OneHotEncoder(n_values=10, sparse=False).fit_transform(Y)

        self.x = X
        self.y = Y
        n_cols = X.shape[1]
        self.input_shape_without_batch = (n_cols, )
        self.n_classes = 10


class DataMNIST_new(DataVendor):
    def get_data(self, test_size=0.3, shuffle=True, random_state=369):
        from sklearn.preprocessing import OneHotEncoder
        f = np.load('Data/MNIST/mnist.npz')
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        x_train = x_train / 256.0
        x_test = x_test / 256.0
        x_train = np.expand_dims(x_train, 3)
        x_test = np.expand_dims(x_test, 3)
        y_train = y_train.reshape([-1, 1])
        y_test = y_test.reshape([-1, 1])

        encoder = OneHotEncoder(n_values=10, sparse=False)
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.fit_transform(y_test)

        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        return ds_train, ds_test, self.input_shape_without_batch, self.n_classes

    def prepare_data(self):
        # If use cnn: Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
        self.input_shape_without_batch = (28, 28, 1)

        # If use dnn
        # self.input_shape_without_batch = (28 * 28, )

        n_labels = 10
        self.n_classes = n_labels


class DataMNIST(DataVendor):
    def get_data(self, test_size=0.3, shuffle=True, random_state=369):
        import tensorflow.examples.tutorials.mnist.input_data as input_data

        mnist = input_data.read_data_sets("Data/MNIST/", one_hot=True)
        trX, trY = mnist.train.images, mnist.train.labels
        teX, teY = mnist.test.images, mnist.test.labels

        return trX, teX, trY, teY, self.input_shape_without_batch, self.n_classes

    def prepare_data(self):
        # If use cnn: Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
        self.input_shape_without_batch = (28, 28, 1)

        # If use dnn
        # self.input_shape_without_batch = (28 * 28, )

        n_labels = 10
        self.n_classes = n_labels


class DataFutureTick(DataVendor):
    def prepare_data(self):
        from tick_data_vendor import load_data
        self.x, self.y = load_data()

        if False:
            n = len(self.x)
            n_cut = n // 3
            self.x = self.x[: n_cut]
            self.y = self.y[: n_cut]

        n_cols = self.x.shape[1]

        self.input_shape_without_batch = (n_cols, )
        n_ticks = 16
        self.n_classes = n_ticks
