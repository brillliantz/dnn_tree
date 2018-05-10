# encoding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split


class DataVendor:
    def __init__(self):
        self.x = None
        self.y = None
        self.input_shape_without_batch = (0, )

        self.prepare_data()

    def prepare_data(self):
        pass

    def get_data(self, test_size=0.3, shuffle=True, random_state=369):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
                                                            test_size=test_size,
                                                            shuffle=shuffle, random_state=random_state)

        return x_train, x_test, y_train, y_test, self.input_shape_without_batch


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


class DataMNIST(DataVendor):
    def get_data(self, test_size=0.3, shuffle=True, random_state=369):
        import tensorflow.examples.tutorials.mnist.input_data as input_data

        mnist = input_data.read_data_sets("Data/MNIST/", one_hot=True)
        trX, trY = mnist.train.images, mnist.train.labels
        teX, teY = mnist.test.images, mnist.test.labels

        return trX, teX, trY, teY, self.input_shape_without_batch

    def prepare_data(self):
        # If use cnn: Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
        self.input_shape_without_batch = (28, 28, 1)

        # If use dnn
        # self.input_shape_without_batch = (28 * 28, )


class DataFutureTick(DataVendor):
    def prepare_data(self):
        from tick_data_vendor import load_data
        self.x, self.y = load_data()

        n_cols = self.x.shape[1]

        self.input_shape_without_batch = (n_cols, )
