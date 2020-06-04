import tensorflow as tf
import numpy as np
import os

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

import keras
from keras import backend as K

class MNIST:
    def __init__(self):
        VALIDATION_SIZE = 5000
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
            x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
        else:
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
 
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        self.validation_data = x_train[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = y_train[:VALIDATION_SIZE]
        self.train_data = x_train[VALIDATION_SIZE:, :, :, :]
        self.train_labels = y_train[VALIDATION_SIZE:]
        self.test_data = x_test
        self.test_labels = y_test

class MNISTModel:
    def __init__(self, session=None):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()

        model.add(Conv2D(32, (3,3), input_shape=(28,28,1)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64, (3,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3,3)))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(10, activation='softmax'))

        self.model = model
        self.sess = session

    def get(self):
        return self.model

    def predict(self, data):
        return self.model(data)
