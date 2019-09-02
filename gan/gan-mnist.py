from __future__ import print_function import tensorflow as tf

import numpy as numpy
import keras
import os
import argparse

import keras
from keras.datasets import fashion_mnist, mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K



# DEBUGGING

# '0' DEBUG   | All messsages shown
# '1' INFO    | Filter out INFO messages
# '2' WARNING | Filter out INFO and WARNING messages
# '3' ERROR   | Filter out all messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define discriminator network
def discriminator(data, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        ## First conv and pooling layers
        ## Find 32 diff 5x5 pixel features
        