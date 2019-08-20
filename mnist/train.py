import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from mnist import MNIST
from mnist import MNISTModel
import os


with tf.Session() as sess:
    data, model = MNIST(), MNISTModel("models/mnist", sess)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer='adam'
            metrics=['accuracy'])

    model.fit(data.training_data, data.training_labels,
            batch_size=256,
            epochs=1,
            verbose=1,
            validation_data=(validation_data, validation_labels))

    model.save("model/mnist")
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
