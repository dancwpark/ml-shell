import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from mnist import MNIST
from mnist import MNISTModel
import os

save_path = "models/mnist"

data, model = MNIST(), MNISTModel().model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(data.train_data, data.train_labels,
          batch_size=256,
          epochs=1,
          verbose=1,
          validation_data=(data.validation_data, data.validation_labels))

model.save("models/mnist")
score = model.evaluate(data.test_data, data.test_labels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])