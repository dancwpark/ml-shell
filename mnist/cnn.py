"""
Shell code for a convolutional neural network
for the MNIST dataset (or F-MNIST) if preferred.
"""

from __future__ import print_function

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Some hardcoded variables
batch_size = 256
nb_classes = 10
epochs = 10

# Input dimensions
img_r, img_c = 28, 28

# Load the fashion_mnist dataset from keras
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_r, img_c)
    x_test = x_test.reshape(x_test.shape[0], 1, img_r, img_c)
    input_shape = (1, img_r, img_c)
else:
    x_train = x_train.reshape(x_train.shape[0], img_r, img_c, 1)
    x_test = x_test.reshape(x_test.shape[0], img_r, img_c, 1)
    input_shape = (img_r, img_c, 1)

x_train = x_train/255.
x_test = x_test/255.

# Change y to binary class matrices
y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
    activation='relu',
    input_shape=input_shape))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
        #optimizer=keras.optimizers.Adadelta(),
        optimizer='adam',
        metrics=['accuracy'])

model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)) # This is cheating!

model.save("model/fmnist")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

