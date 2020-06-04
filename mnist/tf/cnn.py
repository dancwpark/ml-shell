"""
Shell code for a convolutional neural network
for the MNIST dataset (or F-MNIST) if preferred.
"""

from __future__ import print_function
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


# COMMAND LINE ARGUMENT PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--train", 
        default=None, 
        help="flag for train vs test")
parser.add_argument("--load_weights", 
        default=None,
        help="file location of saved weights. None if training from scratch")
parser.add_argument("-s", "--save", 
        default=None, 
        help="file location to save model weights")
args = vars(parser.parse_args())




restore = args['load_weights']
# restore = 'model/fmnist'
save_path = args['save']
if save_path == None:
    save_path = restore




# Some hardcoded variables
batch_size = 256
nb_classes = 10
epochs = 10
# Input dimensions
img_r, img_c = 28, 28





# Load the fashion_mnist dataset from keras
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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




# DEFINE MODEL
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

if restore != None:
    print("\n>> LOADING WEIGHTS FROM : ", restore, "\n")
    model.load_weights(restore)

model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'])



# TRAIN
model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)) # This is cheating!

# SAVE
if save_path != None:
    model.save(save_path)


# TEST
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

