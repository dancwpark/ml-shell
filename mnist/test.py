import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from mnist import MNIST
from mnist import MNISTModel
import os

BATCH_SIZE=1

with tf.Session() as sess:
    restore_path = "model/mnist"
    data, model = MNIST(), MNISTModel(sess).get()
    
    

    x = tf.placeholder(tf.float32,
            (None, model.image_size, model.image_size, model.num_channels))
    y = model.predict(x)

    r = []
    for i in range(0, len(data.test_data), BATCH_SIZE):
        pred = sess.run(y, {x:data.test_data[i:i+BATCH_SIZE]})
        r.append(np.argmax(pred, 1) == np.argmax(data.test_labels[i:i+BATCH_SIZE],1))
        print(np.mean(r))
