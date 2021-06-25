import tensorflow as tf
import numpy as np

from model import MNIST
from model import MNISTModel

save_path = "models\mnist"

data, model = MNIST(), MNISTModel().get()
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# Train
model.fit(data.train_data, data.train_labels,
          batch_size=256,
          epochs=1,
          verbose=1,
          validation_data=(data.validation_data,
                           data.validation_labels))

model.save(save_path)
score = model.evaluate(data.test_data, data.test_labels, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])