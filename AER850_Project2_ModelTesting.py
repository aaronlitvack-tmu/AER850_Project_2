import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_ds = tf.keras.utils.image_dataset_from_directory(
  "Data/test",
  image_size=(500, 500),
  )


img1 = tf.keras.utils.load_img(
    "Data/test/crack/test_crack.jpg", target_size=(500, 500)
)
img2 = tf.keras.utils.load_img(
    "Data/test/missing-head/test_missinghead.jpg", target_size=(500, 500)
)
img3 = tf.keras.utils.load_img(
    "Data/test/paint-off/test_paintoff.jpg", target_size=(500, 500)
)

class_names = test_ds.class_names

img1_array = tf.keras.utils.img_to_array(img1)
img1_array = tf.expand_dims(img1_array, 0)
img1_array = img1_array/255.0


model =  keras.models.load_model("mdl1.keras")


predictions = model.predict(img1_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

img2_array = tf.keras.utils.img_to_array(img2)
img2_array = tf.expand_dims(img2_array, 0)
img2_array = img2_array/255.0
predictions = model.predict(img2_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

img3_array = tf.keras.utils.img_to_array(img3)
img3_array = tf.expand_dims(img3_array, 0)
img3_array = img3_array/255.0
predictions = model.predict(img3_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)