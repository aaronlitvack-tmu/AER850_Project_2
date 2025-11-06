import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#load data
test_ds = tf.keras.utils.image_dataset_from_directory(
  "Data/test",
  image_size=(500, 500),
  )

#load images
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

#loads model
model =  keras.models.load_model("mdl1.keras")


#image 1 array normalization
img1_array = tf.keras.utils.img_to_array(img1)
img1_array = tf.expand_dims(img1_array, 0)
img1_array = img1_array/255.0

#image 1 prediction
predictions1 = model.predict(img1_array)
score1 = tf.nn.softmax(predictions1[0])


#image 1 plot
plt.figure(1)
plt.imshow(img1)
plt.axis('off')
plt.title("True Crack Classification Label: crack\nPredicted Crack Classification Label: " + class_names[np.argmax(score1)])
plt.text(15, 40, s = "Crack: " + str(100*np.round(np.max(score1[0]), 4)) + "%", size = 'x-large', c = 'green')
plt.text(15, 70, s = "Missing Head: " + str(100*np.round(np.max(score1[1]), 4)) + "%", size = 'x-large', c = 'green')
plt.text(15, 100, s = "Paint Off: " + str(100*np.round(np.max(score1[2]), 5)) + "%", size = 'x-large', c = 'green')

#image 2 array normalization
img2_array = tf.keras.utils.img_to_array(img2)
img2_array = tf.expand_dims(img2_array, 0)
img2_array = img2_array/255.0

#image 2 prediction
predictions2 = model.predict(img2_array)
score2 = tf.nn.softmax(predictions2[0])

#image 2 plot
plt.figure(2)
plt.imshow(img2)
plt.axis('off')
plt.title("True Crack Classification Label: missing-head\nPredicted Crack Classification Label: " + class_names[np.argmax(score2)])
plt.text(15, 40, s = "Crack: " + str(100*np.round(np.max(score2[0]), 4)) + "%", size = 'x-large', c = 'green')
plt.text(15, 70, s = "Missing Head: " + str(100*np.round(np.max(score2[1]), 4)) + "%", size = 'x-large', c = 'green')
plt.text(15, 100, s = "Paint Off: " + str(100*np.round(np.max(score2[2]), 5)) + "%", size = 'x-large', c = 'green')

#image 3 array normalization
img3_array = tf.keras.utils.img_to_array(img3)
img3_array = tf.expand_dims(img3_array, 0)
img3_array = img3_array/255.0

#image 3 prediction
predictions3 = model.predict(img3_array)
score3 = tf.nn.softmax(predictions3[0])

#image 3 plot
plt.figure(3)
plt.imshow(img3)
plt.axis('off')
plt.title("True Crack Classification Label: paint-off\nPredicted Crack Classification Label: " + class_names[np.argmax(score3)])
plt.text(15, 40, s = "Crack: " + str(100*np.round(np.max(score3[0]), 4)) + "%", size = 'x-large', c = 'green')
plt.text(15, 70, s = "Missing Head: " + str(100*np.round(np.max(score3[1]), 5)) + "%", size = 'x-large', c = 'green')
plt.text(15, 100, s = "Paint Off: " + str(100*np.round(np.max(score3[2]), 4)) + "%", size = 'x-large', c = 'green')
