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
class_names = test_ds.class_names

img1_array = tf.keras.utils.img_to_array(img1)
img1_array = tf.expand_dims(img1_array, 0)

data_augmentation = keras.Sequential(
  [
   
    layers.RandomFlip("horizontal", input_shape=(500, 500, 3)),
   
    layers.RandomRotation(0.1),

    layers.RandomZoom(0.1),
    
    layers.Rescaling(1./255, input_shape=(500, 500, 3)),

    
  ]
)

mdl1 = keras.Sequential([
    
    data_augmentation,
    
    layers.Conv2D(16, (3,3), activation="relu", input_shape=(500, 500, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(3, activation='softmax')    
])

predictions = mdl1.predict(img1_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)