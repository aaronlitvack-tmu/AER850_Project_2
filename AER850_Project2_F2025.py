import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
keras.utils.set_random_seed(42)


#Data Processing
img_height = 500
img_width = 500
img_channel = 3

train_dir = "Data/train"
valid_dir = "Data/valid"



train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  image_size=(img_height, img_width),
  )

val_ds = tf.keras.utils.image_dataset_from_directory(
  valid_dir,
  image_size=(img_height, img_width),
  )


rescale_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (rescale_layer(x), y))
image_batchtrn, labels_batchtrn = next(iter(train_ds))


val_ds = val_ds.map(lambda x, y: (rescale_layer(x), y))
image_batchval, labels_batchval = next(iter(val_ds))


#Neural Network Architecture Design

BATCH_SIZE = 32 
EPOCHS = 15
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
    )

mdl1 = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="leaky_relu", input_shape=(500, 500, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="leaky_relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation="leaky_relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')    
])

mdl1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = mdl1.fit(image_batchtrn, labels_batchtrn, epochs=EPOCHS, 
                    validation_data=(image_batchval, labels_batchval))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = mdl1.evaluate(image_batchval,  labels_batchval, verbose=2)