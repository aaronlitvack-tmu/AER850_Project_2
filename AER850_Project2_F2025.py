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

#augmentation and normalization layers
rescale_layer = tf.keras.layers.Rescaling(1./255)
shear_layer = tf.keras.layers.RandomShear(0.1)
zoom_layer = tf.keras.layers.RandomTranslation(height_factor=(0.1), width_factor=(0.1))
flip_layer = tf.keras.layers.RandomFlip(mode="horizontal")
rotate_layer = tf.keras.layers.RandomRotation(0.1)
bright_layer = tf.keras.layers.Equalization()

normalized_1 = train_ds.map(lambda x, y: (rotate_layer(x), y))
normalized_2 = normalized_1.map(lambda x, y: (flip_layer(x), y))
normalized_3 = normalized_2.map(lambda x, y: (zoom_layer(x), y))
normalized_4 = normalized_3.map(lambda x, y: (bright_layer(x), y))
normalized_5 = normalized_4.map(lambda x, y: (shear_layer(x), y))
normalized_ds = normalized_5.map(lambda x, y: (rescale_layer(x), y))

val_normalized = val_ds.map(lambda x, y: (rescale_layer(x), y))

image_batchtrn, labels_batchtrn = next(iter(normalized_ds))


image_batchval, labels_batchval = next(iter(val_normalized))


#Neural Network Architecture Design

BATCH_SIZE = 32
EPOCHS = 30
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=7,
    restore_best_weights=True
    )

#model design
mdl1 = keras.Sequential([

    layers.Conv2D(16, (3,3), activation="relu", input_shape=(500, 500, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.AveragePooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')  
    
  
])

mdl1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

history = mdl1.fit(image_batchtrn, labels_batchtrn, epochs=EPOCHS,batch_size=BATCH_SIZE, 
                    validation_data=(image_batchval, labels_batchval), callbacks=[early_stop],
                    verbose=1)

mdl1.save("mdl1.keras")

#accuracy and loss figures
plt.figure(1)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label = 'validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


plt.figure(2)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.0, 6])
plt.legend(loc='lower right')
plt.title('Training and Validation Loss')

test_loss, test_acc = mdl1.evaluate(image_batchval,  labels_batchval, verbose=1)
