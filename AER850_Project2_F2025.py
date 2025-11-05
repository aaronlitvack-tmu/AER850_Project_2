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
flip_layer = tf.keras.layers.RandomFlip(input_shape=(img_height, img_width, 3))
rotate_layer = tf.keras.layers.RandomRotation(factor = (-0.2, 0.3))
zoom_layer = tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.2))



image_batchtrn, labels_batchtrn = next(iter(train_ds))

#image_batchtrn = (image_batchtrn / 255.0).astype("float32")

image_batchval, labels_batchval = next(iter(val_ds))

#image_batchtval = (image_batchval / 255.0).astype("float32")


data_augmentation = keras.Sequential(
  [
   
    #layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
   
    layers.RandomRotation(0.1),

    #layers.RandomShear(0.1),

    layers.RandomZoom(0.1),
    
    
  ]
)
#Neural Network Architecture Design

BATCH_SIZE = 32
EPOCHS = 33
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
    )

mdl1 = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(500, 500, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(3, activation='softmax')    
])

mdl1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

history = mdl1.fit(image_batchtrn, labels_batchtrn, epochs=EPOCHS,batch_size=BATCH_SIZE, 
                    validation_data=(image_batchval, labels_batchval), callbacks=[early_stop],
                    verbose=1)
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
plt.ylabel('Accuracy')
plt.ylim([0.0, 5])
plt.legend(loc='lower right')
plt.title('Training and Validation Loss')

test_loss, test_acc = mdl1.evaluate(image_batchval,  labels_batchval, verbose=1)

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
predictions = mdl1.predict(img1_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)