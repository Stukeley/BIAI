import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

import pathlib

from tensorflow import keras
from keras import layers
from keras.models import Sequential

from file_reorganisation import *

create_folders_and_move_image_files()

# Load the data from the subfolders at /images/
data_dir = pathlib.Path("images")

# Verify image count
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)  # Should be 7390

# Show sample images
abyssinians = list(data_dir.glob('Abyssinian/*.jpg'))
# PIL.Image.open(str(abyssinians[1])).show()

# Around 200 images in each set, split them into 32-image batches
batch_size = 32

# Create a dataset
# Use 20% of images for validation, and 80% for training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)  # ['Abyssinian', 'Bengal', 'Birman', ...]

# Configure the dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardize the data
normalization_layer = layers.Rescaling(1./255)

# RGB values will be in [0, 1] range now
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, label_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))  # Prints 0.0 0.9982423

# Create the model
num_classes = len(class_names)

# TODO: ogarnąć to tutaj
# Also problem: obrazki mają różne rozmiary -> najlepiej chyba je znormalizować jakoś programowo, bo inaczej są dymy (None; GlobalAveragePooling2D)
# Also: dodać obrazki żeby było wszędzie po 200 obrazków
# https://stackoverflow.com/a/47796091/17816815

# Data augumentation - TODO

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(None, None, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(None, None, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.build()

# View the model summary
model.summary()

# Train the model
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Create plots showing loss and accuracy of the training/validation sets

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Todo: save the model to a file (for now it has to be re-built every time)