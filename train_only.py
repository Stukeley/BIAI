import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from tensorflow import keras

from file_reorganisation import *


def main():
    # Sort files and get minimum, maximum and average file sizes.
    # These functions should only be called once.
    create_folders_and_move_image_files()
    calculate_min_max_avg_image_size()

    (model, class_names) = create_and_train_model()


# Function used to create and train a model.
# Only called if model doesn't already exist.
def create_and_train_model():
    # Load the data from the subfolders at /images/
    data_dir = pathlib.Path("images")

    # Verify image count
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)  # Should be 7390

    # Show sample images
    abyssinians = list(data_dir.glob('Abyssinian/*.jpg'))
    # PIL.Image.open(str(abyssinians[1])).show()

    # Around 200 images in each set, split them into 16-image batches
    batch_size = 16

    # Target size - all images will be resized to this size
    target_size = (160, 160)
    input_shape = (target_size[0], target_size[1], 3)

    # Create a dataset
    # Use 20% of images for validation, and 80% for training
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=6798,
        image_size=target_size,
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=6798,
        image_size=target_size,
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)  # ['Abyssinian', 'Bengal', 'Birman', ...]

    # Configure the dataset for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    # Data augumentation - to increase validation accuracy
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=input_shape),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # Create the model
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=input_shape),    # Normalize colors
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Dropout(0.2),    # Dropout 20% of the nodes to increase validation accuracy
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.summary()

    # Compile the model
    model.compile(optimizer='nadam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.build()

    # Train the model
    epochs = 400
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

    plt.savefig('model-size160-epochs400-batch16-nadam-validation20-dropout02x1.png')
    plt.savefig('model-size160-epochs400-batch16-nadam-validation20-dropout02x1.pdf')

    # Save the model to a file
    model.save('model-size160-epochs400-batch16-nadam-validation20-dropout02x1')

    return (model, class_names)


# Code required for the program to run.
if __name__ == "__main__":
    main()
