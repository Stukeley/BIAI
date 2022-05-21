import pathlib

import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras

from file_reorganisation import *
from user_interface import *


def main():
    model = None
    class_names = None

    # If model already exists, load it.
    if os.path.exists('model'):
        model = load_model()
        class_names = load_class_names()
    # Otherwise, create a new model, train it and save it to a folder.
    else:
        # If model doesn't exist, the other file should be run.
        print("Model doesn't exist. Please run train_only.py first.")

    # Create and display User Interface
    window = create_window()
    while True:
        event, values = window.Read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            window.close()
            break
        elif event == 'Predict':
            # Update the UI
            image_path = values["-IN-"]
            print(image_path)
            image = PIL.Image.open(image_path)
            image = image.resize((200, 200), resample=Image.BICUBIC)
            update_image(window, image)

            # Predict the image
            (class_name, score) = predict_image(model, image_path, class_names)

            # Update the UI
            update_prediction(window, class_name, score)


# Function used to load model from disk if it exists.
def load_model():
    loaded_model = keras.models.load_model('model')
    return loaded_model


# Function used to predict class of an image.
def predict_image(model, image_path, class_names):
    img = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("Prediction: " + class_names[np.argmax(score)], " with score: " + str(100 * np.max(score)))

    return (class_names[np.argmax(score)], 100 * np.max(score))


# Function used to load class names based on directories present on disk.
def load_class_names():
    data_dir = pathlib.Path("images")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir)
    class_names = train_ds.class_names

    return class_names


# Code required for the program to run.
if __name__ == "__main__":
    main()
