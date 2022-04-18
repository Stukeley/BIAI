# BIAI
Projekt BIAI (Biologically Inspired Artificial Intelligence), semestr 6.

# Założenia projektu
- A program that distincts between different dog and cat breeds
- Written in Python using TensorFlow
- Trained on a dataset of nearly 7400 pictures split into 37 different dog or cat breeds
- Main idea: load the images, create an input pipeline, split input images into two sets (training and validation), train the model, evaluate its performance (using the validation dataset), if needed adjust model parameters
- User input is an image displaying a cat or a dog
- Program output is the name of the breed (race) of the animal in the input picture
- A simple GUI to simplify the process of choosing a picture, and previewing its contents and label
- Use of external libraries to simplify the process of loading data (“Keras”), training the model (TensorFlow), visualizing its performance (“matplotlib”) and making adjustments such as using more/less images for validation
- Extensive testing (documented with graphs and data) of the model and making changes until reaching satisfactory results
