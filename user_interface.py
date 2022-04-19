import PySimpleGUI as sg
from PIL import ImageTk


# Function that creates a window and returns it.
# Makes use of the PySimpleGUI library.
def create_window():
    sg.theme('Dark Blue 3')

    layout = create_layout()
    window = sg.Window('BIAI', layout, size=(600, 400), element_justification='center')
    return window


# Function that creates a layout for the window.
# Returns a layout later used by create_window().
def create_layout():
    layout = [
        [sg.Text('BIAI, semestr 6 - Rafał Klinowski, Jakub Cisowski')],
        [sg.Text('Choose a file to open:'), sg.Input(), sg.FileBrowse(key="-IN-")],
        [sg.Button('Predict')],
        [],
        [sg.Text('Prediction:', key='-PREDICTION-')],
        [],
        [sg.Image(size=(200, 200), key="-IMAGE-")],
    ]
    return layout


# Function used to update the image in the window after loading it from a file.
def update_image(window, image):
    image_converted = ImageTk.PhotoImage(image=image)
    window['-IMAGE-'].update(data=image_converted)


# Function used to update the prediction in the window after making a prediction.
def update_prediction(window, prediction, score):
    window['-PREDICTION-'].update("Prediction: " + prediction + " " + str(round(score)) + "%")
