import os
from fnmatch import fnmatch

from PIL import Image

# Helper function that groups all the images into separate folders based on the cat/dog breed name
# Example: By default, all the images are in the /images folder.
# After calling this function, all images of Abyssinian cat will be in the /images/Abyssinian folder
# all images of American Shorthair cat will be in the /images/American_Shorthair folder
# etc.
def create_folders_and_move_image_files():

    images_dir = os.path.dirname(os.path.realpath(__file__)) + '/images'

    # Get all file names in the images' directory as a list
    # Only list the files that end with .jpg
    file_list = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    # Group files that start with the same word
    for file_name in file_list:
        # Split the file name based on the last _ character
        # Example: split = "Abyssinian_cat"
        split = file_name.split("_")
        split.pop()

        # Join split list into a string
        split_name = "_".join(split)

        # If folder with the same name doesn't exist, create it
        if not os.path.exists(images_dir + "/" + split_name):
            os.makedirs(images_dir + "/" + split_name)

        # Move the file to the new folder
        os.rename(images_dir + "/" + file_name, images_dir + "/" + split_name + "/" + file_name)

# Helper function that calculates the minimum, maximum and average size of files in the images' directory
# The results will be used to resize the images to a uniform size
def calculate_min_max_avg_image_size():
    
    # Get all files from the images directory as well as the subdirectories
    # Only list the files that end with .jpg
    file_list = []

    for (dirpath, dirnames, filenames) in os.walk(os.path.dirname(os.path.realpath(__file__)) + '/images'):
        file_list += [os.path.join(dirpath, file) for file in filenames if file.endswith('.jpg')]
    
    max_width = 0
    max_height = 0
    min_width = 0
    min_height = 0
    sum_width = 0
    sum_height = 0
    count = 0
    
    # Get info of each file from file_list
    for file_name in file_list:
        img = Image.open(file_name)
        width, height = img.size
        
        # Update max and min width and height
        if width > max_width:
            max_width = width
        if width < min_width:
            min_width = width
        if height > max_height:
            max_height = height
        if height < min_height:
            min_height = height
            
        # Update sum of width and height
        sum_width += width
        sum_height += height
        count += 1
        
    # Calculate average width and height
    avg_width = sum_width / count
    avg_height = sum_height / count
    
    print("Min width: " + str(min_width) + ", Max width: " + str(max_width) + ", Avg width: " + str(avg_width))
    print("Min height: " + str(min_height) + ", Max height: " + str(max_height) + ", Avg height: " + str(avg_height))
    
    # Results:
    # max_width = 3264
    # max_height = 2606
    # min_width = 0
    # min_height = 0
    # avg_width = 437
    # avg_height = 391
