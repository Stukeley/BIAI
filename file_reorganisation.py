import os


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
