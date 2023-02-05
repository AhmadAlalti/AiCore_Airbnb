import os
from PIL import Image
from pprint import pprint



def create_folder(dest_path):
    
    '''If the folder doesn't exist, create it
    
    Parameters
    ----------
    dest_path
        The path to the folder where you want to save the images.
    '''
    
    try:
        os.mkdir(dest_path)
    except FileExistsError:
        pass



def get_list_image_paths(images_directory):
    
    '''It takes a directory as an argument, and returns a list of all the paths to the images in that
    directory
    
    Parameters
    ----------
    images_directory
        The directory where the images are stored.
    
    Returns
    -------
        A list of all the paths to the images in the images directory.
    '''
    
    all_images_paths = []
    images_directories = os.listdir(images_directory)
    
    for directory in images_directories:
        
        directory_relaitve_path = f"{images_directory}/{directory}"
        files = os.listdir(directory_relaitve_path)
        
        for file in files:
            
            if file.endswith(".png"):
                all_images_paths.append(f"{images_directory}/{directory}/{file}")
            else:
                 pass
             
    return all_images_paths



def get_images_heights(image_path):
    
    '''It opens the image, gets the width and height, and appends the height to a list
    
    Parameters
    ----------
    image_path
        The path to the image you want to get the height of.
    
    Returns
    -------
        A list of the heights of the images in the directory.
    
    '''
    list_of_heights = []
    
    with Image.open(image_path) as im:
        width, height = im.size
        list_of_heights.append(height)
        
        return list_of_heights



def get_smallest_height(all_images_paths):
    
    '''It takes a list of image paths, and returns the height of the smallest image in the list
    
    Parameters
    ----------
    all_images_paths
        a list of all the images in the directory
    
    Returns
    -------
        The height of the smallest image in the list of images.
    '''
    
    image_with_smallest_height = min(all_images_paths, key=get_images_heights)
    
    with Image.open(image_with_smallest_height) as im:
        new_height = im.size[1]
        
        return new_height

def get_new_width(new_height, image_path):        
    
    ''' Given a new height and an image path, return the new width
    
    Parameters
    ----------
    new_height
        The height of the image you want to resize to.
    image_path
        The path to the image you want to resize.
    
    Returns
    -------
        The new width of the image.
    '''
    
    with Image.open(image_path) as im:
        width, height = im.size
        new_width  = int(new_height * width / height)
        
        return new_width

def resize_images(images_directory, dest_path):
    
    '''It takes a directory of images, finds the smallest height of all the images, and resizes all the
    images to that height
    
    Parameters
    ----------
    images_directory
        The directory where the images are stored.
    dest_path
        The path to the directory where you want to save the resized images.
    '''
    
    all_images_paths = get_list_image_paths(images_directory)
    new_height = get_smallest_height(all_images_paths)
    
    for image in all_images_paths:
        im = Image.open(image)
        
        if im.mode == "RGB":
            new_width = get_new_width(new_height, image)
            im = im.resize((new_width, new_height))
            new_path = os.path.normpath(image)
            new_path = new_path.split(os.sep)[-1]
            print(new_path)
            im.save(f"{dest_path}/{new_path}")
        else:
            pass

    

if __name__ == "__main__":
    create_folder("airbnb-property-listings/processed_images")
    resize_images("airbnb-property-listings/images", "airbnb-property-listings/processed_images")
        
    