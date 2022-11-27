import os
from PIL import Image
from pprint import pprint

def create_folder(dest_path):
    try:
        os.mkdir(dest_path)
    except FileExistsError:
        pass

def get_list_image_paths(images_directory):
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
    list_of_heights = []
    with Image.open(image_path) as im:
        width, height = im.size
        list_of_heights.append(height)
        return list_of_heights


def get_smallest_height(all_images_paths):
    image_with_smallest_height = min(all_images_paths, key=get_images_heights)
    with Image.open(image_with_smallest_height) as im:
        new_height = im.size[1]
        return new_height

def get_new_width(new_height, image_path):        
    with Image.open(image_path) as im:
        width, height = im.size
        new_width  = int(new_height * width / height)
        return new_width

def resize_images(images_directory, dest_path):
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
        
    