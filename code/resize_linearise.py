from PIL import Image
import numpy as np
import os

def resize_and_linearize_images(image_paths, size=(72, 72)):
    """
    Resize images to the specified size and linearize them (flatten to 1D array).
    
    Parameters:
    - image_paths (list): List of paths to the images.
    - size (tuple): Target size to resize the images (default: 72x72).
    
    Returns:
    - resized_images (list): List of resized and linearized images.
    """
    resized_images = []
    
    for image_path in image_paths:
        # Open image
        image = Image.open(image_path)
        
        # Resize image to 72x72
        image_resized = image.resize(size)
        
        # Convert image to numpy array and flatten it
        image_array = np.array(image_resized)
        
        # Flatten the image (convert it into a 1D vector)
        linearized_image = image_array.flatten()
        
        resized_images.append(linearized_image)
    
    print('Done')
    
    return resized_images