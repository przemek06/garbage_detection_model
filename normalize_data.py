import numpy as np

def standardize_image(image):
    image = image.astype(np.float16)
    mean = image.mean()
    std = image.std()
    standardized_image = (image - mean) / std
    return standardized_image
