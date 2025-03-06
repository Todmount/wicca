import cv2
import numpy as np

def load_image(file_path):
    """
    Load an image from the specified file path using OpenCV.

    Parameters:
        file_path (str): The path to the image file.

    Returns:
        image (numpy.ndarray): The loaded image, or None if the file couldn't be read.
    """
    image = cv2.imread(file_path)

    if image is None:
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_padded_copy(image, ratio, border_type=cv2.BORDER_REPLICATE, border_constant=0):
    """
    Returns a padded copy of the image with height & width that are multiples of ratio

    Parameters:
        image (numpy.ndarray): The input image.
        ratio (int): The input ratio.
        border_type (int): The padding type.
        border_constant (int): Fill value.

    Returns:
        numpy.ndarray: Padded image
    """
    rows, cols, channels = image.shape

    quotient, remainder = divmod(rows, ratio)
    rows_to_add = 0 if remainder == 0 else (quotient + 1) * ratio - rows

    quotient, remainder = divmod(cols, ratio)
    cols_to_add = 0 if remainder == 0 else (quotient + 1) * ratio - cols

    if rows_to_add == 0 and cols_to_add == 0:
        return image

    return cv2.copyMakeBorder(image, 0, rows_to_add, 0, cols_to_add, border_type, None,
                              [border_constant, border_constant, border_constant])
