import cv2
import numpy as np
from typing import Optional


def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Load an image from the specified file path using OpenCV.

    Parameters:
        file_path (str): The path to the image file.

    Returns:
        Optional[numpy.ndarray]: The loaded image in RGB format, or None if loading fails.

    Raises:
        ValueError: If file_path is empty or None
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    try:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image: {file_path}")
            return None

        # Convert only if image is not grayscale
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    except Exception as e:
        print(f"Error loading image {file_path}: {str(e)}")
        return None


def get_padded_copy(image: np.ndarray, ratio: int, border_type: int = cv2.BORDER_REPLICATE,
                    border_constant: int = 0) -> np.ndarray:
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
    # rows, cols, channels = image.shape
    #
    # quotient, remainder = divmod(rows, ratio)
    # rows_to_add = 0 if remainder == 0 else (quotient + 1) * ratio - rows
    #
    # quotient, remainder = divmod(cols, ratio)
    # cols_to_add = 0 if remainder == 0 else (quotient + 1) * ratio - cols
    #
    # if rows_to_add == 0 and cols_to_add == 0:
    #     return image
    #
    # return cv2.copyMakeBorder(image, 0, rows_to_add, 0, cols_to_add, border_type, None,
    #                           [border_constant, border_constant, border_constant])


    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if ratio <= 0:
        raise ValueError("Ratio must be positive")

    # Handle both 2D and 3D images
    if len(image.shape) == 2:
        rows, cols = image.shape
        channels = 1
    elif len(image.shape) == 3:
        rows, cols, channels = image.shape
    else:
        raise ValueError("Image must be 2D or 3D array")

    quotient, remainder = divmod(rows, ratio)
    rows_to_add = 0 if remainder == 0 else (quotient + 1) * ratio - rows
    quotient, remainder = divmod(cols, ratio)
    cols_to_add = 0 if remainder == 0 else (quotient + 1) * ratio - cols

    if rows_to_add == 0 and cols_to_add == 0:
        return image

    border_value = border_constant if channels == 1 else [border_constant] * channels
    return cv2.copyMakeBorder(image, 0, rows_to_add, 0, cols_to_add, border_type,
                              None, border_value)
