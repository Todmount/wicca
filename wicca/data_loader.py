# WICCA: Wavelet-based Image Compression and Classification Analysis
# Copyright (C) 2025  Andrii Lesniak
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import sys

import cv2
import numpy as np
from typing import Any

from tqdm.auto import tqdm

from wicca.validation import validate_image
from wicca.config.constants import MODEL, PRE_INP, DEC_PRED, SHAPE
from wicca.config.aliases import ModelsDict


def load_image(file_path: str) -> np.ndarray | None:
    """
    Loads an image from a file path and converts it to RGB format if it is a color image.

    Reads an image file and attempts to convert it to RGB format when it is in color.
    Returns the loaded image or None in case of failure. Raises a ValueError if the provided
    file path is empty.

    Args:
        file_path (str): Path to the image file to be loaded.

    Returns:
        Optional[np.ndarray]: The loaded image as a NumPy array, or None if the image could not
        be loaded.

    Raises:
        ValueError: If the file path is empty.
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    try:
        image = cv2.imread(file_path)
        validate_image(image)

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
    Pads an image to make its dimensions divisible by a specified ratio. The function
    handles 2D (grayscale) and 3D (color) input images and allows different border
    types and constant values for padding.

    Args:
        image (np.ndarray): Input image as a 2D or 3D numpy array.
        ratio (int): Positive integer defining the divisibility constraint for the
            image dimensions.
        border_type (int, optional): Type of border for padding, as defined by
            OpenCV constants. Defaults to `cv2.BORDER_REPLICATE`.
        border_constant (int, optional): Constant pixel value for padding if
            `border_type` is `cv2.BORDER_CONSTANT`. Defaults to 0.

    Returns:
        np.ndarray: The padded image where dimensions are divisible by the given
        ratio.

    Raises:
        ValueError: If `image` is not a numpy array.
        ValueError: If `ratio` is not a positive integer.
        ValueError: If `image` does not have 2D or 3D dimensions.
    """


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


def load_single_model(model_class,
                      shape: tuple[int, int] = (224, 224),
                      weights: str = 'imagenet'
                      ) -> dict | None:
    """
    Load a classifier model with error handling.

    Args:
        model_class: Model class to instantiate
        shape: Input shape tuple (height, width)
        weights: Pre-trained weights to use, defaults to 'imagenet'

    Returns:
        Dictionary containing model and its associated functions, or None if loading fails
    """
    try:
        # Get the module containing the model function
        module = sys.modules[model_class.__module__]

        return {
            MODEL: model_class(weights=weights),
            PRE_INP: getattr(module, 'preprocess_input'),
            DEC_PRED: getattr(module, 'decode_predictions'),
            SHAPE: shape
        }
    except Exception as e:
        logging.error(f"Error loading: {str(e)}")
        return None


def load_models(models: ModelsDict) -> dict[str, Any]:
    """
    Load multiple image classification models with progress tracking.

    Args:
        models: Dictionary mapping model names to either:
               - a model class, or
               - a tuple of (model_class, config_dict)

    Returns:
        Dictionary mapping model names to loaded classifier instances
    """
    classifiers_dict = {}

    with tqdm(models.items(), desc="Loading classifiers", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]') as pbar:
        for name, model_info in pbar:
            # Check if this is a tuple (model with config) or just a model class
            if isinstance(model_info, tuple):
                model_class, kwargs = model_info
            else:
                model_class = model_info
                kwargs = {}

            classifiers_dict[name] = load_single_model(model_class, **kwargs)

        return classifiers_dict
