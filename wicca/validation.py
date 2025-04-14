import sys
import logging
import numpy as np
from pathlib import Path

from wicca.normalization import normalize_folder


def _handle_folder_errors(folder: str | Path, ftype: str = 'data') -> Path:
    """Handles folder-related errors"""
    folder = normalize_folder(folder)
    if not folder.exists() and ftype == 'data':
        msg = f"Provided {ftype} folder: '{folder}' does not exist."
        logging.error(msg)
        raise FileNotFoundError(msg)
    elif not folder.exists():
        msg = f"Provided {ftype} folder: '{folder}' does not exist.\nCreating folder..."
        logging.warning(msg)
        folder.mkdir(parents=True, exist_ok=True)
    if not folder.is_dir():
        msg = f"Provided {ftype} folder: '{folder}' is not a directory."
        logging.error(msg)
        raise NotADirectoryError(msg)
    try:
        # Test access permissions by listing contents
        next(folder.iterdir(), None)
    except PermissionError:
        msg = f"Provided {ftype} folder: '{folder}' is not accessible."
        logging.error(msg)
        raise PermissionError(msg)

    return folder


def validate_input_folder(folder: str | Path, ftype: str = 'data') -> Path | None:
    """Validates a data folder path"""
    folder = _handle_folder_errors(folder, ftype)

    # Check if folder is empty
    if not any(folder.iterdir()):
        msg = f"The folder '{folder}' is empty. Please provide a non-empty folder. \nExiting... \n"
        logging.error(msg)
        # raise ValueError(msg)
        sys.exit(1)

    return folder


def validate_output_folder(folder: str | Path, ftype: str = 'result') -> Path | None:
    """Validates results folder path"""
    folder = _handle_folder_errors(folder, ftype)

    # Check if folder is not empty and prompt user
    if any(folder.iterdir()):
        user_input = input(
            f"Warning: The folder '{folder}' is not empty. Some of the files may be overwritten. \nContinue? (Y/n): ").strip().lower()
        if user_input in {"n", "no", "not", "-", "nuh"}:
            print("Aborting...")
            sys.exit(0)

    return folder


def validate_image(image: np.ndarray) -> None:
    """
    Validates the input image by checking its existence, dimensions, type, and pixel value range.
    Raises an error if any validation fails.

    Args:
        image (np.ndarray): The input image array to validate.

    Raises:
        ValueError: If the input image is None.
        ValueError: If the input image has no dimensions or is empty.
        ValueError: If the input image type is not np.uint8.
        ValueError: If the input image pixel values exceed the range of 0 to 255.
    """
    if image is None:
        raise ValueError("Image didn't found. Please check your input.")
    if image.shape[0] == 0 or image.shape[1] == 0 or image.size == 0:
        raise ValueError("Image is empty")
    if image.dtype != np.uint8:
        raise ValueError("Image must be of type uint8")
    if np.max(image) > 255:
        raise ValueError("Image pixel values must be between 0 and 255")
