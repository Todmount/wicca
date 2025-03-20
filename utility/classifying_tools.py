import os
import sys
import time
import logging
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import wraps

from utility.data_loader import load_image
from settings.constants import MODEL, PRE_INP, DEC_PRED, SHAPE, SOURCE, ICON, RESULTS_FOLDER

# Logging message formatting
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Type aliases
ModelClass = Callable
ModelWithConfig = Tuple[ModelClass, Dict[str, Any]]  # For models with config like NASNetLarge
ModelsDict = Dict[str, Union[ModelClass, ModelWithConfig]]
Depth = Union[int, Tuple[int, ...], List[int], range]


def load_single_model(model_class,
                      shape: Tuple[int, int] = (224, 224),
                      weights: str = 'imagenet'
                      ) -> Optional[dict]:
    """
    Load a classifier model with error handling.

    Parameters:
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


def load_models(models: Dict[str, int]) -> Dict[str, Any]:
    """
    Load multiple image classification models with progress tracking.

    Args:
        models: Dictionary mapping model names to either:
               - a model class, or
               - a tuple of (model_class, config_dict)

    Returns:
        Dictionary mapping model names to loaded classifier instances
    """
    start = time.time()
    classifiers_dict = {}

    for name, model_info in tqdm(models.items(), desc="Loading classifiers"):
        # Check if this is a tuple (model with config) or just a model class
        if isinstance(model_info, tuple):
            model_class, kwargs = model_info
        else:
            model_class = model_info
            kwargs = {}

        classifiers_dict[name] = load_single_model(model_class, **kwargs)

    end = time.time()
    print(f'Total of {len(classifiers_dict)} classifiers loaded in {end - start:.2f} seconds\n')
    return classifiers_dict


def get_prediction(image: np.ndarray,
                   classifier: dict,
                   top: int = 5
                   ) -> list:
    """
      Returns top predictions for the given image using the specified classifier

      Parameters:
          image (numpy.ndarray): The loaded image to be classified
          classifier (dict): image classifier
          top (int): number of top predicted classes

      Returns:
          predictions for the image
    """
    model = classifier[MODEL]
    preprocess_input = classifier[PRE_INP]
    decode_predictions = classifier[DEC_PRED]

    x = np.expand_dims(image, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)

    return decode_predictions(preds, top=top)


def normalize_depth(depth: Depth):
    """
    Normalizes the given depth input.

    The function ensures the provided `depth` is converted into a uniform tuple of
    integers, ensuring compatibility and usability in further operations. It raises
    errors if the input does not conform to the expected types and conditions.

    Args:
        depth: The depth input to be normalized. It can be an integer greater than
            0, a tuple, a list, or a range. A `None` value and invalid types will
            raise respective errors.

    Returns:
        A tuple of integers representing the normalized depth.

    Raises:
        ValueError: If depth is not provided (None).
        ValueError: If depth is not a positive integer, tuple, list, or range.
        ValueError: If any element in the depth is not an integer.
    """
    if depth is None:
        raise ValueError("Depth must be provided")
    if isinstance(depth, int) and depth > 0:
        depth = (depth,)
    if isinstance(depth, (tuple, list, range)):
        depth = tuple(depth)
    else:
        raise ValueError("Depth must be a positive integer, tuple, list, or range")
    if all(isinstance(x, int) for x in depth):
        return depth
    else:
        raise ValueError("All depths must be integers")


def preserve_depth(func):
    """Decorator that preserves the original depth value.
    Saves depth at the start of function and restores it at the end,
    regardless of any changes made within the function."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Save original depth
        original_depth = self.depth

        # Execute the function (which may modify depth internally)
        result = func(self, *args, **kwargs)

        # Restore original depth
        self.depth = original_depth
        return result

    return wrapper


class ClassifierProcessor:
    """
    Handles the processing of classifiers.

    Provides functionality to process single classifiers, multiple
    classifiers with parallel execution, and classifiers across multiple
    transformation depths. Also includes method for visualizing compression effects.

    Attributes:
        path (str): Path to the folder containing images for classification.
        coder (object): Wavelet coder instance used for image compression and resizing.
        depth (Union[int, Tuple[int, ...], List[int], range]): Depth(s) of transformation
            for wavelet compression.
        interpolation (int): Interpolation method used for image resizing (e.g., cv2.INTER_AREA).
        results_folder (str): Path to the folder for saving classification and comparison results.
        top (int): Number of top classes to consider during classification.
        rsltmgr (module): Result manager module that provides methods like get_short_comparison.
    """

    def __init__(self,
                 path: Union[str, Path],
                 coder: Any,
                 depth: Depth,
                 interpolation: int,
                 top: int,
                 rsltmgr,
                 results_folder=RESULTS_FOLDER):
        """
        Initializes the object with provided attributes and normalizes various inputs
        for depth. Ensures depth is consistently treated as a tuple for subsequent
        operations. Defaults results folder if not provided or not a Path object.
        Raises:
            ValueError if the depth parameter is not of an expected type.
        """
        self.path = path
        self.coder = coder
        self.depth = depth
        self.top = top
        self.interpolation = interpolation
        self.results_folder = results_folder
        self.rsltmgr = rsltmgr

        # Normalizing different inputs for various depths
        self.depth = normalize_depth(self.depth)

        # Validating and defaulting wrong results path
        if not isinstance(self.results_folder, Path):
            logging.warning("Results folder provided is not a Path object. \n"
                            "Using default folder {project_root}/results")
            self.results_folder = RESULTS_FOLDER

    def _save_results(self, result: pd.DataFrame, summary: pd.DataFrame, name: str) -> None:
        """
        Saves result and summary data to CSV files in a folder specific to the current depth.

        The method creates a results folder if it does not already exist, using the configured
        `results_folder` attribute and appending the current depth. The result and summary
        data are saved as separate CSV files with names including the given `name` and the
        current depth.

        Args:
            result: DataFrame containing the result data to be saved.
            summary: DataFrame containing the summary data to be saved.
            name: Base name of the files to be saved, to which depth and type will be appended.
        """
        results_folder = self.results_folder / f"depth-{self.depth}"
        if not results_folder.exists():
            logging.info(f"Created folder {results_folder}")
            results_folder.mkdir(parents=True, exist_ok=True)
        result.to_csv(results_folder / f"{name}-depth-{self.depth}.csv")
        summary.to_csv(results_folder / f"{name}-summary-depth-{self.depth}.csv")

    def _classify(self, classifier: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns top predictions for the images and their icons.

        Args:
             classifier ( Dict[str, Any]): image classifier

        Returns:
            predictions for each image in the folder
        """
        dir_list = os.listdir(self.path)

        results = dict()

        for file_name in dir_list:
            image = load_image(f'{self.path}/{file_name}')

            resized = cv2.resize(image, classifier[SHAPE], interpolation=self.interpolation)

            resized_predictions = get_prediction(resized, classifier)

            icon = self.coder.get_small_copy(image, self.depth)
            resized_icon = cv2.resize(icon, classifier[SHAPE], interpolation=self.interpolation)

            icon_predictions = get_prediction(resized_icon, classifier, self.top)

            results[file_name] = {SOURCE: resized_predictions, ICON: icon_predictions}

        return results

    def _process_core(self, item: Tuple[str, dict]) -> Tuple[str, pd.DataFrame]:
        """
        Processes a given item using the provided classifier and saves the resulting
        data.

        The method performs classification of images and icons from a folder using the
        given classifier. It then generates a summary of the classification results
        and saves the results as CSV files. Finally, the method prints a message
        indicating that the classifier processing is complete.

        Args:
            item: Tuple containing the name of the classifier and the classifier object.

        Returns:
            A tuple containing the name of the classifier and the summary dataframe
            generated from the classification results.

        """
        name, classifier = item
        res = self._classify(classifier)

        # Using the rsltmgr module correctly as passed in the constructor
        res_df = self.rsltmgr.get_short_comparison(res, self.top)
        sum_df = res_df.describe()

        # Save CSV files inside the "results" folder
        self._save_results(res_df, sum_df, name)

        print(f"Classifier {name} processed")  # Shut down temporarily
        return name, sum_df

    def _parallel_proc(self, classifiers: Dict[str, Any], timeout: int = None) -> Dict[str, Any]:
        """
        Processes multiple classifiers using parallel threads.

        This method runs the classifier tasks in parallel threads using
        concurrent.futures.ThreadPoolExecutor. It supports a timeout
        parameter for handling cases where the execution exceeds the
        allowed time limit. Captures and handles exceptions such as
        TimeoutError and ValueError if encountered during execution.

        Args:
            classifiers (Dict[str, Any]): A dictionary where keys
                represent classifier names and values represent the
                corresponding data or objects to process.
            timeout (int, optional): The maximum time, in seconds, to
                allow for processing. Defaults to None, meaning no
                timeout is applied.

        Returns:
            Dict[str, Any]: A dictionary of processed results where
                keys match the classifier names and values match the
                output from `_process_core`. Returns an empty dictionary
                in case of exceptions such as TimeoutError or ValueError.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                results = dict(executor.map(
                    self._process_core,
                    classifiers.items(),
                    timeout=timeout
                ))
                return results
            except concurrent.futures.TimeoutError:
                print("Timeout occurred. Processing aborted.")
                return {}
            except ValueError as e:
                print(f"An error occurred: {str(e)}")
                # raise
                return {}

    def _single_classifier(self, name: str,
                           classifier_dict: Dict[str, Any],
                           timeout: int = None
                           ):
        """
        Helper function to process a single classifier by wrapping it and validating input parameters.

        This function validates the provided classifier and name, wraps the classifier
        into a dictionary format, and delegates the processing to another method with
        a given timeout.

        Args:
            name: Name of the classifier to be processed.
            classifier_dict: A dictionary containing classifier details. Must include
                a 'MODEL' key to specify the classifier model.
            timeout: Timeout in seconds for processing. Defaults to None. Recommended
                to be set to 3600 seconds (1 hour) or more.

        Raises:
            ValueError: If the name is not provided.
            ValueError: If the classifier_dict is not a dictionary or does not contain
                the 'MODEL' key.
        """

        # Validate inputs
        if not name:
            raise ValueError("Name must be provided for single classifier")
        if not isinstance(classifier_dict, dict) or MODEL not in classifier_dict:
            raise ValueError(f"Classifier must be a dictionary containing a '{MODEL}' key")
        if timeout is None or not isinstance(timeout, int) or timeout < 0:
            logging.warning("Timeout for processing is not set or invalid. It's value should be a positive integer.\n"
                            "It is recommended to set it to 3600 seconds (1 hour) or more.\n"
                            "Defaulting to None.")
            timeout = None

        # Wrap and process the classifier
        wrapped_classifier = {name: classifier_dict}
        return self.process_classifiers(wrapped_classifier, timeout)

    def process_single_classifier(self, *args, **kwargs):
        """
        Safely process a classifier with helpful error messages for common mistakes.

        This is a helper wrapper around process_single_classifier that catches and
        explains common errors.

        Returns:
        --------
        The result from process_single_classifier or None if an error occurs
        """
        try:
            return self._single_classifier(*args, **kwargs)
        except TypeError as e:
            if "missing 1 required positional argument: 'classifier_dict'" in str(e):
                logging.error("You need to provide both the name and the classifier dictionary.\n"
                              "Correct usage: process_single_classifier(name, classifier_dict)\n"
                              "Example: process_single_classifier('VGG19', classifiers['VGG19'])\n")
                return None
            else:
                raise

    @preserve_depth
    def process_classifiers(self,
                            classifiers: Dict[str, Any],
                            timeout: int = None
                            ):
        """
        Processes classifiers across specified depths with optional timeout.

        This method iterates over configured depths to process classifiers. It provides
        parallel processing capabilities and logs comprehensive time statistics, including
        dynamic formatting of the elapsed time.

        Args:
            classifiers (Dict[str, Any]): A dictionary of classifiers to be processed. The key
                is the classifier name, and the value is its configuration or data.
            timeout (int, optional): Specifies the timeout value for processing classifiers.

        Raises:
            Exception: If attempting to process a single classifier by mistake .
        """

        # Handle a single classifier
        if MODEL in classifiers:
            raise Exception("\nIt appears you are trying to process a single classifier.\n"
                            "Use process_single_classifier instead.")

        # Debugging
        # If I forget to delete, consider it an eastern egg
        if timeout == 1984:
            raise Exception("Big Brother is watching you...\n"
                            "If you want to proceed try another timeout\n")

        results = {}

        start_time = time.time()
        for depth in self.depth:
            self.depth = depth
            if isinstance(self.depth, int) and self.depth > 0:
                print(f"Processing at depth {depth}")
                depth_res = self._parallel_proc(classifiers, timeout)
                results.update(depth_res)
                print(f"Depth {depth} processed\n")
            else:
                print(f"Depth '{depth}' not valid. Skipping...\n")

        end_time = time.time()
        total_seconds = int(end_time - start_time)
        logging.info(f"Total processing time: {total_seconds} seconds")

        # Convert to hours, minutes, seconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        # Build dynamic time string
        time_parts = []
        if hours > 0:
            time_parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0 or (hours > 0 and seconds > 0):  # Show minutes if hours and seconds exist
            time_parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
        if seconds > 0 or not time_parts:  # Always show seconds if there are no other units
            time_parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")

        time_str = " ".join(time_parts)
        print(f"Total processing time: {time_str}")
        # You can uncomment this in case you want to see resulting dict for debugging
        # Be aware, that output will become cumbersome
        # return results
