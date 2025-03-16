import os
import sys
import time
import logging
import concurrent.futures
from typing import Any, Dict, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from utility.data_loader import load_image
from settings.constants import MODEL, PRE_INP, DEC_PRED, SHAPE, SOURCE, ICON

# Logging message formatting
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Type aliases
ModelClass = Callable
ModelWithConfig = Tuple[ModelClass, Dict[str, Any]]  # For models with config like NASNetLarge
ModelsDict = Dict[str, Union[ModelClass, ModelWithConfig]]


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


def load_models(models: ModelsDict) -> Dict[str, Any]:
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


def classify_images_n_icons_from_folder(classifier: Dict[str, Any],
                                        folder: str,
                                        coder: Any,
                                        depth: Tuple[int, ...],
                                        top: int = 5,
                                        interpolation: int = cv2.INTER_AREA
                                        ) -> Dict[str, Any]:
    """
    Returns top predictions for the images and their icons.

    Parameters:
         classifier (dict): image classifier
         folder (str): folder with images to be classified
         coder (WaveletCoder): wavelet coder
         depth (int): the depth of the discrete wavelet transform (DWT)
         top (int): number of top predicted classes (obsolete)
         interpolation (int): type of interpolation

    Returns:
        predictions for each image in the folder
    """
    dir_list = os.listdir(folder)

    results = dict()

    for file_name in dir_list:
        image = load_image(f'{folder}/{file_name}')

        resized = cv2.resize(image, classifier[SHAPE], interpolation=interpolation)

        resized_predictions = get_prediction(resized, classifier)

        icon = coder.get_small_copy(image, depth)
        resized_icon = cv2.resize(icon, classifier[SHAPE], interpolation=interpolation)

        icon_predictions = get_prediction(resized_icon, classifier, top)

        results[file_name] = {SOURCE: resized_predictions, ICON: icon_predictions}

    return results


def extract_item_from_preds(preds: list, idx: int) -> Optional[list]:
    """
    Extract specified items from predictions

    Parameters:
    preds (list): list of predictions
    idx (int): index of the item in predictions

    Returns:
    Array of extracted items
    """

    if idx > 2:
        return None

    items = []
    for pred in preds:
        items.append(pred[idx])

    return items


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
                 path: str,
                 coder: Any,
                 depth: Union[int, Tuple[int, ...], List[int], range],
                 interpolation: int,
                 results_folder: str,
                 top: int,
                 rsltmgr):
        """
        Initializes the object with provided attributes and normalizes various inputs
        for depth. Ensures depth is consistently treated as a tuple for subsequent
        operations. Raises a ValueError if the depth parameter is not of an expected
        type.
        """
        self.path = path
        self.coder = coder
        self.depth = depth
        self.top = top
        self.interpolation = interpolation
        self.results_folder = results_folder
        self.rsltmgr = rsltmgr

        # Normalizing different inputs for various depths
        if self.depth is None:
            raise ValueError("Depth must be provided")
        if isinstance(self.depth, int) and self.depth > 0:
            self.depth = (self.depth,)
        elif isinstance(self.depth, (tuple, list, range)):
            self.depth = tuple(self.depth)
        else:
            raise ValueError("Depth must be a positive integer, tuple, list, or range")

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
        results_folder = os.path.join(self.results_folder, f"depth_{self.depth}")
        if not os.path.exists(results_folder):
            print(f"ATTENTION. Created folder {results_folder}")
            os.makedirs(results_folder, exist_ok=True)

        result.to_csv(os.path.join(results_folder, f"{name}-depth_{self.depth}.csv"))
        summary.to_csv(os.path.join(results_folder, f"{name}-summary-depth_{self.depth}.csv"))

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
        res = classify_images_n_icons_from_folder(
            classifier, self.path, self.coder, self.depth, self.top, self.interpolation
        )

        # Using the rsltmgr module correctly as passed in the constructor
        res_df = self.rsltmgr.get_short_comparison(res, self.top)
        sum_df = res_df.describe()

        # Save CSV files inside the "results" folder
        self._save_results(res_df, sum_df, name)

        print(f"Classifier {name} processed")
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
                print("Processing timed out")
                return {}
            except ValueError as e:
                print(f"An error occurred: {str(e)}")
                # raise
                return {}

    def _single_classifier(self, name: str,
                           classifier_dict: Dict[str, Any],
                           timeout: int = None
                           ) -> None:
        """
        Processes a single classifier by validating inputs, wrapping the classifier
        in a dictionary, and delegating processing to another method. This method
        ensures that the provided input meets the required structure and logs
        a warning when appropriate configurations are absent.

        Args:
            name (str): The name of the classifier. Must be provided.
            classifier_dict (dict): Dictionary representing the classifier. It must
                contain the key required by the system for identification (e.g., 'MODEL').
            timeout (int, optional): The timeout value for processing the classifier in
                seconds. If not provided, a warning is logged suggesting a default
                value of 3600 seconds (1 hour) or more.

        Returns:
            dict: The result of processing the single classifier, generated by
            the `process_classifiers` method.

        Raises:
            ValueError: If the 'name' argument is not provided.
            ValueError: If 'classifier_dict' is not of type dict or does not contain
            the required 'MODEL' key.
        """

        # Validate inputs
        if not name:
            raise ValueError("Name must be provided for single classifier")
        if not isinstance(classifier_dict, dict) or MODEL not in classifier_dict:
            raise ValueError(f"Classifier must be a dictionary containing a '{MODEL}' key")
        if timeout is None:
            logging.warning("Timeout for processing is not set. "
                            "It is recommended to set it to 3600 seconds (1 hour) or more.")

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

    def process_classifiers(self,
                            classifiers: Dict[str, Any],
                            timeout: int = None
                            ):
        """
        Processes multiple classifiers in parallel with optional timeout and depth management.

        This function takes a dictionary of classifiers and processes them in parallel based on
        the depth configurations defined in the instance. It also includes an optional timeout
        parameter. If a single classifier is detected, an exception is raised with guidance on
        redirecting to a dedicated single classifier processing function. Additional handling is
        included for debugging or ensuring proper timeout configurations.

        Args:
            classifiers (Dict[str, Any]): A dictionary of classifiers to be processed.
                Each key-value pair represents a classifier and its associated data.
            timeout (int, optional): Time in seconds before the processing of classifiers
                should timeout. If not provided, no timeout is enforced.

        Raises:
            Exception: Raised if a single classifier is detected instead of multiple,
                prompting the user to use an appropriate method. Additionally, if
                `timeout` is set to 1984, an exception is raised as an intentional
                debugging or thematic mechanism.
        """

        # Handle a single classifier
        if MODEL in classifiers:
            raise Exception("\nIt appears you are trying to process a single classifier.\n"
                             "Use process_single_classifier instead.")

        # My way to understand where the bug is
        # If I forget to delete, consider it an eastern egg
        if timeout == 1984:
            raise Exception("It's your lucky day! Big Brother is not watching")

        start_time = time.time()
        for depth in self.depth:
            self.depth = depth
            if isinstance(self.depth, int) and self.depth > 0:
                print(f"Processing at depth {depth}")
                self._parallel_proc(classifiers, timeout)
                print(f"Depth {depth} processed\n")
            else:
                print(f"Depth '{depth}' not valid. Skipping...\n")
        end_time = time.time()
        print(f"Total processing time is {end_time - start_time:.2f} seconds")

    def show_image_vs_icon(self, image: np.ndarray) -> None:
        """
        Displays an original image alongside its compressed version with specified
        transformation depth. The method also prints the size of both images.

        This function visualizes the effect of compression on an image by comparing
        the original image with its compressed counterpart. The compression is
        performed using the `get_small_copy` method of the `coder` object, with the
        specified depth defined by `self.depth`. The sizes of both the original and
        compressed images are printed for reference.

        Args:
            image (np.ndarray): The original image to be compared with its compressed version.
            It must be a non-empty instance of `np.ndarray`.

        Raises:
            ValueError: If the provided image is None, has zero size, or is not an
            instance of `np.ndarray`.
        """

        if image is None:
            raise ValueError("Image did not load correctly. Please check the file path and try again.")

        original = image
        compressed = self.coder.get_small_copy(
            image=original,
            transform_depth=self.depth
        )

        # Display original vs. compressed
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].imshow(original)
        ax[0].set_title(f"Source, shape = {original.shape}")
        ax[0].axis('off')

        ax[1].imshow(compressed)
        ax[1].set_title(f"Icon, depth = {self.depth}, shape = {compressed.shape}")
        ax[1].axis('off')

        plt.show()

        # Print image statistics
        # print(f"Original size: {original.shape}")
        # print(f"Compressed size: {compressed.shape}")