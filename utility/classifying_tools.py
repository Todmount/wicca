import os
import sys
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from typing import Optional, Any, Dict, Tuple, Union, List

from utility.data_loader import load_image
from settings.constants import MODEL, PRE_INP, DEC_PRED, SHAPE, SOURCE, ICON


def load_classifier(model_class,
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


def classify_images_n_icons_from_folder(classifier: dict,
                                        folder: str,
                                        coder: Any,
                                        depth: Tuple[int, ...],
                                        top: int = 5,
                                        interpolation: int = cv2.INTER_AREA
                                        ) -> dict:
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

    Including image classification, resizing, and
    comparison tasks.

    Provides functionality to process single classifiers, multiple
    classifiers with parallel execution, and classifiers across multiple
    transformation depths. Also includes method for visualizing compression effects.

    This class is designed to work with wavelet image compressors and classifiers to
    facilitate tasks like image classification, compression, and result management. It
    saves results to disk and supports working with multiple transformation depths to
    analyze classifiers' performance. The class is also capable of visualizing the effect
    of wavelet compression on images.

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
        if isinstance(self.depth, int) and self.depth > 0:
            self.depth = (self.depth,)
        elif isinstance(self.depth, (tuple, list, range)):
            self.depth = tuple(self.depth)
        else:
            raise ValueError("Depth must be a positive integer, tuple, list, or range")

    def _save_results(self, result, summary, name: str) -> None:
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
            print(f"Created folder {results_folder}")
            os.makedirs(results_folder, exist_ok=True)

        result.to_csv(os.path.join(results_folder, f"{name}-depth_{self.depth}.csv"))
        summary.to_csv(os.path.join(results_folder, f"{name}-summary-depth_{self.depth}.csv"))

    def _process_core(self, item):
        """Core processing function for single classifier.

        Is NOT meant to be called directly.
        """

        if self.depth is None:
            raise ValueError("Depth must be provided")
        # if self.depth <= 0:
        #     raise ValueError("Depth must be positive")

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

    # def _process_classifier_for_batch(self, item) -> Tuple[str, Any]:
    #     """
    #     Process a single classifier.
    #
    #     Parameters:
    #     -----------
    #     item : tuple
    #         A tuple containing (name, classifier)
    #
    #     Returns:
    #     --------
    #     tuple
    #         A tuple of (name, summary_dataframe)
    #     """
    #
    #     name, classifier = item
    #     return self._process_core(classifier, name)

    # def process_classifier(self, item, name: str) -> Tuple[str, Any]:
    #     """Process a single classifier."""
    #
    #     if name is None:
    #         raise ValueError("Name must be provided")
    #     if not isinstance(self.depth, int):
    #         raise ValueError("This function is meant for singular depth.\n"
    #                          "Maybe you meant to use process_all_classifiers_by_depths?")
    #
    #     return self._process_core(item, name)

    def _parallel_proc(self, classifiers, timeout: int = 3600) -> Dict[str, Any]:
        """
        Process multiple classifiers in parallel.

        Parameters:
        -----------
        classifiers : dict
            Dictionary of classifiers where keys are names and values are classifier instances
        timeout : int, optional
            Maximum execution time in seconds, defaults to 1 hour

        Returns:
        --------
        dict
            Dictionary with results where keys are classifier names and values are summary dataframes
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
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return {}

    def process_classifiers(self,
                            classifiers,
                            name: str = None,
                            timeout: int = 3600
                            ):

        # Handle a single classifier
        if not isinstance(classifiers, dict):
            if name is None:
                # Try to infer name or raise error if we can't
                if hasattr(classifiers, '__name__'):
                    name = classifiers.__name__
                elif hasattr(classifiers, '__class__'):
                    name = classifiers.__class__.__name__
                else:
                    raise ValueError("When providing a single classifier, the 'name' parameter must be specified")

            # Convert single classifier to a dict with one entry
            classifiers = {name: classifiers}

        for depth in self.depth:
            self.depth = depth
            print(f"Processing at depth {depth}")
            self._parallel_proc(classifiers, timeout)
            print(f"Depth {depth} processed\n")if self.depth > 0 else print("Skipped incorrect depth")

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