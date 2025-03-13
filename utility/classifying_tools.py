import os
import sys
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from typing import Optional, Any, Dict, Tuple

from utility.data_loader import load_image
from settings.constants import MODEL, PRE_INP, DEC_PRED, SHAPE, SOURCE, ICON


def load_classifier(model_class,
                    shape: Tuple[int, int] = (224, 224),
                    weights:str = 'imagenet'
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
            PRE_INP: getattr(module,'preprocess_input'),
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
                                        depth: int = 1,
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
    A class to handle batch processing of multiple classifiers with the same parameters.
    """

    def __init__(self,
                 path: str,
                 coder: Any,
                 depth: int,
                 interpolation: int,
                 results_folder: str,
                 top: int,
                 rsltmgr):
        """
        Initialize the classifier processor with all required dependencies.

        Parameters:
        -----------
        path : str
            Path to the folder containing images to classify
        coder : object
            Wavelet coder instance (like HaarCoder)
        depth : int
            The depth of transforming for wavelet compression
        top : int
            Number of top classes to use for comparison
        interpolation : int
            Type of interpolation used in resizing (e.g., cv2.INTER_AREA)
        results_folder : str
            Path to the folder where results will be saved
        rsltmgr : module
            Result manager module containing get_short_comparison function
        """
        self.path = path
        self.coder = coder
        self.depth = depth
        self.top = top
        self.interpolation = interpolation
        self.results_folder = results_folder
        self.rsltmgr = rsltmgr

        # Ensure results directory exists
        os.makedirs(self.results_folder, exist_ok=True)

    # def process_classifier(self, item) -> Tuple[str, Any]:
    def process_classifier(self, item: Tuple[str, Dict], name: str = None) -> Tuple[str, Any]:
        """
        Process a single classifier.

        Parameters:
        -----------
        item : tuple
            A tuple containing (name, classifier)

        Returns:
        --------
        tuple
            A tuple of (name, summary_dataframe)
        """

        # Handle both tuple input and direct classifier input
        if isinstance(item, tuple) and len(item) == 2:
            name, classifier = item
        else:
            if name is None:
                # Try to get the name from the classifier object if possible
                try:
                    name = item.__class__.__name__
                except AttributeError:
                    name = "UnnamedClassifier"
            classifier = item

        res = classify_images_n_icons_from_folder(
            classifier, self.path, self.coder, self.depth, self.top, self.interpolation
        )

        # Using the rsltmgr module correctly as passed in the constructor
        res_df = self.rsltmgr.get_short_comparison(res, self.top)
        # res_df = res_df.style.format('{:.4f}')

        # Save CSV files inside the "results" folder
        res_df.to_csv(os.path.join(self.results_folder, f"{name}-depth_{self.depth}.csv"))
        sum_df = res_df.describe()
        sum_df.to_csv(os.path.join(self.results_folder, f"{name}-summary-depth_{self.depth}.csv"))

        return name, sum_df

    def process_all_classifiers(self, classifiers: Dict[str, Dict], timeout: int = 3600) -> Dict[str, Any]:
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
                    self.process_classifier,
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        axes[0].imshow(original), axes[0].set_title("Original Image")
        axes[1].imshow(compressed), axes[1].set_title(f"Compressed Image, depth = {self.depth}")
        plt.show()

        # Print image statistics
        print(f"Original size: {original.shape}")
        print(f"Compressed size: {compressed.shape}")
