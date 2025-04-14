import os
import sys
import time
import random
import logging
import concurrent.futures
from pathlib import Path
from typing import Any
from collections.abc import Iterable

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from functools import wraps
import tensorflow as tf

from wicca.data_loader import load_image
from wicca.normalization import normalize_depth
from wicca.validation import validate_input_folder, validate_output_folder
from wicca.config.aliases import Depth
from wicca.config.constants import MODEL, PRE_INP, DEC_PRED, SHAPE, SOURCE, ICON, RESULTS_FOLDER, MAX_INFO_SAMPLE_SIZE

# Basic configs
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
# logging.getLogger().setLevel(logging.INFO)
bar_format = '{desc}: {percentage:3.0f}%|{bar}|[{elapsed}]'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')


def is_jupyter():
    """Check if the code is running in the jupyter notebook."""
    return True if 'ipykernel' in sys.modules else False


def preserve_depth(func):
    """
    Decorator that preserves the original depth value.
    Saves depth at the start of function and restores it at the end,
    regardless of any changes made within the function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        original_depth = self.depth
        try:
            result = func(self, *args, **kwargs)
            return result
        finally:
            self.depth = original_depth

    return wrapper


def format_proc_time(start: float, end: float) -> str:
    """
    Simple function to format processing time

    Args:
        start: Start time
        end: End time

    Returns:
        Time string in format: hours:minutes:seconds
    """
    total_seconds = int(end - start)

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

    return time_str


class ClassifierProcessor:
    """
    Handles the processing of classifiers. See __init__ for more details.

    Provides functionality to process single classifiers, multiple
    classifiers with parallel execution, and classifiers across multiple
    transformation depths.
    """

    def __init__(self,
                 data_folder: str | Path,
                 wavelet_coder: Any,
                 transform_depth: Depth,
                 interpolation: int,
                 top_classes: int,
                 result_manager,
                 results_folder: str | Path = RESULTS_FOLDER,
                 log_info: bool = True,
                 parallel: int = None,
                 batch_size: int = 25):
        """
        Initializes an instance of a class and validates input parameters to ensure they
        comply with expected types and values. Handles potential issues with the `depth`
        parameter by normalizing it and ensures a valid results folder is assigned.

        Args:
            data_folder (str | Path): The path to the resources or directory.
            wavelet_coder (module): Module containing the wavelet processing logic.
            transform_depth (Depth): Provided depth. It can be int, tuple, list, or range. All elements must be positive integers.
            interpolation (int): Configures interpolation level for operations.
            top_classes (int): Limits the number of classes to be compared for each image and icon.
            result_manager (module): Module containing the results management logic.
            results_folder (str | Path): Directory for storing results; falls back to
                a default path if the provided value is invalid.
            log_info (bool): Controls whether to log information about the initialized instance.
            parallel (int): Controls the number of parallel threads to use for processing. Defaults to unlimited.
            batch_size (int): Controls the number of images to process in each batch. Defaults to 25.
        """
        self.path = validate_input_folder(data_folder)
        self.coder = wavelet_coder
        self.depth = normalize_depth(transform_depth)
        if isinstance(top_classes, int) and top_classes > 0:
            self.top = top_classes
        else:
            msg = f"Top classes must be a non-negative integer. Please provide a valid value"
            logging.error(msg)
            raise ValueError(msg)
        self.interpolation = interpolation
        self.results_folder = validate_output_folder(results_folder)
        self.rsltmgr = result_manager
        self._log_init_info() if log_info else None
        self.parallel = parallel
        self.batch_size = batch_size

    def _log_init_info(self):
        """Logs information about the initialized instance"""
        # Count images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        image_files = [f for f in self.path.iterdir()
                       if f.is_file() and f.suffix.lower() in image_extensions]
        image_count = len(image_files)

        # Map interpolation integer values to cv2 constant names
        interpolation_map = {
            0: "cv2.INTER_NEAREST",
            1: "cv2.INTER_LINEAR",
            2: "cv2.INTER_CUBIC",
            3: "cv2.INTER_AREA",
            4: "cv2.INTER_LANCZOS4",
            5: "cv2.INTER_LINEAR_EXACT",
            6: "cv2.INTER_NEAREST_EXACT",
            7: "cv2.INTER_MAX",
            8: "cv2.WARP_FILL_OUTLIERS",
            16: "cv2.WARP_INVERSE_MAP"
        }
        interpolation_name = interpolation_map.get(self.interpolation, f"Unknown ({self.interpolation})")

        # Get image resolution statistics if images exist
        if image_count > 0:
            sample_size = min(MAX_INFO_SAMPLE_SIZE, image_count)  # Limit to 50 images for performance
            sampled_files = random.sample(image_files, sample_size) if image_count > sample_size else image_files

            width, height = [], []

            for img_path in tqdm(sampled_files, desc="Gathering info", bar_format=bar_format):
                try:
                    img = cv2.imread(str(img_path))
                    width.append(img.shape[1])
                    height.append(img.shape[0])
                except Exception as e:
                    logging.warning(f"Unexpected error reading image {img_path}: {e} \nSkipping... \n")
                    continue

            if width and height:
                mean_width = int(np.mean(width))
                mean_height = int(np.mean(height))

                # Calculate mean resolution (pixels)
                mean_resolution = sum(w * h for w, h in zip(width, height)) / len(width)

                imgs_dims = f"{mean_width}x{mean_height} px"
                # Format resolution in a more human-readable way
                if mean_resolution >= 1_000_000:
                    res_info = f"{mean_resolution / 1_000_000:.1f} MP ({(int(mean_resolution))} pixels)"
                else:
                    res_info = f"{(int(mean_resolution))} pixels"

            # Use a cleaner output approach for Jupyter
            if is_jupyter():
                from IPython.display import display, Markdown
                summary = f"""
#### Image Processing Configuration
Note: For image stats was taken a sample of {sample_size} random images.
You may change the sample size [MAX_INFO_SAMPLE_SIZE] in the config.constants module.
- **Data folder:** {self.path}
- **Number of images:** {image_count}
- **Mean image dimensions:** {imgs_dims}
- **Mean image resolution:** {res_info}
- **Transform depth:** {self.depth}
- **Interpolation:** {interpolation_name}
- **Top classes:** {self.top}
- **Results folder:** {self.results_folder}
                """
                display(Markdown(summary))
            else:
                # Regular logging for non-Jupyter environments
                print(f"\nNote: For image stats was taken a sample of {sample_size} random images."
                      "\nYou may change the max sample size [MAX_INFO_SAMPLE_SIZE] in the config.constants module.")
                print(f"Data folder: {self.path}")
                print(f"Number of images: {image_count}")
                print(f"Mean image dimensions: {imgs_dims}")
                print(f"Mean image resolution: {res_info}")
                print(f"Transform depth: {self.depth}")
                print(f"Interpolation: {interpolation_name}")
                print(f"Top classes: {self.top}")
                print(f"Results folder: {self.results_folder}")

        time.sleep(0.5) # to prevent output overlapping

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

    @tf.function(reduce_retracing=True)
    def _model_predict(self, model, preprocessed_images):
        """Dedicated function for model inference with retracing reduction."""
        return model(preprocessed_images)

    def _get_preds(self, images: np.ndarray, classifier: dict[str, Any]) -> list:
        """
        Returns predictions for a batch of images using the specified classifier.

        Args:
            images (numpy.ndarray): Batch of images with shape (batch_size, height, width, channels)
            classifier (dict): Image classifier dictionary

        Returns:
            List of decoded predictions for each image
        """
        model = classifier[MODEL]
        preprocess_input = classifier[PRE_INP]
        decode_predictions = classifier[DEC_PRED]

        # Preprocess the batch
        preprocessed_images = preprocess_input(images)
        preprocessed_images = tf.cast(preprocessed_images, tf.float32)

        # Get raw predictions
        # preds = model.predict(preprocessed_images, verbose=0)
        preds = self._model_predict(model, preprocessed_images)
        preds_np = preds.numpy()

        # Decode and return predictions for each image in the batch
        return [decode_predictions(preds_np[i:i + 1], top=self.top) for i in range(len(images))]

    def _get_img_batch(self, file_paths: Iterable, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares batches of images and their corresponding icons.

        Args:
            file_paths: List of paths to image files
            shape: Target shape for resizing

        Returns:
            Tuple of (batch_images, batch_icons)
        """

        batch_images = []
        batch_icons = []

        for path in file_paths:
            image = load_image(path)

            resized = cv2.resize(image, shape, interpolation=self.interpolation)

            icon = self.coder.get_small_copy(image, self.depth)
            resized_icon = cv2.resize(icon, shape, interpolation=self.interpolation)

            batch_images.append(resized)
            batch_icons.append(resized_icon)

        return np.stack(batch_images), np.stack(batch_icons)

    def _classify(self, classifier: dict[str, Any]) -> dict[str, Any]:
        """
        Returns top predictions for the images and their icons.

        Args:
             classifier (Dict[str, Any]): image classifier

        Returns:
            predictions for each image in the folder
        """
        dir_list = os.listdir(self.path)
        results = {}

        # Process images in smaller batches to manage memory
        for i in range(0, len(dir_list), self.batch_size):
            batch_files = dir_list[i:i + self.batch_size]
            file_paths = [self.path / file_name for file_name in batch_files]

            batch_images, batch_icons = self._get_img_batch(file_paths, classifier[SHAPE])

            img_preds = self._get_preds(batch_images, classifier)
            icon_preds = self._get_preds(batch_icons, classifier)

            for file_name, img_pred, icon_pred in zip(batch_files, img_preds, icon_preds):
                results[file_name] = {
                    SOURCE: img_pred,
                    ICON: icon_pred
                }

        return results

    def _process_core(self, item: tuple[str, dict]) -> tuple[str, pd.DataFrame]:
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

        return name, sum_df

    def _parallel_proc(self, classifiers: dict[str, Any], timeout: int = None) -> dict[str, Any] | None:
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
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel) as executor:
            # Create a dictionary mapping futures to their keys for identification
            future_to_key = {}
            for key, value in classifiers.items():
                future = executor.submit(self._process_core, (key, value))
                future_to_key[future] = key

            # Process results as they complete
            with tqdm(total=len(future_to_key), desc=f"Processing depth {self.depth}",
                      bar_format=bar_format, mininterval=1) as pbar:
                try:
                    # Process futures as they complete with timeout for the entire process
                    for future in concurrent.futures.as_completed(future_to_key.keys(), timeout=timeout):
                        key = future_to_key[future]
                        try:
                            result = future.result()
                            results[key] = result
                        except Exception as exc:
                            logging.warning(f"Classifier {key} generated an exception: {exc}")
                        finally:
                            pbar.update(1)
                except concurrent.futures.TimeoutError:
                    logging.warning(f"Overall processing timed out after {timeout} seconds")

        return results

    def _single_classifier(self, name: str,
                           classifier_dict: dict[str, Any],
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

    """
    Note for future 
    This is most likely not the best approach to handle depth
    But there was a try to reimplement this part to pass self.depth as tuple further
    The problem is `get_small_copy` from `wavelet_coder.py`, that should receive only int
    If you refactor it to receive a tuple of depth it would break save logic
    Saves would be just `name-depth-3,4...n` and values only from last depth
    This is because we would basically pass all depths to this part of algorithm
    And repeat it by all depth in for loop
    LONG STRINGS ARE NOT GOOD, I know
    """

    @preserve_depth
    def process_classifiers(self,
                            classifiers: dict[str, Any],
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
        results = {}
        start_time = time.time()

        # Handle a single classifier
        if MODEL in classifiers:
            raise Exception("\nIt appears you are trying to process a single classifier.\n"
                            "Use process_single_classifier instead.")

        # Debugging
        # If I forget to delete, consider it an eastern egg
        if timeout == 1984:
            raise Exception("Big Brother is watching you...\n"
                            "If you want to proceed try another timeout\n")

        for depth in self.depth:
            self.depth = depth
            if isinstance(self.depth, int) and self.depth > 0:
                _start_time = time.time()
                # print(f"\nProcessing at depth {depth}")
                depth_res = self._parallel_proc(classifiers, timeout)
                results.update(depth_res)
                _end_time = time.time()
                # print(f"Depth {depth} processed in {int(_end_time - _start_time)} seconds\n")
            else:
                print(f"Depth '{depth}' not valid. Skipping...\n")

        end_time = time.time()
        logging.info(f"Total processing time (log): {int(end_time - start_time)} seconds")  # for debug

        total_time = format_proc_time(start_time, end_time)
        print(f"Total processing time: {total_time}")  # for user
