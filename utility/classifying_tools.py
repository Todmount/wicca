import numpy as np
import cv2
import os
import pandas as pd
from typing import Optional

from utility.data_loader import load_image
from settings.constants import MODEL, PRE_INP, DEC_PRED, SHAPE, SOURCE, ICON


def get_prediction(image: np.ndarray, classifier: dict, top: int = 5) -> list:
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


def classify_images_n_icons_from_folder(classifier: dict, folder: str, coder, depth: int = 1,
                                        interpolation: int = cv2.INTER_AREA) -> dict:
    """
    Returns top predictions for the images and their icons.
         classifier (dict): image classifier
         folder (str): folder with images to be classified
         coder (WaveletCoder): wavelet coder
         depth (int): the depth of the discrete wavelet transform (DWT).
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

        icon_predictions = get_prediction(resized_icon, classifier)

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