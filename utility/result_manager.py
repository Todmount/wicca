import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Optional, Union, Tuple

import pandas as pd

from settings.constants import SOURCE, ICON, SIM_CLASSES, SIM_CLASSES_PERC, SIM_BEST_CLASS, FILE
from utility.classifying_tools import extract_item_from_preds
from utility.classifying_tools import normalize_depth

# Type aliases
Depth = Union[int, Tuple[int, ...], List[int], range]


@dataclass
class ResultPaths:
    regular: Path
    summary: Path


def get_short_comparison(results: dict, top: int) -> pd.DataFrame:
    """
    Generates a DataFrame comparing prediction results for source and icon
    image classes, and calculates similarity metrics based on top predictions.

    Args:
        results (dict): A dictionary containing prediction results for multiple files.
            Keys represent file names and values contain predictions for source and
            icon, with each being a list of tuples (class_label, probability).
        top (int): The number of top predictions to consider for calculating
            similarity metrics.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - FILE: List of file names processed from the input dictionary.
            - SIM_CLASSES: Number of similar classes between source and icon for each file.
            - SIM_CLASSES_PERC: Percentage of similar classes based on the top predictions.
            - SIM_BEST_CLASS: Binary indicator if the top class matches between
              source and icon.
    """
    # Initialize lists to store comparison metrics
    file_names = []
    similar_classes = []
    similar_classes_percentage = []
    best_class_eq = []

    # Process each file in results
    for file, preds in results.items():
        file_names.append(file)

        # Extract prediction data for source and icon
        src_preds = preds[SOURCE][0]
        icn_preds = preds[ICON][0]

        # Extract class labels and probabilities
        src_classes = extract_item_from_preds(src_preds, 1)
        icn_classes = extract_item_from_preds(icn_preds, 1)
        src_probs = extract_item_from_preds(src_preds, 2)
        icn_probs = extract_item_from_preds(icn_preds, 2)

        # Calculate similarity metrics
        similar_classes_count = len(set(src_classes) & set(icn_classes))
        similar_classes.append(similar_classes_count)
        similar_classes_percentage.append(float(similar_classes_count / top) * 100)

        # Check if top class matches
        best_class_eq.append(int(src_classes[0] == icn_classes[0]))

    return pd.DataFrame({FILE: file_names,
                         SIM_CLASSES: similar_classes,
                         SIM_CLASSES_PERC: similar_classes_percentage,
                         SIM_BEST_CLASS: best_class_eq})


def _load_result_paths(results_folder: Path, depth: Depth, classifier_name: str) -> ResultPaths:
    """
    Helper function to generate paths for regular and summary results for a specific classifier and depth.

    Args:
        results_folder: Folder you specified containing results
        depth: The depth parameter used in the classification
        classifier_name: Name of the classifier

    Returns:
        ResultPaths object containing paths to regular and summary results
    """
    base_path = results_folder / f'depth-{depth}'
    regular_path = base_path / f"{classifier_name}-depth-{depth}.csv"
    summary_path = base_path / f"{classifier_name}-summary-depth-{depth}.csv"

    return ResultPaths(regular=regular_path, summary=summary_path)


def load_summary_results(results_folder: Path, depth: Depth, classifier_name: str, describe: bool = False) -> Optional[pd.DataFrame]:
    """
    Load summary results for specified depth and classifier.

    Args:
        results_folder(Path): Folder you specified containing results
        depth: The depth parameter used in the classification
        classifier_name: Name of the classifier
        describe: If True, prints details about the summary results

    Returns:
        DataFrame containing the summary results
    """
    if not isinstance(describe, bool):
        logging.warning("describe parameter is not a boolean. Defaulting to False")
        describe = False
    try:
        paths = _load_result_paths(results_folder, depth, classifier_name)
        summary_df = pd.read_csv(paths.summary)

        if describe:
            print(f"\nSummary for {classifier_name} at depth {depth}:")
            print("Shape:", summary_df.shape)
            print("Columns:", summary_df.columns.tolist())
            print("First few rows:")
            summary_df.head()
        return summary_df

    except Exception as e:
        print(f"{e}")
        return None


def compare_summaries(results_folder: Path,
                      classifier_names: List[str],
                      depths: Depth,
                      target_stat: str = "mean"
                      ) -> pd.DataFrame:
    """
    Compares summary results across multiple classifiers and depths.

    Args:
        results_folder(Path): Folder you specified containing results
        classifier_names (List[str]): List of classifier names to compare.
        depths (List[int]): List of depth values to compare.
        target_stat (str, optional): Target value to extract from summary results. Defaults to "mean".

    Returns:
        pd.DataFrame: DataFrame containing mean values for each classifier-depth combination.
    """
    depths = normalize_depth(depths)
    if not isinstance(target_stat, str):
        logging.warning("Target value is not a string. Defaulting to 'mean'")
        target_stat = "mean"

    data_list = []

    for classifier, depth in product(classifier_names, depths):
        summary_df = load_summary_results(results_folder, depth, classifier)
        if summary_df is not None:
            try:
                target_values = summary_df.set_index(summary_df.columns[0]).loc[target_stat]

                data_list.append({
                    "Classifier": classifier,
                    "Depth": depth,
                    SIM_CLASSES: target_values[SIM_CLASSES],
                    SIM_CLASSES_PERC: target_values[SIM_CLASSES_PERC],
                    SIM_BEST_CLASS: target_values[SIM_BEST_CLASS]
                })
            except KeyError:
                print(f"Skipping {classifier} at depth {depth}: {target_stat} row not found.")

    return pd.DataFrame(data_list)

def extract_from_comparison(comparison_data: 'pd.DataFrame', metric: str) -> 'pd.DataFrame':
    """"""
    if metric not in comparison_data.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison data.")

    # Get the classifier names (from the 'Classifier' column)
    names = comparison_data['Classifier'].tolist()

    return names, comparison_data[metric].tolist()