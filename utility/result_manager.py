import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from settings.constants import SOURCE, ICON, SIM_CLASSES, SIM_CLASSES_PERC, SIM_BEST_CLASS, FILE
from utility.classifying_tools import extract_item_from_preds


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


def _load_result_paths(depth: int, classifier_name: str) -> ResultPaths:
    """
    Generate paths for regular and summary results for a specific classifier and depth.

    Args:
        depth: The depth parameter used in the classification
        classifier_name: Name of the classifier

    Returns:
        ResultPaths object containing paths to regular and summary results
    """
    base_path = Path('results') / f'depth-{depth}'
    regular_path = base_path / f"{classifier_name}-depth-{depth}.csv"
    summary_path = base_path / f"{classifier_name}-summary-depth-{depth}.csv"

    return ResultPaths(regular=regular_path, summary=summary_path)


def load_summary_results(depth: int, classifier_name: str, describe: bool = False) -> Optional[pd.DataFrame]:
    """
    Load summary results for specified depth and classifier.

    Args:
        depth: The depth parameter used in the classification
        classifier_name: Name of the classifier
        describe: If True, prints details about the summary results

    Returns:
        DataFrame containing the summary results
    """

    try:
        paths = _load_result_paths(depth, classifier_name)
        summary_df = pd.read_csv(paths.summary)

        if describe:  # Only print details if describe is True
            print(f"\nSummary for {classifier_name} at depth {depth}:")
            print("Shape:", summary_df.shape)
            print("Columns:", summary_df.columns.tolist())
            print("First few rows:")
            summary_df.head()
        return summary_df

    except Exception as e:
        print(f"{e}")
        return None

def compare_summaries(classifier_names: List[str],
                      depths: Union[List[int], int],
                      target_value: str = "mean"
                      ) -> pd.DataFrame:
    """
    Compares summary results across multiple classifiers and depths.

    Args:
        classifier_names (List[str]): List of classifier names to compare.
        depths (List[int]): List of depth values to compare.
        target_value (str, optional): Target value to extract from summary results. Defaults to "mean".

    Returns:
        pd.DataFrame: DataFrame containing mean values for each classifier-depth combination.
    """
    if isinstance(depths, int):
        logging.info("Provided int. Converting to list")
        depths = [depths]
    if depths is None:
        logging.warning("Depths list is empty. Defaulting to depth 5 instead")
        depths = [5]
    if not isinstance(target_value, str):
        logging.warning("Target value is not a string. Defaulting to 'mean'")
        target_value = "mean"

    data_list = []

    for classifier, depth in product(classifier_names, depths):
        summary_df = load_summary_results(depth, classifier)
        if summary_df is not None:
            try:
                target_values = summary_df.set_index(summary_df.columns[0]).loc[target_value]

                data_list.append({
                    "Classifier": classifier,
                    "Depth": depth,
                    SIM_CLASSES: target_values[SIM_CLASSES],
                    SIM_CLASSES_PERC: target_values[SIM_CLASSES_PERC],
                    SIM_BEST_CLASS: target_values[SIM_BEST_CLASS]
                })
            except KeyError:
                print(f"Skipping {classifier} at depth {depth}: {target_value} row not found.")

    return pd.DataFrame(data_list)


def visualize_comparison(comparison_data: pd.DataFrame, metric: str, title: str = None, figsize=(12, 6)) -> None:
    """
    Visualizes a heatmap comparison for a specified metric across classifiers and depths.

    Args:
        comparison_data (pd.DataFrame): The input data containing classifier, depth,
            and metric columns for visualization.
        metric (str): The metric to visualize, must exist in the provided data columns.
        title (str, optional): The title for the visualization. Defaults to None. If
            None, the metric name will be used as the title.
        figsize (tuple): The size of the figure (width, height). Both dimensions must
            be positive.

    Raises:
        ValueError: If `comparison_data` is empty.
        ValueError: If the `metric` is not found in the columns of `comparison_data`.
        ValueError: If any dimension in `figsize` is non-positive.

    """

    if comparison_data.empty:
        raise ValueError("Comparison data cannot be empty")
    if metric not in comparison_data.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison data columns")
    if not all(x > 0 for x in figsize):
        raise ValueError("Figure size dimensions must be positive")

    # Pivot the data so that classifiers are rows and depths are columns
    heatmap_data = comparison_data.pivot(index= "Classifier", columns="Depth", values=metric)

    plt.figure(figsize=figsize)

    ax = sns.heatmap(heatmap_data,
                annot=True,
                cmap="viridis",
                fmt=".2f",
                linewidths=0.5,
                cbar=True,
                # cbar_kws={"label": metric}  # Label the color bar with the metric name
                )

    # Hide the 'Classifier' axis label
    ax.set_ylabel("")
    ax.set_xlabel("Depth")

    if title:
        plt.title(title)
    else:
        plt.title(f"{metric}")

    plt.tight_layout()
    plt.show()
