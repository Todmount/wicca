from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import pandas as pd
from itertools import product

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
    Compare results of classifying source vs. icons

    Parameters:
      results (dict): results of classifying
      top (int): top classes count
    Returns:
      Dataframe summarizing classifying results
    """
    file_names = []
    similar_classes = []
    similar_classes_percentage = []
    best_class_eq = []

    for file, preds in results.items():
        file_names.append(file)

        src_preds = preds[SOURCE][0]
        icn_preds = preds[ICON][0]

        src_classes = extract_item_from_preds(src_preds, 1)
        icn_classes = extract_item_from_preds(icn_preds, 1)

        src_probs = extract_item_from_preds(src_preds, 2)
        icn_probs = extract_item_from_preds(icn_preds, 2)

        similar_classes_count = len(set(src_classes) & set(icn_classes))

        similar_classes.append(similar_classes_count)
        similar_classes_percentage.append(float(similar_classes_count / top) * 100)

        best_class_eq.append(int(src_classes[0] == icn_classes[0]))

    return pd.DataFrame({FILE: file_names, SIM_CLASSES: similar_classes, SIM_CLASSES_PERC: similar_classes_percentage,
                         SIM_BEST_CLASS: best_class_eq})


def load_result_paths(depth: int, classifier_name: str) -> ResultPaths:
    """
    Generate paths for regular and summary results for a specific classifier and depth.

    Args:
        depth: The depth parameter used in the classification
        classifier_name: Name of the classifier

    Returns:
        ResultPaths object containing paths to regular and summary results
    """
    base_path = Path('results') / f'depth_{depth}'
    regular_path = base_path / f"{classifier_name}-depth_{depth}.csv"
    summary_path = base_path / f"{classifier_name}-summary-depth_{depth}.csv"

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
        paths = load_result_paths(depth, classifier_name)
        summary_df = pd.read_csv(paths.summary)

        if describe:  # Only print details if describe is True
            print(f"\nSummary for {classifier_name} at depth {depth}:")
            print("Shape:", summary_df.shape)
            print("Columns:", summary_df.columns.tolist())
            print("First few rows:")
            summary_df.head()
        return summary_df

    except Exception as e:
        print(f"Error loading summary: {e}")
        return None

def compare_summaries(classifier_names: List[str], depths: List[int], target_value: str = "mean") -> pd.DataFrame:
    """
    Compare summary results across multiple classifiers and depths.

    Args:
        classifier_names (List[str]): List of classifier names to compare.
        depths (List[int]): List of depth values to compare.
        target_value (str, optional): Target value to extract from summary results. Defaults to "mean".

    Returns:
        pd.DataFrame: DataFrame containing mean values for each classifier-depth combination.
    """
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
                print(f"Skipping {classifier} at depth {depth}: 'mean' row not found.")

    return pd.DataFrame(data_list)


def visualize_comparison(comparison_data: pd.DataFrame, metric: str, title: str = None, figsize=(12, 6)) -> None:
    """
    Visualizes the comparison of classification performance across depths and classifiers using a heatmap.

    Args:
        comparison_data (pd.DataFrame): DataFrame with values for different classifiers and depths.
        metric (str): The metric to visualize (e.g., 'SIM_CLASSES', 'SIM_CLASSES_PERC', 'SIM_BEST_CLASS').
        title (str, optional): Title for the plot.
        figsize (tuple, optional): Figure size.

    Returns:
        None

    Raises:
        ValueError: If comparison_data is empty or metric is not found in columns.

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
