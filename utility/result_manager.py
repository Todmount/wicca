from dataclasses import dataclass
from pathlib import Path
import pandas as pd

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


def load_summary_results(depth: int, classifier_name: str) -> pd.DataFrame:
    """
    Load summary results for specified depth and classifier.

    Args:
        depth: The depth parameter used in the classification
        classifier_name: Name of the classifier

    Returns:
        DataFrame containing the summary results
    """
    paths = load_result_paths(depth, classifier_name)
    return pd.read_csv(paths.summary)