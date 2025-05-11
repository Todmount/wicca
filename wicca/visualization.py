import logging

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import TYPE_CHECKING#,Tuple, Union, List

from wicca.normalization import normalize_depth
from wicca.validation import validate_image
from wicca.config.aliases import Depth

if TYPE_CHECKING:
    import pandas as pd


def show_image_vs_icon(image: np.ndarray,
                       depth_value: Depth,
                       coder,
                       figsize: tuple[int, int] = None
                       ) -> None:
    """
    Displays a series of images showcasing the original image alongside its transformed
    icons at different depth levels. The method dynamically calculates the grid
    structures required for plotting these images and presents them in a single figure.

    Args:
        image (np.ndarray): The original image to be used for generating transformed
            icons.
        coder: An instance of the Coder class containing the transformation methods.
        depth_value (Depth): A depth parameter or iterable determining the levels
            of transformation for the icons.
        figsize (Tuple[int, int], optional): Dimensions (width, height) of the
            figure in inches. Defaults to an automatically calculated size if not provided.

    Raises:
        ValueError: If the `image` parameter is None.
    """
    # Validation
    validate_image(image)
    depth_value = normalize_depth(depth_value)

    original = image  # new var for better readability

    # Calculate the number of subplots needed (original + one for each depth)
    n_depths = len(depth_value)
    total_plots = n_depths + 1

    # Calculate grid
    ncols = int(min(3, total_plots))
    nrows = int(np.ceil(total_plots / ncols))

    if figsize:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    else:
        logging.warning(f'No figsize provided. Using calculated figsize: {(4 * ncols, 4 * nrows)}')
        fig, ax = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # fig, ax = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # Flatten axes array if multidimensional
    if total_plots > 1:
        ax = ax.flatten()
    else:
        ax = tuple(ax)  # making it iterable

    # Original image going first
    ax[0].imshow(original)
    ax[0].set_title(f"Source, shape = {original.shape}")
    ax[0].axis('off')

    for i, depth in enumerate(depth_value, start=1):
        depth_value = depth
        compressed = coder.get_small_copy(
            image=original,
            transform_depth=depth_value
        )

        ax[i].imshow(compressed)
        ax[i].set_title(f"Icon, depth = {depth}, shape = {compressed.shape}")
        ax[i].axis('off')

    # Turn off unused subplots
    for i in range(total_plots, len(ax)):
        ax[i].axis('off')
        ax[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def show_icon_on_image(image: np.ndarray,
                       depth_value: Depth,
                       coder,
                       border_width: int = 1,
                       border_color: tuple = (255, 255, 255),
                       figsize: tuple[int, int] = None
                       ) -> None:
    """
    Visualizes the Discrete Wavelet Transform (DWT) of an image by displaying the original image
    with a smaller transformed copy overlaid in the top-left corner. Optionally, a border can be
    applied around the transformed copy.

    Args:
        image (np.ndarray): The input image to visualize.
        depth_value (Depth): Transformation depth values applied for wavelet encoding.
        coder: An object responsible for generating wavelet-transformed versions of the input image.
        border_width (int): The width of the border around the transformed portion, default is 1.
        border_color (tuple): The color of the border as an RGB tuple, default is (255, 255, 255).
        figsize (Tuple[int, int]): The dimensions of the figure for plotting, default is None.
    """
    # Validation
    validate_image(image)
    depth_value = normalize_depth(depth_value)

    original = image.copy()  # new var for better readability

    for i, depth in enumerate(depth_value, start=1):
        depth_value = depth
        # Getting small copy (icon)
        icon = coder.get_small_copy(
            image=original,
            transform_depth=depth_value
        )
        # Get icon dimension
        icon_height, icon_width = icon.shape[:2]

        # Draw border if provided
        if border_width > 0:
            if len(original.shape) == 3:
                original[0:icon_height + 2 * border_width, 0:icon_width + 2 * border_width] = border_color
            else:
                original[0:icon_height + 2 * border_width, 0:icon_width + 2 * border_width, 0] = border_color[0]

        original[border_width:icon_height + border_width,
        border_width:icon_width + border_width] = icon

    # Display the result
    if figsize:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=(10, 10))

    plt.imshow(original)
    plt.title(f"Original image with icon in top-left corner")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_comparison(comparison_data: 'pd.DataFrame', metric: str, title: str = None, figsize=(12, 6)) -> None:
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
    heatmap_data = comparison_data.pivot(index="Classifier", columns="Depth", values=metric)

    plt.figure(figsize=figsize)

    ax = sns.heatmap(heatmap_data,
                     annot=True,
                     cmap="viridis",
                     fmt=".2f",
                     linewidths=0.5,
                     cbar=False,
                     # cbar_kws={"label": metric}  # Label the color bar with the metric name
                     )
    ax.set_ylabel("")
    ax.set_xlabel("Depth")

    if title:
        plt.title(title)
    else:
        plt.title(f"{metric}")

    plt.tight_layout()
    plt.show()


def plot_metric_radar(names, metric, title: str = None, min_value: int = None, max_value: int = 100) -> None:
    fig = go.Figure(data=go.Scatterpolar(
        r=metric,
        theta=names,
        fill="toself",
        fillcolor='rgba(0,100,80,0.2)',
        mode="lines+markers",
        line=dict(color='rgb(0,100,80)', width=2)
    ))
    if min_value:
        minimum = min_value
    else:
        minimum = min(metric) * 0.9
        logging.warning("Minimum value not provided. \n"
                        f"Calculated minimum value from metric values: "
                        f"{minimum:.0f}%"
                        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[minimum, max_value],
                angle=360,
                tickangle=0
            ),
        ),
        showlegend=False,
        title=dict(
            text=title,
            x=0.5,  # 0.5 for center alignment
            xanchor='center'
        )
    )
    fig.show()


def plot_compare_metrics(names, metric1, metric2, xlabel: str = None, ylabel: str = None, title: str = None):
    xlable = "Similar classes, %" if not xlabel else xlabel
    ylable = "Best class similarity, %" if not ylabel else ylabel
    title = "Classifier Performance Comparison" if not title else title

    # Create a scatter plot
    fig = px.scatter(
        x=metric1,
        y=metric2,
        text=names,
        labels={'x': xlable, 'y': ylable},
        title=title
    )
    # Adjust text positioning
    fig.update_traces(textposition='top center', marker_size=12)
    # Add a line for reference
    # fig.add_shape(
    #     type="line",
    #     x0=min(metric1) * 0.95,
    #     y0=min(metric2) * 0.95,
    #     x1=max(metric1) * 1.05,
    #     y1=max(metric2) * 1.05,
    #     line=dict(color="Gray", width=1, dash="dash")
    # )
    fig.show()
