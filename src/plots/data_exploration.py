import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# NOTE: just writing the basic functions to start. Will come back and write a function
# that wraps these and creates subplots for selected data visualizations


def heatmap(X, annot=True, ax=None, figsize=(8,8)):
    if type(X) != pd.DataFrame:
        raise Exception("data must be pd.DataFrame")

    if ax:
        fig = None
    else:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(X, cmap='magma')
    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_yticks(np.arange(len(X.columns)))
    ax.set_yticklabels(X.columns)

    ax.set_xticks(np.arange(len(X.columns)))
    ax.set_xticklabels(X.columns)

    # Add annotations
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            text = ax.text(j, i, round(X.iloc[i, j], 2), ha="center", va="center", color="k")

    return fig, ax


def scatter_plot(data, feature_name_1, feature_name_2, label_name, ax, figsize=(8,5)):
    """
    Note that labels need to be numeric or else the ax.scatter will throw an error.
    """
    if ax:
        fig = None
    else:
        fig, ax = plt.subplots()

    f1 = data[feature_name_1]
    f2 = data[feature_name_2]
    labels = data[label_name]

    # Get the unique classes
    unique_labels = np.unique(labels)
    scatter = ax.scatter(f1, f2, alpha=0.5, c=labels, cmap="rainbow")

    # Add labels to the x and y axes
    ax.set_xlabel(feature_name_1)
    ax.set_ylabel(feature_name_2)

    legend = ["Class {}".format(i) for i in unique_labels]
    ax.legend(scatter.legend_elements()[0], legend, loc="upper right")

    return fig, ax


def violin_plot(X, ax=None, figsize=(8, 5)):
    """
    """
    if not(ax):
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    positions = range(X.shape[1])
    ax.violinplot(X, positions=positions, showmeans=True, showextrema=True, showmedians=True)

    ax.set_xticks(positions)
    ax.set_xticklabels(X.columns)
    ax.set_title("Feature Variation")
    ax.grid(color='gray', linestyle='--', linewidth=0.1)

    return fig, ax


def histogram(feature, labels, feature_name=None, bins=10, ax=None, figsize=(8,5)):
    """
    """
    if len(feature.shape) > 1:
        raise Exception("Too many features for historgram")

    if not(ax):
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    for label in np.unique(labels):
        ax.hist(feature[labels == label], 10, alpha=0.5, label=f'Label {label}')

    ax.set_ylabel('Frequency')

    if feature_name:
        ax.set_xlabel(feature_name)

    ax.legend(loc='upper right')

    return fig, ax


