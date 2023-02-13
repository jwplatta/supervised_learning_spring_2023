from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ScatterPlot(PlotBase):
    def __init__(self, X, feature1_col, feature2_col, label_col, label_names=[], feature2_ticks=[]):
        if type(X) != pd.DataFrame:
            raise Exception("Expected pd.DataFrame")

        self.X = X
        self.feature1_col = feature1_col
        self.feature2_col = feature2_col
        self.feature2_ticks = feature2_ticks
        self.label_col = label_col
        self.label_names = label_names


    def plot(self, ax=None, figsize=(8,5), title=None):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(figsize=figsize)

        f1 = self.X[self.feature1_col]
        labels = self.X[self.label_col]

        f2 = self.X[self.feature2_col]
        scatter = ax.scatter(f1, f2, alpha=0.5, c=labels, cmap="rainbow")

        if self.feature2_ticks:
            ax.set_yticks(np.arange(len(self.feature2_ticks)), self.feature2_ticks)

        ax.set_ylabel(self.feature2_col)


        # Add labels to the x and y axes
        ax.set_xlabel(self.feature1_col)

        if self.label_names:
            legend = ["{}".format(i) for i in self.label_names]
        else:
            unique_labels = np.unique(labels)
            legend = ["Class {}".format(i) for i in unique_labels]

        ax.legend(scatter.legend_elements()[0], legend, loc="best")

        if title:
            ax.set_title(title)

        return fig, ax