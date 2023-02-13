from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ScatterPlot3D(PlotBase):
    def __init__(self, X, feature1_col, feature2_col, feature3_col, label_col):
        if type(X) != pd.DataFrame:
            raise Exception("Expected pd.DataFrame")

        self.X = X
        self.feature1_col = feature1_col
        self.feature2_col = feature2_col
        self.feature3_col = feature3_col
        self.label_col = label_col


    def plot(self, ax=None, figsize=(8,5)):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': '3d'})

        f1 = self.X[self.feature1_col]
        f2 = self.X[self.feature2_col]
        f3 = self.X[self.feature3_col]

        labels = self.X[self.label_col]

        scatter = ax.scatter(f1, f2, f3, alpha=0.5, c=labels, cmap="rainbow")

        # Get the unique classes
        unique_labels = np.unique(labels)

        # Add labels to the x and y axes
        ax.set_xlabel(self.feature1_col)
        ax.set_ylabel(self.feature2_col)
        ax.set_zlabel(self.feature3_col)

        legend = ["Class {}".format(i) for i in unique_labels]
        ax.legend(scatter.legend_elements()[0], legend, loc="best")

        plt.legend()
        plt.show()

        return fig, ax