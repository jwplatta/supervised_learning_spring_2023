from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Heatmap(PlotBase):
    def __init__(self, X):
        if type(X) != pd.DataFrame:
            raise Exception("Expected pd.DataFrame")

        self.X = X


    def plot(self, ax=None, figsize=(8,8)):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(self.X, cmap='magma')
        cbar = ax.figure.colorbar(im, ax=ax)

        ax.set_yticks(np.arange(len(self.X.columns)))
        ax.set_yticklabels(self.X.columns, rotation=45)

        ax.set_xticks(np.arange(len(self.X.columns)))
        ax.set_xticklabels(self.X.columns, rotation=45)

        # NOTE: Annotations
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                text = ax.text(j, i, round(self.X.iloc[i, j], 2), ha="center", va="center", color="w")

        return fig, ax

