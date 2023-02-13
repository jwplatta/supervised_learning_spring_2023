from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import clone


class DecisionBoundary(PlotBase):
    def __init__(self, model, X, feature1_col, feature2_col, label_col=None, labels=None):
        if type(X) != pd.DataFrame:
            raise Exception("Expected pd.DataFrame")

        self.model = clone(model)
        self.X = X
        self.feature1_col = feature1_col
        self.feature2_col = feature2_col
        self.label_col = label_col
        self.labels = labels
        self.Z = np.array([])


    def plot(self, ax=None, figsize=(8,5)):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(figsize=figsize)


        if self.label_col:
            self.labels = self.X[self.label_col]

        try:
            self.model.classes_
            print('Model is already trained.')
        except AttributeError as e:
            self.model.fit(self.X[[self.feature1_col, self.feature2_col]], self.labels)

        h = 0.5  # NOTE: step size in the mesh
        x_min, x_max = self.X[self.feature1_col].min() - 0.1, self.X[self.feature1_col].max() + 0.1
        y_min, y_max = self.X[self.feature2_col].min() - 0.1, self.X[self.feature2_col].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        if not(self.Z.any()):
            Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
            self.Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, self.Z, cmap='plasma', alpha=0.3)
        scatter = ax.scatter(
            self.X[self.feature1_col],
            self.X[self.feature2_col],
            c=self.labels,
            cmap='plasma',
            alpha=0.6
        )
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(self.feature1_col)
        ax.set_ylabel(self.feature2_col)

        unique_labels = np.unique(self.labels)

        legend = ["Class {}".format(i) for i in unique_labels]
        ax.legend(scatter.legend_elements()[0], legend, loc="best")

        return fig, ax