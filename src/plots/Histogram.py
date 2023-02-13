from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Histogram(PlotBase):
    def __init__(self, feature, labels=pd.Series(dtype=object), feature_name=None, xticks=[], label_names=[], bins=10, title=None, legend=False):
        self.feature = feature
        self.labels = labels
        self.feature_name = feature_name
        self.label_names = label_names
        self.bins = bins
        self.legend = legend
        self.title = title
        self.xticks = xticks


    def plot(self, ax=None, figsize=(8,5)):
        if len(self.feature.shape) > 1:
            raise Exception("Too many features for historgram")

        if not(ax):
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None

        if self.labels.any():
            for label in np.unique(self.labels):
                ax.hist(self.feature[self.labels == label], self.bins, alpha=0.5, label=f'{label}')

            if self.xticks:
                ax.set_xticks(range(np.unique(self.xticks).shape[0]))
                ax.set_xticklabels(self.xticks)

            if self.label_names:
                legend = ["{0}".format(i) for i in self.label_names]
                ax.legend(legend, loc="best")
        else:
            if self.label_names:
                legend = ["{0}".format(i) for i in self.label_names]
                ax.legend(legend, loc="best")
            else:
                ax.legend(loc="best")

            ax.hist(self.feature, self.bins, alpha=0.5)

        ax.set_ylabel('Frequency')

        if self.title:
            ax.set_title(self.title)

        if self.feature_name:
            ax.set_xlabel(self.feature_name)

        return fig, ax


