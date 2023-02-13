from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.tree import plot_tree

class DecisionTree(PlotBase):
    def __init__(self, data, model=None, experiment=None):
        if experiment:
            self.model = experiment.model
            self.model_name = type(experiment.model).__name__
        else:
            self.model = model
            self.model_name = type(model).__name__

        self.class_names = data.class_labeler.classes_
        self.feature_names = data.X_train.columns


    def plot(self, ax=None, figsize=(8,5), max_depth=7, fontsize=8, title=None):
        plt.figure(figsize=figsize)
        plt.title(title)

        artists = plot_tree(
            self.model,
            filled=True,
            rounded=True,
            class_names=self.class_names,
            feature_names=self.feature_names,
            max_depth=max_depth,
            fontsize=fontsize,
            node_ids=True
        )
        # plt.savefig("snli_decision_tree.pdf")
        # plt.show()
        return artists
        # return fig, ax
