from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import copy
import numpy as np

class LearningCurve(PlotBase):
    def __init__(self, model_name=None, scoring=None, train_scores=None, test_scores=None, train_sizes=None, experiment=None):
        if experiment:
            self.model = experiment.model
            self.model_name = type(experiment.model).__name__
            self.scoring = experiment.scoring
            self.train_scores = np.nan_to_num(copy.copy(experiment.train_scores), nan=0.0)
            self.test_scores = np.nan_to_num(copy.copy(experiment.test_scores), nan=0.0)
            self.train_sizes = copy.copy(experiment.train_sizes)
        else:
            self.model = None
            self.model_name = model_name
            self.scoring = scoring
            self.train_scores = train_scores
            self.test_scores = test_scores
            self.train_sizes = train_sizes


    def plot(self, ax=None, figsize=(8,5), title=None, ymin=None, ymax=None):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.style.use('seaborn')

        if title:
            ax.set_title(title)
        else:
            ax.set_title('{0}'.format(self.model))

        ax.plot(self.train_scores, 'b', label='train')
        ax.plot(self.test_scores, 'g', label='validation')

        ax.set_xlim(0, len(self.train_sizes))
        ax.set_xticks(range(len(self.train_sizes)), self.train_sizes, rotation=45)
        ax.set_xlabel('Sample Count')

        if ymin or ymax:
            ax.set_ylim(ymin, ymax)
        else:
            min_score = np.concatenate([self.train_scores, self.test_scores], axis=0).min()
            max_score = np.concatenate([self.train_scores, self.test_scores], axis=0).max()
            ax.set_ylim(min_score - 0.1, max_score + 0.1)

        ax.set_ylabel(self.scoring)

        ax.legend(loc='upper right')

        return fig, ax
