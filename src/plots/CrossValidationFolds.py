from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import numpy as np

class CrossValidationFolds(PlotBase):
    def __init__(self, model_name=None, scoring=None, train_scores=None, test_scores=None, experiment=None):
        if experiment:
            self.model = experiment.model
            self.model_name = type(experiment.estimators[0]).__name__
            self.scoring = experiment.scoring
            self.train_scores = experiment.train_scores
            self.test_scores = experiment.test_scores
        else:
            self.model = None
            self.model_name = model_name
            self.scoring = scoring
            self.train_scores = train_scores
            self.test_scores = test_scores


    def plot(self, ax=None, figsize=(8,5), title=None):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.style.use('seaborn')

        if title:
            ax.set_title(title)
        else:
            ax.set_title('{0}'.format(self.model))

        ax.set_ylabel(self.scoring)
        ax.set_xlabel("Scores")

        min_score = np.concatenate([self.train_scores, self.test_scores], axis=0).min()
        max_score = np.concatenate([self.train_scores, self.test_scores], axis=0).max()

        ax.set_ylim(0.0, max_score + 0.1)
        x = np.arange(1, len(self.test_scores)+1)

        ax.bar(x, self.train_scores, color='b', width=0.25, alpha=0.6, label='train')
        ax.bar(x + 0.25, self.test_scores, color='g', width=0.25, alpha=0.6, label='validation')
        ax.legend(loc='upper right')

        ax.set_xlim(0.5, len(self.train_scores) + 0.5)
        ax.set_xticks(np.arange(1.1, len(self.train_scores)+1), np.arange(1, len(self.train_scores)+1), size='small')

        return fig, ax


