from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import numpy as np


class BoostingIterations(PlotBase):
    def __init__(self, model_name=None, train_losses=None, test_losses=None, scoring=None, experiment=None):
        if experiment:
            self.model_name = type(experiment.model).__name__
            self.train_scores = experiment.train_scores
            self.test_scores = experiment.test_scores
            self.scoring = experiment.scoring
        else:
            self.model_name = model_name
            self.train_scores = train_scores
            self.test_scores = test_scores
            self.scoring = scoring

    def plot(self, ax=None, figsize=(8,5)):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plt.style.use('seaborn')

        ax.set_title('Learning Curve (Boosting Iterations)')

        ax.plot(self.train_scores, 'b', label='train')
        ax.plot(self.test_scores, 'g', label='validation')

        ax.set_ylabel(self.scoring)
        ax.set_xlabel("Number of Learners")

        min_loss = np.concatenate([self.train_scores, self.test_scores], axis=0).min()
        max_loss = np.concatenate([self.train_scores, self.test_scores], axis=0).max()

        ax.set_ylim(min_loss - 0.1, max_loss + 0.1)
        ax.set_xlim(1, len(self.train_scores)+1)

        plt.legend(loc='best')

        return fig, ax