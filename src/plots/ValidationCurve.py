from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import numpy as np
import copy


class ValidationCurve(PlotBase):
    def __init__(self, model_name=None, scoring=None, param_name=None, param_range=None, train_scores=None, test_scores=None, experiment=None):
        if experiment:
            self.model = experiment.model
            self.model_name = type(experiment.model).__name__
            self.param_name = experiment.param_name
            self.param_range = copy.copy(experiment.param_range)
            self.scoring = experiment.scoring
            self.train_scores = copy.copy(experiment.train_scores)
            self.train_scores = np.nan_to_num(self.train_scores, nan=0.0)
            self.test_scores = copy.copy(experiment.test_scores)
            self.text_scores = np.nan_to_num(self.test_scores, nan=0.0)
        else:
            self.model = None
            self.model_name = model_name
            self.param_name = param_name
            self.param_range = param_range
            self.scoring = scoring
            self.train_scores = train_scores
            self.test_scores = test_scores


    def plot(self, ax=None, figsize=(8,5), title=None, ylabel=None):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plt.style.use('seaborn')

        if title:
            ax.set_title(title)
        else:
            ax.set_title('{0}'.format(self.model))

        ax.plot(self.train_scores, 'b', label='train')
        ax.plot(self.test_scores, 'g', label='validation')

        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(self.scoring)

        ax.set_xlabel(self.param_name)


        min_loss = np.concatenate([self.train_scores, self.test_scores], axis=0).min()
        max_loss = np.concatenate([self.train_scores, self.test_scores], axis=0).max()

        ax.set_ylim(min_loss - 0.1, max_loss + 0.1)

        ax.set_xlim(0, len(self.param_range))
        ax.set_xticks(range(len(self.param_range)), self.param_range, size='small', rotation=45)

        ax.legend(loc='upper right')

        return fig, ax
