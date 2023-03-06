from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import numpy as np
import copy


class FitnessValidationCurve(PlotBase):
    def __init__(self, experiment):
        self.problem_name = type(experiment.problem).__name__
        self.model_name = type(experiment.model).__name__
        self.model = experiment.model
        self.param_name = experiment.param_name
        self.param_range = copy.copy(experiment.param_range)
        self.best_fitnesses = copy.copy(experiment.best_fitnesses)


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

        ax.plot(self.best_fitnesses, 'b')

        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel('Fitness')

        ax.set_xlabel(self.param_name)

        min_fitness = np.min(self.best_fitnesses)
        max_fitness = np.max(self.best_fitnesses)

        ax.set_ylim(min_fitness * 0.9, max_fitness * 1.1)

        ax.set_xlim(0, len(self.param_range))
        ax.set_xticks(range(len(self.param_range)), self.param_range, size='small', rotation=45)

        return fig, ax
