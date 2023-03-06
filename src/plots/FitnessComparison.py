from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import numpy as np
import copy


class FitnessComparison(PlotBase):
    def __init__(self, experiments=[], ratio=False):
        self.experiment_cnt = len(experiments)
        self.colors = ['b', 'y', 'r', 'g', 'c', 'o', 'k']
        self.max_fitness = 0.0
        self.min_fitness = float('inf')
        self.compare_attr = experiments[0].param_name
        self.param_range = experiments[0].param_range

        self.model_results = {}
        model_cnt = 1
        for exp in experiments:
            if exp.name:
                experiment_name = exp.name
            else:
                experiment_name = type(exp.model).__name__

            if experiment_name in self.model_results:
                experiment_name = "{}-{}".format(experiment_name, model_cnt)
                model_cnt += 1

            self.model_results[experiment_name] = {}

            self.model_results[experiment_name]['best_fitnesses'] = copy.copy(exp.best_fitnesses)
            self.model_results[experiment_name][self.compare_attr] = copy.copy(exp.param_range)

            self.max_fitness = max(self.max_fitness, max(exp.best_fitnesses))
            self.min_fitness = min(self.min_fitness, min(exp.best_fitnesses))


    def plot(self, ax=None, figsize=(8,5), title=None, ylabel=None):
        if self.experiment_cnt == 0:
            return None, None

        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plt.style.use('seaborn')

        if title:
            ax.set_title(title)
        else:
            ax.set_title("Compare: {0}".format(self.compare_attr))

        for idx, experiment_name in enumerate(self.model_results):
            ax.plot(
            #   self.model_results[model_name][self.compare_attr],
              self.model_results[experiment_name]['best_fitnesses'],
              self.colors[idx],
              label=experiment_name
            )

        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel('Fitness')

        ax.set_ylim(self.min_fitness * 0.9, self.max_fitness * 1.1)

        ax.set_xlabel(self.compare_attr)

        # ax.set_xlim(0, len(self.param_range))
        ax.set_xticks(range(len(self.param_range)), self.param_range, size='small', rotation=45)

        ax.legend(loc='best')

        return fig, ax
