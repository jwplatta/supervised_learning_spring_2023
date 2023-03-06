from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import numpy as np
import copy


class FitnessCurve(PlotBase):
    def __init__(self, fitness_curve):
        self.fitness_curve = fitness_curve
        self.fitnesses = fitness_curve[:, 0]
        self.func_evals = fitness_curve[:, 1]


    def plot(self, ax=None, figsize=(8,5), title=None, ylabel=None):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plt.style.use('seaborn')

        if title:
            ax.set_title(title)
        else:
            ax.set_title('Fitness Curve')

        ax.plot(self.func_evals, self.fitnesses, 'b')

        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel('Fitness')

        ax.set_xlabel('Function Evals')

        min_fitness = np.min(self.fitnesses)
        max_fitness = np.max(self.fitnesses)

        ax.set_ylim(min_fitness * 0.99, max_fitness * 1.01)

        max_func_evals = self.func_evals.max()
        ax.set_xlim(0, max_func_evals + 1)
        # ax.set_xticks(range(len(self.param_range)), self.param_range, size='small', rotation=45)

        return fig, ax
