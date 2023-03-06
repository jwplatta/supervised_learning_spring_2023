from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import numpy as np
import copy


class FuncCallsByStateSizeComparison(PlotBase):
    def __init__(self, experiments=[], ratio=None):
        self.experiment_cnt = len(experiments)
        self.colors = ['b', 'y', 'r', 'g', 'c', 'o', 'k']
        self.max_metric = 0.0
        self.min_metric = float('inf')
        self.compare_attr = "state_sizes"
        self.metric_name = "Function Evals"

        self.model_results = {}
        for exp in experiments:
            model_name = type(exp.model).__name__
            self.model_results[model_name] = {}

            function_calls = [curve[-1,-1] for curve in exp.fitness_curves]

            if ratio == 'func_evals':
                self.metric_name = "Func evals to State Size"
                metric = [function_calls[idx] / state_size for idx, state_size in enumerate(exp.state_sizes)]
            elif ratio == 'fitness':
                self.metric_name = "Fitness Score to State Size"
                metric = [exp.best_fitnesses[idx] / state_size for idx, state_size in enumerate(exp.state_sizes)]
            else:
                metric = function_calls

            self.model_results[model_name]['metric'] = metric
            self.model_results[model_name]['state_sizes'] = copy.copy(exp.state_sizes)
            self.max_metric = max(self.max_metric, max(metric))
            self.min_metric = min(self.min_metric, min(metric))


    def plot(self, ax=None, figsize=(8,5), title=None, xlabel=None, ylabel=None, xlim=None, ):
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
            ax.set_title("State Size Comparison")

        for idx, model_name in enumerate(self.model_results):
            state_sizes = self.model_results[model_name]['state_sizes']
            ax.plot(
              self.model_results[model_name]['metric'],
              self.colors[idx],
              label=model_name
            )

        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(self.metric_name)

        ax.set_ylim(self.min_metric * 0.9, self.max_metric * 1.1)

        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel("State Size")

        ax.set_xticks(range(len(state_sizes)), state_sizes)
        ax.legend(loc='best')

        return fig, ax
