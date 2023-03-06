from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import copy
import numpy as np


class OptimizationFitTimes(PlotBase):
    def __init__(self, experiments=[], normalized=False):
        self.model_times = {}
        self.min_fit_time = 0.0
        self.max_fit_time = 0.0
        self.min_state_size = 0.0
        self.max_state_size = 0.0
        self.colors = ['b', 'y', 'r', 'g', 'c', 'o', 'k']
        self.state_sizes = experiments[0].state_sizes

        # if normalized:
        #     # NOTE: find the largest fit time
        #     largest_fit_time = max(for exp in experiments)

        for exp in experiments:
            model_name = type(exp.model).__name__
            self.model_times[model_name] = {}
            self.model_times[model_name]['fit_times'] = copy.copy(exp.fit_times)
            self.model_times[model_name]['state_sizes'] = copy.copy(exp.state_sizes)

            self.min_fit_time = min(self.min_fit_time, min(exp.fit_times))
            self.max_fit_time = max(self.max_fit_time, max(exp.fit_times))
            self.max_state_size = max(self.max_state_size, exp.state_sizes.max())


    def plot(self, ax=None, figsize=(8,5), title=None, ymin=None, ymax=None, xmax=None):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plt.style.use('seaborn')

        color_idx = 0

        for model_name in self.model_times:
            ax.plot(
              self.model_times[model_name]['fit_times'],
              self.colors[color_idx],
              label=model_name
            )
            color_idx += 1

        ax.set_ylabel('Seconds')
        if ymin:
            y_minumum=ymin
        else:
            y_minumum=self.min_fit_time - 0.1

        if ymax:
            y_maximum=ymax
        else:
            y_maximum=self.max_fit_time + 0.1

        ax.set_ylim(y_minumum, y_maximum)

        if xmax:
            x_maximum=xmax + 0.1
        else:
            x_maximum=max(self.state_sizes) + 0.1

        ax.set_xlim(0, len(self.state_sizes) + 0.1)
        ax.set_xticks(range(len(self.state_sizes)), self.state_sizes, rotation=45)
        ax.set_xlabel('State Size')

        ax.set_title("Fit Times")
        ax.legend(loc='best')

        return fig, ax