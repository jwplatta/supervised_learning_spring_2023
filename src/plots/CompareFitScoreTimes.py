from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import copy
import numpy as np

class CompareFitScoreTimes(PlotBase):
    def __init__(self, experiments=[]):
        self.model_times = {}
        self.min_fit_time = 0.0
        self.max_fit_time = 0.0
        self.min_score_time = 0.0
        self.max_score_time = 0.0
        self.colors = ['b', 'y', 'r', 'g', 'c', 'o', 'k']
        self.train_sizes = np.array([])

        for exp in experiments:
            self.model_times[type(exp.model).__name__] = {}
            self.model_times[type(exp.model).__name__]['score_times'] = copy.copy(exp.score_times)
            self.model_times[type(exp.model).__name__]['fit_times'] = copy.copy(exp.fit_times)

            if not(self.train_sizes.any()):
                self.train_sizes = copy.copy(exp.train_sizes)

            self.min_fit_time = min(self.min_fit_time, exp.fit_times.min())
            self.max_fit_time = max(self.max_fit_time, exp.fit_times.max())
            self.min_score_time = min(self.min_score_time, exp.score_times.min())
            self.max_score_time = max(self.max_score_time, exp.fit_times.max())


    def plot(self, ax=None, figsize=(8,5), title=None, ymin=None, ymax=None):
        if ax:
            fig = None
        else:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            plt.style.use('seaborn')

        color_idx = 0

        for model_name in self.model_times:
            axs[0].plot(self.model_times[model_name]['fit_times'], self.colors[color_idx], label=model_name)
            axs[1].plot(self.model_times[model_name]['score_times'], self.colors[color_idx], label=model_name)
            color_idx += 1

        if ymin:
            y_minumum=ymin
        else:
            y_minumum=self.min_fit_time - 0.1

        if ymax:
            y_maximum=ymax
        else:
            y_maximum=self.max_fit_time + 0.1

        axs[0].set_title("Fit Times")
        axs[0].set_xlim(0, len(self.train_sizes))
        axs[0].set_xticks(range(len(self.train_sizes)), self.train_sizes, rotation=45)
        axs[0].set_xlabel('Sample Count')
        axs[0].set_ylabel('Seconds')

        axs[0].set_ylim(y_minumum, self.max_fit_time + 0.1)
        axs[0].legend(loc='best')

        if ymin:
            y_minumum=ymin
        else:
            y_minumum=self.min_score_time - 0.1

        if ymax:
            y_maximum=ymax
        else:
            y_maximum=self.max_score_time + 0.1

        axs[1].set_title("Score Times")
        axs[1].set_xlim(0, len(self.train_sizes))
        axs[1].set_xticks(range(len(self.train_sizes)), self.train_sizes, rotation=45)
        axs[1].set_xlabel('Sample Count')
        axs[1].set_ylabel('Seconds')
        axs[1].set_ylim(y_minumum, y_maximum)
        axs[1].legend(loc='best')

        return fig, axs
