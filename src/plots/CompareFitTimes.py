from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import copy
import numpy as np

class CompareFitTimes(PlotBase):
    def __init__(self, experiments=[]):
        self.model_times = {}
        self.min_fit_time = 0.0
        self.max_fit_time = 0.0
        self.min_score_time = 0.0
        self.max_score_time = 0.0
        self.colors = ['b', 'y', 'r', 'g', 'c', 'o', 'k']
        self.train_sizes = np.array([])
        exp_cnt = 0

        for exp in experiments:
            if exp.name:
                exp_name = exp.name
            else:
                exp_name = "{0}-{1}".format(type(exp.model).__name__, exp_cnt)
                exp_cnt += 1

            self.model_times[exp_name] = {}
            self.model_times[exp_name]['fit_times'] = copy.copy(exp.fit_times)

            if not(self.train_sizes.any()):
                self.train_sizes = copy.copy(exp.train_sizes)

            self.min_fit_time = min(self.min_fit_time, exp.fit_times.min())
            self.max_fit_time = max(self.max_fit_time, exp.fit_times.max())


    def plot(self, ax=None, figsize=(8,5), title=None, ymin=None, ymax=None):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plt.style.use('seaborn')

        color_idx = 0

        for model_name in self.model_times:
            ax.plot(self.model_times[model_name]['fit_times'], self.colors[color_idx], label=model_name)
            color_idx += 1

        if ymin:
            y_minumum=ymin
        else:
            y_minumum=0.0

        if ymax:
            y_maximum=ymax
        else:
            y_maximum=self.max_fit_time * 1.1

        ax.set_title("Fit Times")
        ax.set_xlim(0, len(self.train_sizes))
        ax.set_xticks(range(len(self.train_sizes)), self.train_sizes, rotation=45)
        ax.set_xlabel('Sample Count')
        ax.set_ylabel('Seconds')

        ax.set_ylim(y_minumum, y_maximum)
        ax.legend(loc='best')

        return fig, ax
