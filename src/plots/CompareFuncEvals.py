from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import copy
import numpy as np

class CompareFuncEvals(PlotBase):
    def __init__(self, models=[], ratio=False):
        self.model_func_evals = {}
        self.min_func_evals = float("inf")
        self.max_func_evals = 0
        self.colors = ['b', 'y', 'r', 'g', 'c', 'o', 'k']
        self.epochs = None

        model_cnt = 0
        all_func_evals = []
        for model in models:
            model_name = type(model).__name__
            if model_name in self.model_func_evals:
                model_name = "{0}-{1}".format(model_name, model_cnt)
                model_cnt += 1

            self.model_func_evals[model_name] = {}

            self.model_func_evals[model_name]['func_evals'] = copy.copy(model.epoch_mean_func_evals)
            all_func_evals.append(copy.copy(model.epoch_mean_func_evals))

            self.min_func_evals = min(self.min_func_evals, min(model.epoch_mean_func_evals))
            self.max_func_evals = max(self.max_func_evals, max(model.epoch_mean_func_evals))

            if not(self.epochs):
                self.epochs = model.epochs

        if ratio:
            self.min_func_evals = float("inf")
            self.max_func_evals = 0
            sum_func_evals_per_epoch = np.array(all_func_evals).sum(axis=0)
            for model_name in self.model_func_evals:
                mean_func_evals_for_model = self.model_func_evals[model_name]['func_evals'] / sum_func_evals_per_epoch
                self.model_func_evals[model_name]['func_evals'] = mean_func_evals_for_model

                self.min_func_evals = min(self.min_func_evals, min(mean_func_evals_for_model))
                self.max_func_evals = max(self.max_func_evals, max(mean_func_evals_for_model))


    def plot(self, ax=None, figsize=(8,5), title=None, ymin=None, ymax=None):
        if ax:
            fig = None
        else:
            fig, axs = plt.subplots(1, 1, figsize=figsize)
            plt.style.use('seaborn')

        color_idx = 0

        for model_name in self.model_func_evals:
            axs.plot(self.model_func_evals[model_name]['func_evals'], self.colors[color_idx], label=model_name)
            color_idx += 1

        if ymin:
            y_minumum=ymin
        else:
            y_minumum=0.0

        if ymax:
            y_maximum=ymax
        else:
            y_maximum=self.max_func_evals * 1.1

        axs.set_title("Average Func Evals per Epoch")
        axs.set_xlim(1, self.epochs + 1)
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Func Evals')
        axs.set_ylim(y_minumum, y_maximum)
        axs.legend(loc='best')

        return fig, axs
