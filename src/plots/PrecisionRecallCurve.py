from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import numpy as np
import copy


class PrecisionRecallCurve(PlotBase):
    def __init__(self,
        model_name=None,
        train_recall_scores=np.array([]),
        train_precision_scores=np.array([]),
        train_thresholds=np.array([]),
        test_recall_scores=np.array([]),
        test_precision_scores=np.array([]),
        test_thresholds=np.array([]),
        experiment=None):

        if experiment:
            self.model_name = type(experiment.model).__name__
            self.train_recall_scores = experiment.train_recall_scores
            self.train_precision_scores = experiment.train_precision_scores
            self.train_thresholds = experiment.train_thresholds
            self.test_recall_scores = experiment.test_recall_scores
            self.test_precision_scores = experiment.test_precision_scores
            self.test_thresholds = experiment.test_thresholds
        else:
            self.model_name = model_name
            self.train_recall_scores = train_recall_scores
            self.train_precision_scores = train_precision_scores
            self.train_thresholds = train_thresholds
            self.test_recall_scores = test_recall_scores
            self.test_precision_scores = test_precision_scores
            self.test_thresholds = test_thresholds


    def plot(self, ax=None, figsize=(8,5), title=None, show_thresholds=True):
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plt.style.use('seaborn')

        if title:
            ax.set_title(title)
        else:
            ax.set_title('Precision Recall Curve for {0}'.format(self.model_name))

        # Plot the precision-recall curve
        ax.plot(self.train_recall_scores, self.train_precision_scores, 'b', label='train')
        ax.plot(self.test_recall_scores, self.test_precision_scores, 'g', label='validation')

        # Plot the thresholds
        # ax.plot(self.train_recall_scores[:-1], self.train_thresholds, linestyle='--', label='Train Thresholds')
        # ax.plot(self.test_recall_scores[:-1], self.test_thresholds, linestyle='--', label='Test Thresholds')
        # Plot decision thresholds on the precision-recall curve
        # thresholds = np.append(thresholds, 1)
        if show_thresholds:
            plt.vlines(
                self.train_recall_scores[:-1],
                0,
                self.train_precision_scores[:-1],
                color='b',
                linestyle='dotted',
                alpha=0.2
            )
            for i, threshold in enumerate(self.train_thresholds):
                if i % 5 == 0:
                    plt.text(self.train_recall_scores[i], self.train_precision_scores[i], '%.2f' % threshold, color='gray', ha='left', va='bottom')

            plt.vlines(
                self.test_recall_scores[:-1],
                0,
                self.test_precision_scores[:-1],
                color='g',
                linestyle='dotted',
                alpha=0.2
            )
            for i, threshold in enumerate(self.test_thresholds):
                if i % 5 == 0:
                    plt.text(self.test_recall_scores[i], self.test_precision_scores[i], '%.2f' % threshold, color='gray', ha='left', va='bottom')


        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        min_loss = np.concatenate([self.train_precision_scores, self.test_precision_scores], axis=0).min()
        max_loss = np.concatenate([self.train_precision_scores, self.test_precision_scores], axis=0).max()

        ax.set_ylim(min_loss, max_loss + 0.1)

        # ax.set_xlim(0, len(self.param_range))
        # ax.set_xticks(range(len(self.param_range)), self.param_range, size='small', rotation=45)

        ax.legend(loc='best')

        return fig, ax
