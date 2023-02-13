from .PlotBase import PlotBase
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ConfusionMatrix(PlotBase):
    def __init__(self, model_name=None, y_pred=None, y_test=None, class_labels=None, target_names=None, experiment=None):
        if experiment:
            self.model_name = type(experiment.model).__name__
            self.y_pred = experiment.y_pred
            self.y_test = experiment.y_test
            self.class_labels = experiment.class_labels
        else:
            self.model_name = model_name
            self.y_pred = y_pred
            self.y_test = y_test
            self.class_labels = class_labels

        self.target_names = target_names

    def plot(self, title=None, ax=None):

        if ax:
            disp = ConfusionMatrixDisplay.from_predictions(
              y_true=self.y_test,
              y_pred=self.y_pred,
              ax=ax,
              cmap='magma',
              display_labels=self.target_names
            )
            ax.grid(False)
            if title:
                ax.set_title(title)
        else:
            disp = ConfusionMatrixDisplay.from_predictions(
              y_true=self.y_test,
              y_pred=self.y_pred,
              cmap='magma',
              display_labels=self.target_names
            )
            plt.grid(False)
            if title:
                disp.ax_.set_title(title)

        return disp.figure_, disp.ax_

