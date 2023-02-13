from sklearn.metrics import classification_report

class ClassificationReport():
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

    def plot(self):
        if self.target_names:
            return classification_report(self.y_test, self.y_pred, target_names=self.target_names)
        else:
            return classification_report(self.y_test, self.y_pred)
