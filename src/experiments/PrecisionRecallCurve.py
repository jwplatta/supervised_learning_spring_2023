from .ExperimentBase import ExperimentBase
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class PrecisionRecallCurve(ExperimentBase):
    def __init__(self, model, random_state=42, n_splits=5):
        self.model = clone(model)
        self.random_state = random_state
        self.n_splits = n_splits
        self.estimators = []
        self.test_size = 0.3

        self.train_recall_scores = np.array([])
        self.train_precision_scores = np.array([])
        self.train_thresholds = np.array([])

        self.test_recall_scores = np.array([])
        self.test_precision_scores = np.array([])
        self.test_thresholds = np.array([])


    def run(self, X, y):
        if self.train_recall_scores.any() or self.train_precision_scores.any():
            raise Exception('Experiment already has results')

        if type(self.model).__name__ == 'NeuralNetworkClassifier':
            n_jobs = 1
        else:
            n_jobs = 3

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            model_clone = clone(self.model)
            self.estimators.append(model_clone)

            model_clone.fit(X_train, y_train)

            if type(model_clone) in [DecisionTreeClassifier, KNeighborsClassifier]:
                y_scores = model_clone.predict_proba(X_train)[:, 1]
            else:
                y_scores = model_clone.decision_function(X_train)

            precision, recall, thresholds = precision_recall_curve(y_train, y_scores)
            self.train_recall_scores = recall
            self.train_precision_scores = precision
            self.train_thresholds = thresholds


            if type(model_clone) in [DecisionTreeClassifier, KNeighborsClassifier]:
                y_scores = model_clone.predict_proba(X_test)[:, 1]
            else:
                y_scores = model_clone.decision_function(X_test)
            precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
            self.test_recall_scores = recall
            self.test_precision_scores = precision
            self.test_thresholds = thresholds

            return self
        except Exception as e:
            print(e)
            return False


        def __repr__(self):
            return '<{0} model: {1}, random_state: {2}, n_split: {3}>'.format(
                type(self).__name__,
                type(self.model).__name__,
                self.random_state,
                self.n_splits
            )