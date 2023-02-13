from .ExperimentBase import ExperimentBase
from sklearn.model_selection import validation_curve, KFold
from sklearn.base import clone
import warnings
import numpy as np

class ValidationCurve(ExperimentBase):
    def __init__(self, model, param_name, param_range, scoring='accuracy', random_state=None, n_splits=5):
        self.model = clone(model)
        self.param_name = param_name
        self.param_range = param_range
        self.scoring = scoring
        self.train_scores = None
        self.test_scores = None
        self.random_state = random_state
        self.n_splits = n_splits


    def run(self, X, y):
        if self.train_scores or self.test_scores:
            raise Exception('Experiment already has results')

        try:
            if type(self.model).__name__ == 'NeuralNetworkClassifier':
                n_jobs = 1
            else:
                n_jobs = 3

            train_scores, test_scores = validation_curve(
                self.model,
                X,
                y,
                cv=KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True),
                scoring=self.scoring,
                param_name=self.param_name,
                param_range=self.param_range,
                verbose=5,
                n_jobs=n_jobs
            )
            train_scores = np.nan_to_num(train_scores, nan=0.0)
            test_scores = np.nan_to_num(test_scores, nan=0.0)

            self.train_scores = train_scores.mean(axis=1)
            self.test_scores = test_scores.mean(axis=1)

            return True
        except Exception as e:
            print(e)
            return False


        def __repr__(self):
            return '<{0} model: {1}, scoring: {2}, param_name: {3}, param_range: {4}, random_state: {5}, n_split: {6}>'.format(
                type(self).__name__,
                type(self.model).__name__,
                self.scoring,
                self.param_name,
                self.param_range,
                self.random_state,
                self.n_splits
            )